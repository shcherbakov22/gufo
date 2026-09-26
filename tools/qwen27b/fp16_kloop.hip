#define ENGINE_ENABLE_HIP 1
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <hip/hip_runtime.h>
#include "src/models/qwen/hip/kernels/prefill_fp16.hip"

using gufo::core::GgmlType;

namespace {

// Pure K-loop probe at the production shape (256x256, BK=4, same s_a/s_b layout
// and swizzle, no decode/commit/global fetch). MWA/MWB are the warp split, so
// the LDS fragment bytes per MMA, 512*(1/WRS + 1/WTS), can be traded against the
// number of warps available to hide MMA latency. Mode 3 replaces the fragments
// with registers to isolate the LDS feed from everything else.
constexpr int PBM = 256, PBN = 256, PBK = 4, PKS = 16;
constexpr int PRS = PBM / 16, PTS = PBN / 16;

template<int Mode, int MWA = 8, int MWB = 4>
__global__ void __launch_bounds__(1024) PureKernel(
    const void* __restrict__, const short* __restrict__,
    float* __restrict__ y, int batch, int m, int k) {
  constexpr int NT = MWA * MWB * 32;
  constexpr int WRS = PRS / MWA, WTS = PTS / MWB;
  __shared__ short s_a[PBK][PRS][16][PKS];
  __shared__ short s_b[PBK][PTS][16][PKS];
  const int tid = (int)threadIdx.x;
  const int wave = tid / 32, lane = tid % 32, sl = lane & 15;
  const int wr = wave / MWB, wt = wave % MWB;
  {
    const int na = PBK * PRS * 16 * PKS, nb = PBK * PTS * 16 * PKS;
    for (int i = tid; i < na; i += NT)
      (&s_a[0][0][0][0])[i] = (short)((i & 15) - 8);
    for (int i = tid; i < nb; i += NT)
      (&s_b[0][0][0][0])[i] = (short)((i & 15) - 8);
  }
  __syncthreads();
  const int ksteps = k / 16;
  gufo::hip::v8f acc[WRS][WTS] = {};
  gufo::hip::v16bf ca, cb;
  for (int i = 0; i < 16; ++i) {
    ca[i] = (_Float16)(i - 8);
    cb[i] = (_Float16)(8 - i);
  }
  for (int ks0 = 0; ks0 < ksteps; ks0 += PBK) {
    __syncthreads();
#pragma unroll
    for (int ks = 0; ks < PBK; ++ks) {
      // Mode 5 reuses one sub-stage's fragments for all four, cutting the LDS
      // request count 4x to test whether requests or bytes set the feed cost.
      const int lks = (Mode == 5) ? 0 : ks;
      gufo::hip::v16bf a[WRS];
#pragma unroll
      for (int i = 0; i < WRS; ++i)
        a[i] = Mode == 3
                   ? ca
                   : gufo::hip::LoadSwizzled(
                         &s_a[lks][wr * WRS + i][sl ^ (lks & 3)][0]);
#pragma unroll
      for (int j = 0; j < WTS; ++j) {
        const gufo::hip::v16bf b =
            Mode == 3 ? cb
                      : gufo::hip::LoadSwizzled(
                            &s_b[lks][wt * WTS + j][sl ^ (lks & 3)][0]);
#pragma unroll
        for (int i = 0; i < WRS; ++i)
          acc[i][j] = gufo::hip::WmmaHalf(b, a[i], acc[i][j]);
      }
    }
    if (ks0 == 0x40000000)
      (&s_b[0][0][0][0])[tid & 15] = (short)acc[0][0][0];
    __syncthreads();
  }
  unsigned row_tile = blockIdx.y, token_tile = blockIdx.x;
  {
    const unsigned gs = 3, group = 1U << gs;
    const unsigned first = (blockIdx.y >> gs) << gs,
                   within = (blockIdx.y & (group - 1)) * gridDim.x +
                            blockIdx.x,
                   rows = min(group, gridDim.y - first);
    row_tile = first + within % rows;
    token_tile = within / rows;
  }
  const int r_block = (int)row_tile * PBM, t_block = (int)token_tile * PBN;
  const int hi = lane / 16;
#pragma unroll
  for (int i = 0; i < WRS; ++i)
#pragma unroll
    for (int j = 0; j < WTS; ++j) {
      const int row = r_block + ((wr * WRS + i) * 16) + sl;
      const int t0 = t_block + ((wt * WTS + j) * 16);
#pragma unroll
      for (int l = 0; l < 8; ++l)
        y[std::size_t(t0 + 2 * l + hi) * m + row] = acc[i][j][l];
    }
}

void LaunchProd(const void* w, const short* x, float* y, int batch, int m,
                int k) {
  const dim3 grid((batch + 255) / 256, (m + 255) / 256);
  gufo::hip::HalfPrefillGemmKernel<256, 256, 8, 4, GgmlType::kQ4_K,
                                   gufo::hip::HalfEpilogue::kStore, true,
                                   GgmlType::kQ4_K, 4, 0>
      <<<grid, 1024>>>(w, x, y, batch, m, k, nullptr);
}

template<int Mode, int MWA, int MWB>
void LaunchPure(const void* w, const short* x, float* y, int batch, int m,
                int k) {
  const dim3 grid((batch + 255) / 256, (m + 255) / 256);
  PureKernel<Mode, MWA, MWB><<<grid, MWA * MWB * 32>>>(w, x, y, batch, m, k);
}

constexpr int kArms = 5;
const char* kNames[kArms] = {"ctrl A", "LDS 48 req/warp",
                             "LDS 12 req/warp", "no LDS feed", "ctrl B"};
}  // namespace

int main(int argc, char** argv) {
  const int m = argc > 1 ? std::atoi(argv[1]) : 17408;
  const int k = argc > 2 ? std::atoi(argv[2]) : 5120;
  const int batch = argc > 3 ? std::atoi(argv[3]) : 2048;
  const int rounds = argc > 4 ? std::atoi(argv[4]) : 4;
  std::printf("m=%d k=%d batch=%d rounds=%d\n", m, k, batch, rounds);
  std::fflush(stdout);
  const std::size_t rb = (std::size_t)(k / 256) * 144;
  void *w = nullptr, *x = nullptr, *y = nullptr;
  const std::size_t yN = (std::size_t)batch * m;
  if (hipMalloc(&w, rb * m) != hipSuccess) return 1;
  if (hipMalloc(&x, (std::size_t)batch * k * 2) != hipSuccess) return 1;
  if (hipMalloc(&y, yN * 4) != hipSuccess) return 1;
  hipMemset(w, 0, rb * m);
  hipMemset(x, 0, (std::size_t)batch * k * 2);
  hipMemset(y, 0, yN * 4);
  LaunchPure<5, 8, 4>(w, (const short*)x, (float*)y, batch, m, k);
  hipError_t e = hipDeviceSynchronize();
  if (e != hipSuccess) {
    std::printf("launch failed: %s\n", hipGetErrorString(e));
    return 1;
  }
  const double fl = 2.0 * batch * (double)m * (double)k;
  std::vector<double> sv[kArms];
  for (int r = 0; r < rounds; ++r) {
    const bool fwd = (r % 2) == 0;
    for (int idx = 0; idx < kArms; ++idx) {
      const int a = fwd ? idx : (kArms - 1 - idx);
      double ms = 0;
      for (int i = 0; i < 3; ++i) {
        hipDeviceSynchronize();
        const auto s0 = std::chrono::steady_clock::now();
        switch (a) {
          case 0: LaunchProd(w, (const short*)x, (float*)y, batch, m, k); break;
          case 1: LaunchPure<0, 8, 4>(w, (const short*)x, (float*)y, batch, m, k); break;
          case 2: LaunchPure<5, 8, 4>(w, (const short*)x, (float*)y, batch, m, k); break;
          case 3: LaunchPure<3, 8, 4>(w, (const short*)x, (float*)y, batch, m, k); break;
          case 4: LaunchProd(w, (const short*)x, (float*)y, batch, m, k); break;
        }
        hipDeviceSynchronize();
        const auto s1 = std::chrono::steady_clock::now();
        ms += std::chrono::duration<double, std::milli>(s1 - s0).count();
      }
      sv[a].push_back(ms / 3.0);
    }
  }
  double a0 = 0;
  for (int a = 0; a < kArms; ++a) {
    std::vector<double> v = sv[a];
    std::sort(v.begin(), v.end());
    const double med = v[v.size() / 2];
    if (a == 0) a0 = med;
    std::printf("%-18s median=%8.3f min=%8.3f max=%8.3f tflops=%6.2f (%+.1f%%)\n",
                kNames[a], med, v.front(), v.back(), fl / (med * 1e9),
                100.0 * (a0 - med) / a0);
    std::fflush(stdout);
  }
  std::printf("done\n");
  return 0;
}