#define ENGINE_ENABLE_HIP 1
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <hip/hip_runtime.h>
#include "src/models/qwen/hip/kernels/prefill_fp16.hip"

using gufo::core::GgmlType;

namespace {

// Warp-specialised kQ4_K prefill GEMM. 16 producer warps fill one of three
// LDS slots while 16 consumer warps run WMMA on another, so the weight decode
// overlaps live matrix work instead of sitting in the post-barrier bubble.
// BK=1 with three slots is what fits: at BK=2 a double-buffered A+B tile is
// exactly 64 KiB and there is no room left for the handoff counters.
constexpr int WS_BM = 256, WS_BN = 256;
constexpr int WS_RS = WS_BM / 16, WS_TS = WS_BN / 16;
constexpr int WS_WM = 4, WS_WN = 4;
constexpr int WS_WRS = WS_RS / WS_WM, WS_WTS = WS_TS / WS_WN;
constexpr int WS_SLOTS = 3;
constexpr int WS_AW = 8;   // producer warps for A (256 rows)
constexpr unsigned WS_PROD = (WS_AW + 8) * 32;  // 16 producer warps

template<int Sync>
__global__ void __launch_bounds__(1024) WarpSpecKernel(
    const void* __restrict__ w, const short* __restrict__ x,
    float* __restrict__ y, int batch, int m, int k) {
  __shared__ short sa[WS_SLOTS][WS_RS][16][16];
  __shared__ short sb[WS_SLOTS][WS_TS][16][16];
  __shared__ unsigned gate[WS_SLOTS];

  const int tid = (int)threadIdx.x;
  if (tid < WS_SLOTS) gate[tid] = 0u;
  __syncthreads();

  const int ksteps = k / 16;
  unsigned row_tile = blockIdx.y, token_tile = blockIdx.x;
  {
    const unsigned group_shift = 3, group = 1U << group_shift;
    const unsigned first = (blockIdx.y >> group_shift) << group_shift,
                   within = (blockIdx.y & (group - 1)) * gridDim.x +
                            blockIdx.x,
                   rows = min(group, gridDim.y - first);
    row_tile = first + within % rows;
    token_tile = within / rows;
  }
  const int r_block = (int)row_tile * WS_BM;
  const int t_block = (int)token_tile * WS_BN;

  if (tid < (int)WS_PROD) {
    // ---------------- producers ----------------
    const int lane = tid & 31;
    const bool is_a = tid < WS_AW * 32;
    const int row = is_a ? tid : tid - WS_AW * 32;
    const std::size_t row_bytes =
        gufo::hip::QuantRowBytes(GgmlType::kQ4_K, k);
    for (int n = 0; n < ksteps; ++n) {
      const int s = n % WS_SLOTS, r = n / WS_SLOTS;
      if constexpr (Sync != 0) {
        if (lane == 0)
          while (*(volatile unsigned*)&gate[s] < 32u * unsigned(r)) {}
        __syncwarp();
        __threadfence_block();
      }
      if (is_a) {
        const void* src = static_cast<const char*>(w) +
                          std::size_t(r_block + row) * row_bytes;
        auto raw = gufo::hip::LoadRaw<GgmlType::kQ4_K>(src, unsigned(n));
        auto q = gufo::hip::DecodeRaw<GgmlType::kQ4_K>(raw);
        union {
          gufo::hip::v16bf f;
          uint4 u[2];
        } packed;
#pragma unroll
        for (int j = 0; j < 16; ++j)
          packed.f[j] = static_cast<_Float16>(__fsub_rn(
              __fmul_rn(q.scale, static_cast<float>(q.q[j])), q.offset));
        gufo::hip::StoreSwizzled(&sa[s][row >> 4][row & 15][0], packed.u[0],
                                 packed.u[1], unsigned(row));
      } else {
        const short* src = x + std::size_t(t_block + row) * k +
                           std::size_t(n) * 16;
        gufo::hip::StoreSwizzled(&sb[s][row >> 4][row & 15][0],
                                 *reinterpret_cast<const uint4*>(src),
                                 *reinterpret_cast<const uint4*>(src + 8),
                                 unsigned(row));
      }
      if constexpr (Sync != 0) {
        __threadfence_block();
        __syncwarp();
        if (lane == 0) atomicAdd(&gate[s], 1u);
      }
    }
  } else {
    // ---------------- consumers ----------------
    const int cw = (tid - (int)WS_PROD) >> 5, lane = tid & 31, sl = lane & 15;
    const int wr = cw / WS_WN, wt = cw % WS_WN;
    gufo::hip::v8f acc[WS_WRS][WS_WTS] = {};
    for (int n = 0; n < ksteps; ++n) {
      const int s = n % WS_SLOTS, r = n / WS_SLOTS;
      const unsigned target = 32u * unsigned(r) + 16u;
      if constexpr (Sync != 0) {
        if (lane == 0)
          while (*(volatile unsigned*)&gate[s] < target) {}
        __syncwarp();
        __threadfence_block();
      }
      gufo::hip::v16bf av[WS_WRS];
#pragma unroll
      for (int i = 0; i < WS_WRS; ++i)
        av[i] = gufo::hip::LoadSwizzled(&sa[s][wr * WS_WRS + i][sl][0]);
#pragma unroll
      for (int j = 0; j < WS_WTS; ++j) {
        const gufo::hip::v16bf bv =
            gufo::hip::LoadSwizzled(&sb[s][wt * WS_WTS + j][sl][0]);
#pragma unroll
        for (int i = 0; i < WS_WRS; ++i)
          acc[i][j] = gufo::hip::WmmaHalf(bv, av[i], acc[i][j]);
      }
      if constexpr (Sync != 0) {
        __threadfence_block();
        __syncwarp();
        if (lane == 0) atomicAdd(&gate[s], 1u);
      }
    }
    const int hi = lane / 16;
#pragma unroll
    for (int i = 0; i < WS_WRS; ++i)
#pragma unroll
      for (int j = 0; j < WS_WTS; ++j) {
        const int row = r_block + ((wr * WS_WRS + i) * 16) + sl;
        const int t0 = t_block + ((wt * WS_WTS + j) * 16);
#pragma unroll
        for (int l = 0; l < 8; ++l)
          y[std::size_t(t0 + 2 * l + hi) * m + row] = acc[i][j][l];
      }
  }
}

constexpr int kBM = 256, kBN = 256;

void LaunchProd(const void* w, const short* x, float* y, int batch, int m,
                int k) {
  const dim3 grid((batch + kBN - 1) / kBN, (m + kBM - 1) / kBM);
  gufo::hip::HalfPrefillGemmKernel<256, 256, 8, 4, GgmlType::kQ4_K,
                                   gufo::hip::HalfEpilogue::kStore, true,
                                   GgmlType::kQ4_K, 4, 0>
      <<<grid, 1024>>>(w, x, y, batch, m, k, nullptr);
}

void LaunchWsNoSync(const void* w, const short* x, float* y, int batch, int m,
                    int k) {
  const dim3 grid((batch + WS_BN - 1) / WS_BN, (m + WS_BM - 1) / WS_BM);
  WarpSpecKernel<0><<<grid, 1024>>>(w, x, y, batch, m, k);
}

void LaunchWs(const void* w, const short* x, float* y, int batch, int m,
              int k) {
  const dim3 grid((batch + WS_BN - 1) / WS_BN, (m + WS_BM - 1) / WS_BM);
  WarpSpecKernel<1><<<grid, 1024>>>(w, x, y, batch, m, k);
}
}  // namespace

int main(int argc, char** argv) {
  const int m = argc > 1 ? std::atoi(argv[1]) : 17408;
  const int k = argc > 2 ? std::atoi(argv[2]) : 5120;
  const int batch = argc > 3 ? std::atoi(argv[3]) : 2048;
  const int rounds = argc > 4 ? std::atoi(argv[4]) : 4;
  std::printf("m=%d k=%d batch=%d rounds=%d\n", m, k, batch, rounds);
  std::fflush(stdout);
  if (m % kBM || batch % kBN) {
    std::printf("need m %% %d == 0 and batch %% %d == 0\n", kBM, kBN);
    return 2;
  }
  const std::size_t rb = (std::size_t)(k / 256) * 144;
  void *w = nullptr, *x = nullptr, *y1 = nullptr, *y2 = nullptr;
  const std::size_t yN = (std::size_t)batch * m;
  if (hipMalloc(&w, rb * m) != hipSuccess) return 1;
  if (hipMalloc(&x, (std::size_t)batch * k * 2) != hipSuccess) return 1;
  if (hipMalloc(&y1, yN * 4) != hipSuccess) return 1;
  if (hipMalloc(&y2, yN * 4) != hipSuccess) return 1;
  {
    // Deterministic but non-trivial inputs so a layout bug cannot pass by
    // reading only zeros.
    // Valid-looking Q4_K blocks: d = 1.0, dmin = 0, small finite scales, and
    // patterned qs. Arbitrary bytes make the fp16 block scale NaN and both
    // kernels then return NaN, which compares equal but proves nothing.
    std::vector<unsigned char> hw(rb * m);
    for (std::size_t i = 0; i < hw.size(); ++i) {
      const std::size_t off = i % 144;
      if (off < 2) hw[i] = off == 0 ? 0x00 : 0x3C;  // half 0x3C00 = 1.0
      else if (off < 4) hw[i] = 0x00;               // dmin = 0
      else if (off < 16) hw[i] = 0x21;              // scales
      else hw[i] = (unsigned char)((i * 37 + 11) & 0xFF);
    }
    // Activations must be valid finite fp16: random raw 16-bit patterns are
    // ~1 in 32 NaN or Inf and make every output NaN regardless of the kernel.
    std::vector<short> hx((std::size_t)batch * k);
    for (std::size_t i = 0; i < hx.size(); ++i) {
      const float v = (float)((int)(i % 97) - 48) * 0.03125f;
      const _Float16 h = (_Float16)v;
      std::memcpy(&hx[i], &h, 2);
    }
    hipMemcpy(w, hw.data(), hw.size(), hipMemcpyHostToDevice);
    hipMemcpy(x, hx.data(), hx.size() * 2, hipMemcpyHostToDevice);
  }
  hipMemset(y1, 0, yN * 4);
  hipMemset(y2, 0, yN * 4);
  LaunchProd(w, (const short*)x, (float*)y1, batch, m, k);
  LaunchWs(w, (const short*)x, (float*)y2, batch, m, k);
  hipError_t e = hipDeviceSynchronize();
  if (e != hipSuccess) {
    std::printf("launch failed: %s\n", hipGetErrorString(e));
    return 1;
  }
  std::vector<float> h1(yN), h2(yN);
  hipMemcpy(h1.data(), y1, yN * 4, hipMemcpyDeviceToHost);
  hipMemcpy(h2.data(), y2, yN * 4, hipMemcpyDeviceToHost);
  double maxd = 0; std::size_t nmis = 0, nan1 = 0, nan2 = 0;
  for (std::size_t i = 0; i < yN; ++i) {
    if (std::isnan(h1[i])) ++nan1;
    if (std::isnan(h2[i])) ++nan2;
    const double d = std::fabs((double)h1[i] - (double)h2[i]);
    if (d > maxd) maxd = d;
    if (d != 0.0) ++nmis;
  }
  std::printf("nan: prod=%zu ws=%zu   sample prod=%.6g ws=%.6g\n", nan1, nan2,
              (double)h1[0], (double)h2[0]);
  std::printf("precision: max|diff|=%.6g mismatches=%zu of %zu\n", maxd, nmis, yN);
  std::fflush(stdout);
  const double fl = 2.0 * batch * (double)m * (double)k;
  std::vector<double> tp, tw, tn;
  for (int r = 0; r < rounds; ++r) {
    const bool fwd = (r % 2) == 0;
    for (int a = 0; a < 3; ++a) {
      const int which = fwd ? a : 2 - a;
      hipDeviceSynchronize();
      const auto s0 = std::chrono::steady_clock::now();
      for (int i = 0; i < 2; ++i) {
        if (which == 0) LaunchProd(w, (const short*)x, (float*)y1, batch, m, k);
        else if (which == 1)
          LaunchWs(w, (const short*)x, (float*)y2, batch, m, k);
        else LaunchWsNoSync(w, (const short*)x, (float*)y2, batch, m, k);
      }
      hipDeviceSynchronize();
      const auto s1 = std::chrono::steady_clock::now();
      const double ms =
          std::chrono::duration<double, std::milli>(s1 - s0).count() / 2.0;
      if (which == 0) tp.push_back(ms);
      else if (which == 1) tw.push_back(ms);
      else tn.push_back(ms);
    }
  }
  auto report = [&](const char* name, std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    const double med = v[v.size() / 2];
    std::printf("%-18s median=%8.3f min=%8.3f max=%8.3f tflops=%6.2f\n",
                name, med, v.front(), v.back(), fl / (med * 1e9));
  };
  report("production", tp);
  report("warp-specialised", tw);
  report("ws no sync (UB)", tn);
  std::printf("ratio ws/prod = %.3f\n", tw[tw.size() / 2] / tp[tp.size() / 2]);
  std::printf("done\n");
  return 0;
}