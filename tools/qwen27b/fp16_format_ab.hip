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

// The shipped 3.84bpw shard runs every FFN tensor as IQ3_XXS, not Q4_K. This
// measures the two formats at identical geometry so the decode cost can be
// read off directly, with the epilogue held fixed.
constexpr int kM = 17408, kK = 5120;

template<GgmlType Type, int Ablate>
void LaunchStore(const void* w, const short* x, float* y, int batch, int m,
                 int k) {
  const dim3 grid((batch + 255) / 256, (m + 255) / 256);
  gufo::hip::HalfPrefillGemmKernel<256, 256, 8, 4, Type,
                                   gufo::hip::HalfEpilogue::kStore, false,
                                   Type, 4, Ablate>
      <<<grid, 1024>>>(w, x, y, batch, m, k, nullptr);
}

// Paired gate/up needs gate and up together, hence half the row tile.
template<int Ablate>
void LaunchPaired(const void* gw, const void* uw, const short* x, float* y,
                  int batch, int m, int k) {
  const dim3 grid((batch + 255) / 256, (m + 127) / 128);
  gufo::hip::HalfPrefillGemmKernel<256, 256, 8, 4, GgmlType::kQ4_K,
                                   gufo::hip::HalfEpilogue::kGateUp, true,
                                   GgmlType::kQ4_K, 4, Ablate>
      <<<grid, 1024>>>(gw, x, y, batch, m, k, nullptr, uw);
}

using Fn = void (*)(const void*, const void*, const short*, float*, int, int,
                    int);
template<GgmlType Type, int Ablate>
void StoreArm(const void* w, const void*, const short* x, float* y, int b,
              int m, int k) {
  LaunchStore<Type, Ablate>(w, x, y, b, m, k);
}
template<int Ablate>
void PairArm(const void* gw, const void* uw, const short* x, float* y, int b,
             int m, int k) {
  LaunchPaired<Ablate>(gw, uw, x, y, b, m, k);
}

constexpr int kArms = 6;
const char* kNames[kArms] = {
    "q4k store",        "iq3xxs store",       "q4k store B",
    "iq3xxs no-epilog", "iq3xxs store B",     "q4k paired gate/up"};
const Fn kFn[kArms] = {
    StoreArm<GgmlType::kQ4_K, 0>,    StoreArm<GgmlType::kIQ3_XXS, 0>,
    StoreArm<GgmlType::kQ4_K, 0>,    StoreArm<GgmlType::kIQ3_XXS, 8>,
    StoreArm<GgmlType::kIQ3_XXS, 0>, PairArm<0>};
// Paired produces gate and up, so twice the MACs of a single store GEMM.
constexpr double kFlopMul[kArms] = {2, 2, 2, 2, 2, 4};
}  // namespace

int main(int argc, char** argv) {
  const int batch = argc > 1 ? std::atoi(argv[1]) : 2048;
  const int rounds = argc > 2 ? std::atoi(argv[2]) : 6;
  const int m = kM, k = kK;
  std::printf("m=%d k=%d batch=%d rounds=%d\n", m, k, batch, rounds);
  std::fflush(stdout);
  // Q4_K is the larger per-row stride, so one allocation serves both formats.
  const std::size_t rb_q4k = (std::size_t)(k / 256) * 144;
  const int rows = m;
  void *gw = nullptr, *uw = nullptr, *x = nullptr;
  float* y = nullptr;
  if (hipMalloc(&gw, rb_q4k * rows) != hipSuccess) return 1;
  if (hipMalloc(&uw, rb_q4k * rows) != hipSuccess) return 1;
  if (hipMalloc(&x, (std::size_t)batch * k * 2) != hipSuccess) return 1;
  // 4 bytes per output element: kStore writes floats, kGateUp writes halves
  // into the same allocation. At 2 bytes the kStore arms overrun.
  if (hipMalloc(&y, (std::size_t)batch * m * 4) != hipSuccess) return 1;
  (void)hipMemset(gw, 0, rb_q4k * rows);
  (void)hipMemset(uw, 0, rb_q4k * rows);
  (void)hipMemset(x, 0, (std::size_t)batch * k * 2);
  (void)hipMemset(y, 0, (std::size_t)batch * m * 4);
  for (int a = 0; a < kArms; ++a)
    kFn[a](gw, uw, (const short*)x, y, batch, m, k);
  const hipError_t e = hipDeviceSynchronize();
  if (e != hipSuccess) {
    std::printf("launch failed: %s\n", hipGetErrorString(e));
    return 1;
  }
  std::vector<double> sv[kArms];
  for (int r = 0; r < rounds; ++r) {
    const bool fwd = (r % 2) == 0;
    for (int idx = 0; idx < kArms; ++idx) {
      const int a = fwd ? idx : (kArms - 1 - idx);
      double ms = 0;
      for (int i = 0; i < 3; ++i) {
        (void)hipDeviceSynchronize();
        const auto s0 = std::chrono::steady_clock::now();
        kFn[a](gw, uw, (const short*)x, y, batch, m, k);
        (void)hipDeviceSynchronize();
        const auto s1 = std::chrono::steady_clock::now();
        ms += std::chrono::duration<double, std::milli>(s1 - s0).count();
      }
      sv[a].push_back(ms / 3.0);
    }
  }
  for (int a = 0; a < kArms; ++a) {
    std::vector<double> v = sv[a];
    std::sort(v.begin(), v.end());
    const double med = v[v.size() / 2];
    const double fl = kFlopMul[a] * batch * (double)m * k;
    std::printf("%-18s median=%8.3f min=%8.3f max=%8.3f tflops=%6.2f\n",
                kNames[a], med, v.front(), v.back(), fl / (med * 1e9));
    std::fflush(stdout);
  }
  std::printf("done\n");
  return 0;
}