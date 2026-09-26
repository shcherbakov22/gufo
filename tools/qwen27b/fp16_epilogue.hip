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

// The FFN gate/up GEMM is 70% of prefill FLOPs and runs the Paired kGateUp
// epilogue: two accumulators per wave, an expf and a half round per output.
// Ablation bit 8 disables the store in every epilogue branch, so the only
// difference between the two arms is the epilogue itself.
constexpr int kM = 17408, kK = 5120;

// Paired gate/up: each 256-row tile covers 128 output rows because it loads
// gate and up together, hence the (m + 127) / 128 grid.
template<int Ablate>
void LaunchPaired(const void* gw, const void* uw, const short* x, float* y,
                  int batch, int m, int k) {
  const dim3 grid((batch + 255) / 256, (m + 127) / 128);
  gufo::hip::HalfPrefillGemmKernel<256, 256, 8, 4, GgmlType::kQ4_K,
                                   gufo::hip::HalfEpilogue::kGateUp, true,
                                   GgmlType::kQ4_K, 4, Ablate>
      <<<grid, 1024>>>(gw, x, y, batch, m, k, nullptr, uw);
}

// Plain store at the same row count, as an internal reference point.
template<int Ablate>
void LaunchStore(const void* w, const short* x, float* y, int batch, int m,
                 int k) {
  const dim3 grid((batch + 255) / 256, (m + 255) / 256);
  gufo::hip::HalfPrefillGemmKernel<256, 256, 8, 4, GgmlType::kQ4_K,
                                   gufo::hip::HalfEpilogue::kStore, true,
                                   GgmlType::kQ4_K, 4, Ablate>
      <<<grid, 1024>>>(w, x, y, batch, m, k, nullptr);
}

// One uniform signature so the arms can be selected at run time.
using Fn = void (*)(const void*, const void*, const short*, float*, int, int,
                    int);
template<int Ablate>
void PairArm(const void* gw, const void* uw, const short* x, float* y, int b,
             int m, int k) {
  LaunchPaired<Ablate>(gw, uw, x, y, b, m, k);
}
template<int Ablate>
void StoreArm(const void* w, const void*, const short* x, float* y, int b,
              int m, int k) {
  LaunchStore<Ablate>(w, x, y, b, m, k);
}

constexpr int kArms = 5;
const char* kNames[kArms] = {"gate/up full", "gate/up no-epilogue",
                             "store full", "store no-epilogue",
                             "gate/up full B"};
// FLOPs per call: paired does gate and up, so 4*m*k*batch; store is half.
const Fn kFn[kArms] = {PairArm<0>, PairArm<8>, StoreArm<0>, StoreArm<8>,
                       PairArm<0>};
constexpr bool kPaired[kArms] = {true, true, false, false, true};
}  // namespace

int main(int argc, char** argv) {
  const int batch = argc > 1 ? std::atoi(argv[1]) : 2048;
  const int rounds = argc > 2 ? std::atoi(argv[2]) : 4;
  const int m = kM, k = kK;
  std::printf("paired gate/up m=%d k=%d batch=%d rounds=%d\n", m, k, batch,
              rounds);
  std::fflush(stdout);
  const std::size_t row_bytes = (std::size_t)(k / 256) * 144;
  void *gw = nullptr, *uw = nullptr, *x = nullptr;
  float* y = nullptr;
  if (hipMalloc(&gw, row_bytes * m) != hipSuccess) return 1;
  if (hipMalloc(&uw, row_bytes * m) != hipSuccess) return 1;
  if (hipMalloc(&x, (std::size_t)batch * k * 2) != hipSuccess) return 1;
  // 4 bytes per output row element: kStore writes floats, kGateUp writes
  // halves into the same allocation. At 2 bytes the kStore arm overruns.
  if (hipMalloc(&y, (std::size_t)batch * m * 4) != hipSuccess) return 1;
  (void)hipMemset(gw, 0, row_bytes * m);
  (void)hipMemset(uw, 0, row_bytes * m);
  (void)hipMemset(x, 0, (std::size_t)batch * k * 2);
  (void)hipMemset(y, 0, (std::size_t)batch * m * 4);
  kFn[0](gw, uw, (const short*)x, y, batch, m, k);
  kFn[2](gw, uw, (const short*)x, y, batch, m, k);
  hipError_t e = hipDeviceSynchronize();
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
    const double fl = (kPaired[a] ? 4.0 : 2.0) * batch * (double)m * k;
    std::printf("%-18s median=%8.3f min=%8.3f max=%8.3f tflops=%6.2f\n",
                kNames[a], med, v.front(), v.back(), fl / (med * 1e9));
    std::fflush(stdout);
  }
  std::printf("done\n");
  return 0;
}