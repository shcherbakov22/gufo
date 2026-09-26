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

// Same instantiation both ways: HalfPrefillGemmKernel<256,256,8,4,kQ4_K,
// kStore,true,kQ4_K,4,0>.  DirectGemm reaches it through the production
// launcher; the other path launches it by hand.  If these differ, the
// launcher itself is costing something.
constexpr int kM = 17408, kK = 5120;

void LaunchDirect(const void* w, const short* x, float* y, int batch, int m,
                  int k) {
  const dim3 grid((batch + 255) / 256, (m + 255) / 256);
  gufo::hip::HalfPrefillGemmKernel<256, 256, 8, 4, GgmlType::kQ4_K,
                                   gufo::hip::HalfEpilogue::kStore, true,
                                   GgmlType::kQ4_K, 4, 0>
      <<<grid, 1024>>>(w, x, y, batch, m, k, nullptr);
}

void LaunchProd(const void* w, const short* x, float* y, int batch, int m,
                int k) {
  gufo::hip::LaunchBatchedQuantGEMMFp16(GgmlType::kQ4_K, w, x, y,
                                        (std::size_t)batch, (std::size_t)m,
                                        (std::size_t)k, nullptr);
}

using Fn = void (*)(const void*, const short*, float*, int, int, int);
constexpr int kArms = 4;
const char* kNames[kArms] = {"direct", "production", "direct B", "prod B"};
const Fn kFn[kArms] = {LaunchDirect, LaunchProd, LaunchDirect, LaunchProd};
}  // namespace

int main(int argc, char** argv) {
  const int batch = argc > 1 ? std::atoi(argv[1]) : 2048;
  const int rounds = argc > 2 ? std::atoi(argv[2]) : 6;
  const int m = kM, k = kK;
  std::printf("m=%d k=%d batch=%d rounds=%d\n", m, k, batch, rounds);
  std::fflush(stdout);
  const std::size_t rb = (std::size_t)(k / 256) * 144;
  void *w = nullptr, *x = nullptr;
  float* y = nullptr;
  if (hipMalloc(&w, rb * m) != hipSuccess) return 1;
  if (hipMalloc(&x, (std::size_t)batch * k * 2) != hipSuccess) return 1;
  if (hipMalloc(&y, (std::size_t)batch * m * 4) != hipSuccess) return 1;
  (void)hipMemset(w, 0, rb * m);
  (void)hipMemset(x, 0, (std::size_t)batch * k * 2);
  (void)hipMemset(y, 0, (std::size_t)batch * m * 4);
  for (int a = 0; a < kArms; ++a)
    kFn[a](w, (const short*)x, y, batch, m, k);
  if (hipDeviceSynchronize() != hipSuccess) return 1;
  std::vector<double> sv[kArms];
  for (int r = 0; r < rounds; ++r) {
    const bool fwd = (r % 2) == 0;
    for (int idx = 0; idx < kArms; ++idx) {
      const int a = fwd ? idx : (kArms - 1 - idx);
      double ms = 0;
      for (int i = 0; i < 3; ++i) {
        (void)hipDeviceSynchronize();
        const auto s0 = std::chrono::steady_clock::now();
        kFn[a](w, (const short*)x, y, batch, m, k);
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
    const double fl = 2.0 * batch * (double)m * k;
    std::printf("%-12s median=%8.3f min=%8.3f max=%8.3f tflops=%6.2f\n",
                kNames[a], med, v.front(), v.back(), fl / (med * 1e9));
    std::fflush(stdout);
  }
  return 0;
}