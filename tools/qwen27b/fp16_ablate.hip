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
constexpr int kArms = 4;
// Arms 0 and 3 are the SAME configuration: the gap between them is the
// residual bias left in the interleaved harness.
const char* kNames[kArms] = {"ctrl A", "decode in K loop",
                             "staggered decode", "ctrl B"};
constexpr int kAbl[kArms] = {0, 512, 1024, 0};

template<int Ablate>
void Launch(const void* w, const short* x, float* y, int batch, int m, int k) {
  const dim3 grid((batch + 255) / 256, (m + 255) / 256);
  gufo::hip::HalfPrefillGemmKernel<256, 256, 8, 4, GgmlType::kQ4_K,
                                   gufo::hip::HalfEpilogue::kStore, true,
                                   GgmlType::kQ4_K, 4, Ablate>
      <<<grid, 1024>>>(w, x, y, batch, m, k, nullptr);
}

template<int Ablate>
double Run(const void* w, const short* x, float* y, int batch, int m, int k,
           int iters) {
  Launch<Ablate>(w, x, y, batch, m, k);
  hipDeviceSynchronize();
  const auto s = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) Launch<Ablate>(w, x, y, batch, m, k);
  hipDeviceSynchronize();
  const auto e = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(e - s).count() / iters;
}
}  // namespace

int main(int argc, char** argv) {
  const int m = argc > 1 ? std::atoi(argv[1]) : 17408;
  const int k = argc > 2 ? std::atoi(argv[2]) : 5120;
  const int batch = argc > 3 ? std::atoi(argv[3]) : 2048;
  const int rounds = argc > 4 ? std::atoi(argv[4]) : 5;
  std::printf("m=%d k=%d batch=%d rounds=%d\n", m, k, batch, rounds);
  std::fflush(stdout);
  const std::size_t rb = (std::size_t)(k / 256) * 144;
  void* w = nullptr;
  void* x = nullptr;
  void* y = nullptr;
  const std::size_t yN = (std::size_t)batch * m;
  if (hipMalloc(&w, rb * m) != hipSuccess) return 1;
  if (hipMalloc(&x, (std::size_t)batch * k * 2) != hipSuccess) return 1;
  if (hipMalloc(&y, yN * 4) != hipSuccess) return 1;
  hipMemset(w, 0, rb * m);
  hipMemset(x, 0, (std::size_t)batch * k * 2);
  hipMemset(y, 0, yN * 4);
  const double fl = 2.0 * batch * (double)m * (double)k;
  std::vector<double> s[kArms];
  for (int r = 0; r < rounds; ++r) {
    const bool fwd = (r % 2) == 0;
    for (int idx = 0; idx < kArms; ++idx) {
      const int a = fwd ? idx : (kArms - 1 - idx);
      double ms = 0;
      for (int i = 0; i < 4; ++i) {
        switch (a) {
          case 0: ms += Run<0>(w, (const short*)x, (float*)y, batch, m, k, 1); break;
          case 1: ms += Run<512>(w, (const short*)x, (float*)y, batch, m, k, 1); break;
          case 2: ms += Run<1024>(w, (const short*)x, (float*)y, batch, m, k, 1); break;
          case 3: ms += Run<0>(w, (const short*)x, (float*)y, batch, m, k, 1); break;
        }
      }
      s[a].push_back(ms / 4.0);
    }
  }
  double a0 = 0;
  for (int a = 0; a < kArms; ++a) {
    std::vector<double> v = s[a];
    std::sort(v.begin(), v.end());
    const double med = v[v.size() / 2];
    if (a == 0) a0 = med;
    std::printf("%-12s (ablate=%2d) median=%8.3f min=%8.3f max=%8.3f tflops=%6.2f (%+.1f%%)\n",
                kNames[a], kAbl[a], med, v.front(), v.back(), fl / (med * 1e9),
                100.0 * (a0 - med) / a0);
    std::fflush(stdout);
  }
  std::printf("done\n");
  return 0;
}
