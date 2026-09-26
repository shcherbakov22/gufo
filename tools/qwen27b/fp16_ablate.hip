#define ENGINE_ENABLE_HIP 1
#include <chrono>
#include <cstdio>
#include <hip/hip_runtime.h>
#include "src/models/qwen/hip/kernels/prefill_fp16.hip"
using gufo::core::GgmlType;
namespace {
constexpr int BM = 256, BN = 256, WM = 8, WN = 4, BK = 4, kT = WM * WN * 32;
template<int Ablate>
double Run(const void* w, const short* x, float* y, int batch, int m, int k,
           int iters) {
  const dim3 grid((batch + BN - 1) / BN, (m + BM - 1) / BM);
  const auto launch = [&] {
    gufo::hip::HalfPrefillGemmKernel<BM, BN, WM, WN, GgmlType::kQ4_K,
                                     gufo::hip::HalfEpilogue::kStore, true,
                                     GgmlType::kQ4_K, BK, Ablate>
        <<<grid, kT>>>(w, x, y, batch, m, k, nullptr);
  };
  launch(); hipDeviceSynchronize();
  const auto s = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) launch();
  hipDeviceSynchronize();
  const auto e = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(e - s).count() / iters;
}
}  // namespace
int main() {
  const int m = 17408, k = 5120, batch = 2048;
  const std::size_t rb = (std::size_t)(k / 256) * 144;
  void *w = nullptr, *x = nullptr, *y = nullptr;
  hipMalloc(&w, rb * m);
  hipMalloc(&x, (std::size_t)batch * k * 2);
  hipMalloc(&y, (std::size_t)batch * m * 4);
  hipMemset(w, 0, rb * m);
  hipMemset(x, 0, (std::size_t)batch * k * 2);
  hipMemset(y, 0, (std::size_t)batch * m * 4);
  const double fl = 2.0 * batch * (double)m * (double)k;
  struct { const char* n; int a; } cs[] = {
      {"baseline", 0},
      {"no commit (no staging at all)", 2},
      {"no fetch (commit kept)", 4},
      {"no barriers", 1},
      {"no fetch, no barriers", 5},
  };
  for (auto c : cs) {
    double ms = 0;
    switch (c.a) {
      case 0: ms = Run<0>(w,(const short*)x,(float*)y,batch,m,k,8); break;
      case 1: ms = Run<1>(w,(const short*)x,(float*)y,batch,m,k,8); break;
      case 2: ms = Run<2>(w,(const short*)x,(float*)y,batch,m,k,8); break;
      case 4: ms = Run<4>(w,(const short*)x,(float*)y,batch,m,k,8); break;
      case 5: ms = Run<5>(w,(const short*)x,(float*)y,batch,m,k,8); break;
    }
    std::printf("%-32s ms=%8.3f tflops=%6.2f\n", c.n, ms, fl / (ms * 1e9));
  }
  return 0;
}
