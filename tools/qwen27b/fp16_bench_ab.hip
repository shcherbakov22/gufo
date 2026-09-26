#define ENGINE_ENABLE_HIP 1
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
// Exactly the slow harness include set, in the same order.
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen/hip/kernels/prefill_fp16.hip"
#include "src/models/qwen/hip/ops/prefill_fp16.hpp"
#include "tests/models/qwen/hip/support/device_buffer.hpp"

using gufo::core::GgmlType;
using gufo::test::DeviceBuffer;

namespace {
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
  const int rounds = argc > 2 ? std::atoi(argv[2]) : 5;
  const int m = kM, k = kK;
  const std::size_t rb = (std::size_t)(k / 256) * 144;
  DeviceBuffer<unsigned char> w(rb * m);
  DeviceBuffer<short> x((std::size_t)batch * k);
  DeviceBuffer<float> y((std::size_t)batch * m);
  (void)hipMemset(w.data(), 0, rb * m);
  (void)hipMemset(x.data(), 0, (std::size_t)batch * k * 2);
  (void)hipMemset(y.data(), 0, (std::size_t)batch * m * 4);
  for (int a = 0; a < kArms; ++a)
    kFn[a](w.data(), x.data(), y.data(), batch, m, k);
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
        kFn[a](w.data(), x.data(), y.data(), batch, m, k);
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
    std::printf("%-12s median=%8.3f ms tflops=%6.2f\n", kNames[a], med,
                fl / (med * 1e9));
    std::fflush(stdout);
  }
  return 0;
}