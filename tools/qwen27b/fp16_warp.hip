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
// Same tile, same BK, same LDS budget; only the warp split changes. At 8x4 each
// warp owns 8 MMAs and the fragment traffic is 512*(1/2 + 1/4) = 384 B/MMA. At
// 4x4 each warp owns 16 MMAs and it is 512*(1/4 + 1/4) = 256 B/MMA, at the cost
// of 512 threads per CTA instead of 1024.
template<int WM, int WN>
void Launch(const void* w, const short* x, float* y, int batch, int m, int k) {
  const dim3 grid((batch + 255) / 256, (m + 255) / 256);
  gufo::hip::HalfPrefillGemmKernel<256, 256, WM, WN, GgmlType::kQ4_K,
                                   gufo::hip::HalfEpilogue::kStore, true,
                                   GgmlType::kQ4_K, 4, 0>
      <<<grid, WM * WN * 32>>>(w, x, y, batch, m, k, nullptr);
}

constexpr int kArms = 3;
const char* kNames[kArms] = {"8x4 ctrl A", "4x4 warp split", "8x4 ctrl B"};
}  // namespace

int main(int argc, char** argv) {
  const int m = argc > 1 ? std::atoi(argv[1]) : 17408;
  const int k = argc > 2 ? std::atoi(argv[2]) : 5120;
  const int batch = argc > 3 ? std::atoi(argv[3]) : 2048;
  const int rounds = argc > 4 ? std::atoi(argv[4]) : 4;
  std::printf("m=%d k=%d batch=%d rounds=%d\n", m, k, batch, rounds);
  std::fflush(stdout);
  const std::size_t rb = (std::size_t)(k / 256) * 144;
  void *w = nullptr, *x = nullptr, *y1 = nullptr, *y2 = nullptr;
  const std::size_t yN = (std::size_t)batch * m;
  if (hipMalloc(&w, rb * m) != hipSuccess) return 1;
  if (hipMalloc(&x, (std::size_t)batch * k * 2) != hipSuccess) return 1;
  if (hipMalloc(&y1, yN * 4) != hipSuccess) return 1;
  if (hipMalloc(&y2, yN * 4) != hipSuccess) return 1;
  {
    std::vector<unsigned char> hw(rb * m);
    for (std::size_t i = 0; i < hw.size(); ++i) {
      const std::size_t off = i % 144;
      if (off < 2) hw[i] = off == 0 ? 0x00 : 0x3C;
      else if (off < 4) hw[i] = 0x00;
      else if (off < 16) hw[i] = 0x21;
      else hw[i] = (unsigned char)((i * 37 + 11) & 0xFF);
    }
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
  Launch<8, 4>(w, (const short*)x, (float*)y1, batch, m, k);
  Launch<4, 4>(w, (const short*)x, (float*)y2, batch, m, k);
  hipError_t e = hipDeviceSynchronize();
  if (e != hipSuccess) {
    std::printf("launch failed: %s\n", hipGetErrorString(e));
    return 1;
  }
  std::vector<float> h1(yN), h2(yN);
  hipMemcpy(h1.data(), y1, yN * 4, hipMemcpyDeviceToHost);
  hipMemcpy(h2.data(), y2, yN * 4, hipMemcpyDeviceToHost);
  double maxd = 0; std::size_t nmis = 0;
  for (std::size_t i = 0; i < yN; ++i) {
    const double d = std::fabs((double)h1[i] - (double)h2[i]);
    if (d > maxd) maxd = d;
    if (d != 0.0) ++nmis;
  }
  std::printf("4x4 vs 8x4: max|diff|=%.6g mismatches=%zu of %zu\n", maxd, nmis, yN);
  std::fflush(stdout);
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
        if (a == 1) Launch<4, 4>(w, (const short*)x, (float*)y2, batch, m, k);
        else Launch<8, 4>(w, (const short*)x, (float*)y1, batch, m, k);
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
    std::printf("%-16s median=%8.3f min=%8.3f max=%8.3f tflops=%6.2f (%+.1f%%)\n",
                kNames[a], med, v.front(), v.back(), fl / (med * 1e9),
                100.0 * (a0 - med) / a0);
    std::fflush(stdout);
  }
  std::printf("done\n");
  return 0;
}