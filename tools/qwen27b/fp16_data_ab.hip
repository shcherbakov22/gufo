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

// Does the kernel time depend on the *data*?  Same instantiation both ways,
// zeroed buffers vs non-zero patterns.
constexpr int kM = 17408, kK = 5120;

__global__ void FillBytes(unsigned char* p, unsigned long long n,
                          unsigned seed) {
  const unsigned long long i =
      (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = (unsigned char)((i * 37ull + seed * 131ull + 17ull) & 0xFFull);
}

__global__ void FillShorts(short* p, unsigned long long n, short v) {
  const unsigned long long i =
      (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = v;
}

void LaunchDirect(const void* w, const short* x, float* y, int batch, int m,
                  int k) {
  const dim3 grid((batch + 255) / 256, (m + 255) / 256);
  gufo::hip::HalfPrefillGemmKernel<256, 256, 8, 4, GgmlType::kQ4_K,
                                   gufo::hip::HalfEpilogue::kStore, true,
                                   GgmlType::kQ4_K, 4, 0>
      <<<grid, 1024>>>(w, x, y, batch, m, k, nullptr);
}

using Fn = void (*)(const void*, const short*, float*, int, int, int);
}  // namespace

int main(int argc, char** argv) {
  const int batch = argc > 1 ? std::atoi(argv[1]) : 2048;
  const int rounds = argc > 2 ? std::atoi(argv[2]) : 4;
  const int fill = argc > 3 ? std::atoi(argv[3]) : 0;
  const int m = kM, k = kK;
  const std::size_t rb = (std::size_t)(k / 256) * 144;
  const std::size_t wbytes = rb * m;
  const std::size_t nshort = (std::size_t)batch * k;
  void *w = nullptr, *x = nullptr;
  float* y = nullptr;
  if (hipMalloc(&w, wbytes) != hipSuccess) return 1;
  if (hipMalloc(&x, nshort * 2) != hipSuccess) return 1;
  if (hipMalloc(&y, (std::size_t)batch * m * 4) != hipSuccess) return 1;
  if (fill) {
    FillBytes<<<(unsigned)((wbytes + 255) / 256), 256>>>(
        (unsigned char*)w, wbytes, 7u);
    FillShorts<<<(unsigned)((nshort + 255) / 256), 256>>>((short*)x, nshort,
                                                           (short)0x3800);
  } else {
    (void)hipMemset(w, 0, wbytes);
    (void)hipMemset(x, 0, nshort * 2);
  }
  (void)hipMemset(y, 0, (std::size_t)batch * m * 4);
  LaunchDirect(w, (const short*)x, y, batch, m, k);
  if (hipDeviceSynchronize() != hipSuccess) return 1;
  std::vector<double> sv;
  for (int r = 0; r < rounds; ++r) {
    double ms = 0;
    for (int i = 0; i < 3; ++i) {
      (void)hipDeviceSynchronize();
      const auto s0 = std::chrono::steady_clock::now();
      LaunchDirect(w, (const short*)x, y, batch, m, k);
      (void)hipDeviceSynchronize();
      const auto s1 = std::chrono::steady_clock::now();
      ms += std::chrono::duration<double, std::milli>(s1 - s0).count();
    }
    sv.push_back(ms / 3.0);
  }
  std::sort(sv.begin(), sv.end());
  const double med = sv[sv.size() / 2];
  const double fl = 2.0 * batch * (double)m * k;
  const auto host = [&] { std::vector<float> h((std::size_t)batch * m);
    (void)hipMemcpy(h.data(), y, h.size() * sizeof(float),
                    hipMemcpyDeviceToHost); return h; }();
  std::size_t nonfinite = 0; double amax = 0;
  for (float v : host) { nonfinite += !std::isfinite(v); amax = std::max(amax, (double)std::fabs(v)); }
  std::printf("fill=%d median=%8.3f ms tflops=%6.2f nonfinite=%zu maxabs=%.3g\n",
              fill, med, fl / (med * 1e9), nonfinite, amax);
  return 0;
}