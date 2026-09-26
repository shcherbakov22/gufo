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
constexpr int kM = 17408, kK = 5120;

// Fill each 144-byte Q4_K block with a *valid* d field plus a byte pattern.
// d = 0 reproduces the memset case; a finite d makes the decoded weights
// non-zero, which is the main difference from the fast harness.
__global__ void FillBlocks(unsigned char* p, unsigned long long nblocks,
                           unsigned short dval, unsigned char pat) {
  const unsigned long long b =
      (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= nblocks) return;
  unsigned char* q = p + b * 144;
  q[0] = (unsigned char)(dval & 0xFF);
  q[1] = (unsigned char)(dval >> 8);
  q[2] = 0; q[3] = 0;
  for (int i = 4; i < 144; ++i) q[i] = pat;
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
}  // namespace

int main(int argc, char** argv) {
  const int batch = argc > 1 ? std::atoi(argv[1]) : 2048;
  const int rounds = argc > 2 ? std::atoi(argv[2]) : 5;
  const int m = kM, k = kK;
  const std::size_t rb = (std::size_t)(k / 256) * 144;
  const std::size_t wbytes = rb * m;
  const std::size_t nblocks = wbytes / 144;
  const std::size_t nshort = (std::size_t)batch * k;
  void *w = nullptr, *x = nullptr, *extra = nullptr;
  float* y = nullptr;
  if (hipMalloc(&w, wbytes) != hipSuccess) return 1;
  if (hipMalloc(&x, nshort * 2) != hipSuccess) return 1;
  if (hipMalloc(&y, (std::size_t)batch * m * 4) != hipSuccess) return 1;
  if (hipMalloc(&extra, 42u * 1024 * 1024) != hipSuccess) return 1;
  (void)hipMemset(extra, 0, 42u * 1024 * 1024);
  std::vector<float> host((std::size_t)batch * m);

  struct Mode { const char* name; unsigned short d; unsigned char pat;
                short xv; bool copyback; };
  const Mode modes[] = {
      {"d=0 x=0",            0,      0x00, 0x0000, false},
      {"d=1 x=0",            0x3C00, 0x11, 0x0000, false},
      {"d=1e-4 x=0",         0x0400, 0x11, 0x0000, false},
      {"d=1e-4 x=0.5",       0x0400, 0x11, 0x3800, false},
      {"d=1e-4 x=0.5 +copy", 0x0400, 0x11, 0x3800, true},
  };
  const int nm = (int)(sizeof(modes) / sizeof(modes[0]));
  for (int i = 0; i < nm; ++i) {
    const Mode& mo = modes[i];
    FillBlocks<<<(unsigned)((nblocks + 255) / 256), 256>>>(
        (unsigned char*)w, nblocks, mo.d, mo.pat);
    FillShorts<<<(unsigned)((nshort + 255) / 256), 256>>>((short*)x, nshort,
                                                            mo.xv);
    (void)hipMemset(y, 0, (std::size_t)batch * m * 4);
    LaunchDirect(w, (const short*)x, y, batch, m, k);
    (void)hipDeviceSynchronize();
    if (mo.copyback)
      (void)hipMemcpy(host.data(), y, host.size() * sizeof(float),
                      hipMemcpyDeviceToHost);
    std::vector<double> sv;
    for (int r = 0; r < rounds; ++r) {
      (void)hipDeviceSynchronize();
      const auto s0 = std::chrono::steady_clock::now();
      LaunchDirect(w, (const short*)x, y, batch, m, k);
      (void)hipDeviceSynchronize();
      const auto s1 = std::chrono::steady_clock::now();
      sv.push_back(
          std::chrono::duration<double, std::milli>(s1 - s0).count());
    }
    std::sort(sv.begin(), sv.end());
    const double med = sv[sv.size() / 2];
    const double fl = 2.0 * batch * (double)m * k;
    std::printf("%-22s median=%8.3f ms tflops=%6.2f\n", mo.name, med,
                fl / (med * 1e9));
    std::fflush(stdout);
  }
  return 0;
}