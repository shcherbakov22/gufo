// What does the NPU<->iGPU handoff actually cost?
//
//   atb_share <xclbin> <insts.bin> <MB>
//
// The ATB path needs the GPU's A and B in the NPU's buffers and the NPU's C
// back in GPU memory. dma-buf import fails and amdxdna rejects userptr BOs, so
// this measures the fallback: register variants for an XRT BO mapping, then
// hipMemcpy bandwidth between a device buffer and both that mapping and a
// hipHostMalloc buffer of the same size. The two should be the same speed --
// both are plain RAM -- which is what makes the copy handoff affordable.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

namespace {

int Fail(const char* what, hipError_t e) {
  std::fprintf(stderr, "HIP %s -> %s (%d)\n", what, hipGetErrorString(e),
               static_cast<int>(e));
  return 3;
}

double NowMs() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch())
      .count();
}

// Best of k repetitions, in GB/s where a copy of n bytes moves n bytes.
double CopyRate(void* dst, const void* src, std::size_t n, hipMemcpyKind kind,
                int reps) {
  hipError_t e = hipMemcpy(dst, src, n, kind);
  if (e != hipSuccess) return -1.0;
  double best = 1e30;
  for (int i = 0; i < reps; ++i) {
    const double t0 = NowMs();
    e = hipMemcpy(dst, src, n, kind);
    const double t1 = NowMs();
    if (e != hipSuccess) return -1.0;
    if (t1 - t0 < best) best = t1 - t0;
  }
  return static_cast<double>(n) / (best / 1000.0) / 1e9;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s <xclbin> <insts.bin> <MB>\n", argv[0]);
    return 2;
  }
  const std::size_t n = std::strtoull(argv[3], nullptr, 10) * 1024 * 1024;
  std::printf("buffer %zu bytes (%.1f MB)\n", n, static_cast<double>(n) / 1048576.0);

  xrt::device device(0);
  auto xclbin = xrt::xclbin(std::string(argv[1]));
  const auto kernels = xclbin.get_kernels();
  if (kernels.empty()) { std::fprintf(stderr, "no kernels\n"); return 1; }
  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, kernels.front().get_name());

  auto bo = xrt::bo(device, n, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  void* host_ptr = bo.map<void*>();
  std::printf("xrt bo host ptr %p\n", host_ptr);

  const struct { const char* name; unsigned int f; } regs[] = {
      {"default", hipHostRegisterDefault},
      {"portable", hipHostRegisterPortable},
      {"mapped", hipHostRegisterMapped},
      {"portable|mapped", hipHostRegisterPortable | hipHostRegisterMapped},
  };
  for (const auto& r : regs) {
    const hipError_t e = hipHostRegister(host_ptr, n, r.f);
    std::printf("hipHostRegister %-16s -> %s (%d)\n", r.name,
                hipGetErrorString(e), static_cast<int>(e));
    if (e == hipSuccess) {
      void* dp = nullptr;
      const hipError_t e2 = hipHostGetDevicePointer(&dp, host_ptr, 0);
      std::printf("  hipHostGetDevicePointer -> %s (%d) ptr=%p\n",
                  hipGetErrorString(e2), static_cast<int>(e2), dp);
      (void)hipHostUnregister(host_ptr);
    }
  }

  unsigned char* dev = nullptr;
  if (hipMalloc(&dev, n) != hipSuccess) return Fail("hipMalloc", hipGetLastError());
  unsigned char* pinned = nullptr;
  if (hipHostMalloc(&pinned, n) != hipSuccess) return Fail("hipHostMalloc", hipGetLastError());
  unsigned char* pageable = static_cast<unsigned char*>(std::malloc(n));
  std::memset(host_ptr, 1, n);
  std::memset(pinned, 2, n);
  std::memset(pageable, 3, n);
  std::memset(dev, 0, n);

  const int reps = 20;
  std::printf("\nkind: best-of-%d GB/s (n bytes moved per direction)\n", reps);
  std::printf("H2D  dev <- xrt bo       %8.2f\n", CopyRate(dev, host_ptr, n, hipMemcpyHostToDevice, reps));
  std::printf("D2H  xrt bo <- dev       %8.2f\n", CopyRate(host_ptr, dev, n, hipMemcpyDeviceToHost, reps));
  std::printf("H2D  dev <- hipHostMalloc %6.2f\n", CopyRate(dev, pinned, n, hipMemcpyHostToDevice, reps));
  std::printf("D2H  hipHostMalloc <- dev %6.2f\n", CopyRate(pinned, dev, n, hipMemcpyDeviceToHost, reps));
  std::printf("H2D  dev <- pageable     %8.2f\n", CopyRate(dev, pageable, n, hipMemcpyHostToDevice, reps));
  std::printf("D2D  dev <- dev          %8.2f\n", CopyRate(dev, dev, n, hipMemcpyDeviceToDevice, reps));
  std::printf("memcpy cpu bo<-pageable  %8.2f\n", [&] {
    double best = 1e30;
    for (int i = 0; i < reps; ++i) {
      const double t0 = NowMs();
      std::memcpy(host_ptr, pageable, n);
      const double t1 = NowMs();
      if (t1 - t0 < best) best = t1 - t0;
    }
    return static_cast<double>(n) / (best / 1000.0) / 1e9;
  }());

  (void)hipFree(dev);
  (void)hipHostFree(pinned);
  std::free(pageable);
  return 0;
}
