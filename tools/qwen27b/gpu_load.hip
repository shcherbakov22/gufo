// Two GPU loads, to separate what actually interferes with the NPU.
//
//   gpu_load <mem|compute> <seconds>
//
// "mem" streams a large buffer and nothing else, so it saturates DRAM
// bandwidth. "compute" keeps its working set in registers and L2 and burns ALU,
// so it raises the GPU's clocks and its share of the SoC power budget without
// touching DRAM. Running the NPU loop against each says whether its ~25%
// concurrency tax is bandwidth or power/scheduling.
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <hip/hip_runtime.h>

namespace {

constexpr std::size_t kMemBytes = 256ull << 20;  // 256 MiB
constexpr int kMemPasses = 24;

__global__ void StreamKernel(const float* __restrict__ in,
                             float* __restrict__ out, unsigned long long n,
                             int passes) {
  const unsigned long long stride =
      static_cast<unsigned long long>(gridDim.x) * blockDim.x;
  unsigned long long i =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  for (int p = 0; p < passes; ++p) {
    for (unsigned long long j = i; j < n; j += stride) {
      out[j] = __fmaf_rn(in[j], 1.000001f, 0.5f);
    }
    __syncthreads();
  }
}

// Register-resident: the same 8 KiB is read once and then only ALU runs.
__global__ void ComputeKernel(float* __restrict__ out, int iters) {
  float a = threadIdx.x * 1.0e-6f + 0.5f;
  float b = 1.0000001f;
  float c = 0.0000001f;
#pragma unroll 16
  for (int i = 0; i < iters; ++i) {
    a = __fmaf_rn(a, b, c);
    b = __fmaf_rn(b, 1.0f, 1.0e-9f);
  }
  if (a == 12345.678f) out[threadIdx.x] = b;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <mem|compute> <seconds>\n", argv[0]);
    return 2;
  }
  const bool mem = std::strcmp(argv[1], "mem") == 0;
  const int seconds = std::atoi(argv[2]);
  // Duty cycle in percent. Pacing the device between syncs sets the average
  // package power without touching any power or clock setting, which is what
  // separates "the NPU is squeezed by the shared budget" from "the NPU is
  // clamped whenever the GPU is active at all".
  const int duty = argc > 3 ? std::atoi(argv[3]) : 100;
  const std::size_t n = kMemBytes / sizeof(float);
  float* in = nullptr;
  float* out = nullptr;
  if (hipMalloc(&in, kMemBytes) != hipSuccess) return 1;
  if (hipMalloc(&out, kMemBytes) != hipSuccess) return 1;
  if (hipMemset(in, 0x3f, kMemBytes) != hipSuccess) return 1;
  const int blocks = 2048;
  const int threads = 256;
  std::printf("gpu_load %s for %d s at %d%% duty\n", mem ? "mem" : "compute",
              seconds, duty);
  std::fflush(stdout);
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
  // Bound the queue to two launches. Without this the host enqueues thousands
  // of them during the deadline and then sits in hipDeviceSynchronize draining
  // the backlog long after the requested duration: a 14 s "load" that pegged
  // the GPU for more than fifteen minutes. Two in flight keeps the device
  // saturated and the process finite.
  int in_flight = 0;
  while (std::chrono::steady_clock::now() < deadline) {
    const auto t0 = std::chrono::steady_clock::now();
    if (mem) {
      StreamKernel<<<blocks, threads>>>(in, out, n, kMemPasses);
    } else {
      ComputeKernel<<<blocks, threads>>>(out, 200000);
    }
    if (++in_flight == 2) {
      if (hipDeviceSynchronize() != hipSuccess) return 1;
      in_flight = 0;
    }
    if (in_flight == 0 && duty < 100) {
      const auto busy = std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::steady_clock::now() - t0)
                            .count();
      const long long off = busy * (100 - duty) / duty;
      if (off > 0) usleep(static_cast<unsigned>(off));
    }
  }
  if (hipDeviceSynchronize() != hipSuccess) return 1;
  std::printf("gpu_load done\n");
  return 0;
}
