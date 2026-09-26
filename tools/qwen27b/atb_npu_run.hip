// NPU GEMM runner over shared GPU memory, with a varying-data stress mode.
//
//   atb_npu_run <xclbin> <insts.bin> <M> <K> <N> [reps] [dev|host] [stress]
//
// A, B and C are hipMalloc allocations exported as amdgpu dma-bufs and imported
// by XRT as BOs, so the ATB kernel reads and writes the iGPU's own memory with
// no staging copy in either direction. This is the engine's data path: A and B
// are filled by a GPU kernel, never the host.
//
// Every stress iteration uses new A and B bytes and recomputes the reference
// against them. Repeating one fixed input set would hide a stale-cache bug
// completely, because the stale value equals the expected value; varying the
// data is the only way the comparison means anything.
//
// Correctness is checked twice: through a GPU kernel into a second allocation
// (the engine decodes C on the GPU), and by mmap of the dma-buf, which is a
// plain CPU load that sees what the NPU sees.
#include <sys/mman.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

namespace {

// The ERT start-CU opcode for these AIE xclbins is the literal 3, not
// group_id(0): passing the group id leaves C untouched.
constexpr unsigned int kErtOpcode = 3;

int Fail(const char* what, hipError_t e) {
  std::fprintf(stderr, "HIP %s -> %s (%d)\n", what, hipGetErrorString(e),
               static_cast<int>(e));
  return 3;
}

int FailLast(const char* what) { return Fail(what, hipGetLastError()); }

std::vector<unsigned char> FillHost(std::size_t n, unsigned int seed) {
  std::vector<unsigned char> v(n);
  for (std::size_t i = 0; i < n; ++i) {
    unsigned int s = seed ^ static_cast<unsigned int>(i * 2654435761u);
    s = s * 1664525u + 1013904223u;
    s = s * 1664525u + 1013904223u;
    v[i] = static_cast<unsigned char>(s >> 24);
  }
  return v;
}

__global__ void FillPatternKernel(unsigned char* p, unsigned long long n,
                                  unsigned int seed) {
  const unsigned long long i =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned int s = seed ^ static_cast<unsigned int>(i * 2654435761u);
  s = s * 1664525u + 1013904223u;
  s = s * 1664525u + 1013904223u;
  p[i] = static_cast<unsigned char>(s >> 24);
}

__global__ void CopyKernel(const unsigned char* __restrict__ src,
                           unsigned char* __restrict__ dst,
                           unsigned long long n) {
  const unsigned long long i =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = src[i];
}

std::size_t Compare(const std::vector<unsigned char>& a,
                    const std::vector<unsigned char>& b) {
  std::size_t d = 0;
  for (std::size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) ++d;
  return d;
}

// CPU load of the dma-buf: neither the GPU's L2 nor a CPU cache of a GPU
// allocation is in the path.
std::size_t DramCompare(int fd, const std::vector<unsigned char>& expected) {
  if (fd < 0) return static_cast<std::size_t>(-1);
  const std::size_t bytes = expected.size();
  void* m = mmap(nullptr, bytes, PROT_READ, MAP_SHARED, fd, 0);
  if (m == MAP_FAILED) return static_cast<std::size_t>(-2);
  const auto* p = static_cast<const unsigned char*>(m);
  std::size_t diff = 0;
  for (std::size_t i = 0; i < bytes; ++i) if (p[i] != expected[i]) ++diff;
  munmap(m, bytes);
  return diff;
}

double NowMs() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch())
      .count();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 6) {
    std::fprintf(stderr,
                 "usage: %s <xclbin> <insts.bin> <M> <K> <N> [reps] [dev|host]"
                 " [stress]\n",
                 argv[0]);
    return 2;
  }
  const std::size_t M = std::strtoull(argv[3], nullptr, 10);
  const std::size_t K = std::strtoull(argv[4], nullptr, 10);
  const std::size_t N = std::strtoull(argv[5], nullptr, 10);
  const int reps = argc > 6 ? std::atoi(argv[6]) : 20;
  const std::string alloc_mode = argc > 7 ? argv[7] : "dev";
  const int stress = argc > 8 ? std::atoi(argv[8]) : 1;
  // Idle time between timed launches, to test whether the NPU's clocks decay
  // when the engine leaves it idle between layers.
  const int gap_ms = argc > 9 ? std::atoi(argv[9]) : 0;
  const bool host_alloc = (alloc_mode == "host");
  const std::size_t a_vol = M * K * 9 / 8;
  const std::size_t b_vol = N * K * 9 / 8;
  const std::size_t c_vol = M * N * 9 / 8;

  std::vector<std::uint32_t> instr;
  {
    std::ifstream in(argv[2], std::ios::binary | std::ios::ate);
    if (!in) { std::fprintf(stderr, "cannot read %s\n", argv[2]); return 1; }
    const std::streamsize n = in.tellg();
    in.seekg(0);
    instr.resize(static_cast<std::size_t>(n) / sizeof(std::uint32_t));
    in.read(reinterpret_cast<char*>(instr.data()), n);
    if (!in) { std::fprintf(stderr, "short read %s\n", argv[2]); return 1; }
  }

  xrt::device device(0);
  auto xclbin = xrt::xclbin(std::string(argv[1]));
  const auto kernels = xclbin.get_kernels();
  auto it = kernels.begin();
  for (; it != kernels.end(); ++it) {
    if (it->get_name().rfind("MLIR_AIE", 0) == 0) break;
  }
  if (it == kernels.end()) { std::fprintf(stderr, "MLIR_AIE not found\n"); return 1; }
  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, it->get_name());

  auto bo_instr = xrt::bo(device, instr.size() * sizeof(std::uint32_t),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  std::memcpy(bo_instr.map<void*>(), instr.data(),
              instr.size() * sizeof(std::uint32_t));
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  std::printf("M=%zu K=%zu N=%zu  A=%.1f B=%.1f C=%.1f MB  instr=%zu alloc=%s"
              " stress=%d\n",
              M, K, N, a_vol / 1048576.0, b_vol / 1048576.0, c_vol / 1048576.0,
              instr.size(), alloc_mode.c_str(), stress);

  // Reference BOs, allocated once and refilled per iteration.
  auto ra = xrt::bo(device, a_vol, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto rb = xrt::bo(device, b_vol, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto rc = xrt::bo(device, c_vol, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
  std::vector<unsigned char> c_ref(c_vol);
  const auto reference_run = [&](const std::vector<unsigned char>& a,
                                 const std::vector<unsigned char>& b) -> int {
    std::memcpy(ra.map<void*>(), a.data(), a_vol);
    std::memcpy(rb.map<void*>(), b.data(), b_vol);
    std::memset(rc.map<void*>(), 0, c_vol);
    ra.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    rb.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    rc.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto run = kernel(kErtOpcode, bo_instr, instr.size(), ra, rb, rc);
    const int st = static_cast<int>(run.wait());
    rc.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    std::memcpy(c_ref.data(), rc.map<const unsigned char*>(), c_vol);
    return st;
  };
  {
    const auto a0 = FillHost(a_vol, 1u);
    const auto b0 = FillHost(b_vol, 2u);
    const int st = reference_run(a0, b0);
    std::size_t nz = 0;
    for (unsigned char v : c_ref) if (v) ++nz;
    std::printf("reference state=%d nonzero=%zu\n", st, nz);
    if (nz == 0) { std::fprintf(stderr, "reference C is all zero\n"); return 1; }
  }

  // Shared path: GPU-addressable allocations exported as dma-bufs.
  unsigned char* da = nullptr;
  unsigned char* db = nullptr;
  unsigned char* dc = nullptr;
  unsigned char* ka = nullptr;
  unsigned char* kb = nullptr;
  unsigned char* kc = nullptr;
  if (host_alloc) {
    void* pa = nullptr;
    void* pb = nullptr;
    void* pc = nullptr;
    if (hipHostMalloc(&pa, a_vol) != hipSuccess) return FailLast("hipHostMalloc a");
    if (hipHostMalloc(&pb, b_vol) != hipSuccess) return FailLast("hipHostMalloc b");
    if (hipHostMalloc(&pc, c_vol) != hipSuccess) return FailLast("hipHostMalloc c");
    ka = static_cast<unsigned char*>(pa);
    kb = static_cast<unsigned char*>(pb);
    kc = static_cast<unsigned char*>(pc);
    if (hipHostGetDevicePointer(reinterpret_cast<void**>(&da), pa, 0) != hipSuccess)
      return FailLast("hipHostGetDevicePointer a");
    if (hipHostGetDevicePointer(reinterpret_cast<void**>(&db), pb, 0) != hipSuccess)
      return FailLast("hipHostGetDevicePointer b");
    if (hipHostGetDevicePointer(reinterpret_cast<void**>(&dc), pc, 0) != hipSuccess)
      return FailLast("hipHostGetDevicePointer c");
  } else {
    if (hipMalloc(&da, a_vol) != hipSuccess) return FailLast("hipMalloc a");
    if (hipMalloc(&db, b_vol) != hipSuccess) return FailLast("hipMalloc b");
    if (hipMalloc(&dc, c_vol) != hipSuccess) return FailLast("hipMalloc c");
    ka = da; kb = db; kc = dc;
  }
  if (hsa_init() != HSA_STATUS_SUCCESS) { std::fprintf(stderr, "hsa_init failed\n"); return 3; }
  int fd_a = -1, fd_b = -1, fd_c = -1;
  std::uint64_t off = 0;
  if (hsa_amd_portable_export_dmabuf(ka, a_vol, &fd_a, &off) != HSA_STATUS_SUCCESS) return 4;
  if (hsa_amd_portable_export_dmabuf(kb, b_vol, &fd_b, &off) != HSA_STATUS_SUCCESS) return 4;
  if (hsa_amd_portable_export_dmabuf(kc, c_vol, &fd_c, &off) != HSA_STATUS_SUCCESS) return 4;
  auto bo_a = xrt::bo(device, static_cast<xrt::bo::export_handle>(fd_a));
  auto bo_b = xrt::bo(device, static_cast<xrt::bo::export_handle>(fd_b));
  auto bo_c = xrt::bo(device, static_cast<xrt::bo::export_handle>(fd_c));

  unsigned char* view = nullptr;
  if (hipMalloc(&view, c_vol) != hipSuccess) return FailLast("hipMalloc view");
  std::vector<unsigned char> got(c_vol);
  std::vector<unsigned char> got_a(a_vol);
  std::vector<unsigned char> got_b(b_vol);

  int failures = 0;
  for (int iter = 0; iter < stress; ++iter) {
    const unsigned int seed_a = 0x9e3779b9u + 0x1000u * static_cast<unsigned>(iter);
    const unsigned int seed_b = 0x85ebca6bu + 0x1000u * static_cast<unsigned>(iter);
    const auto a_host = FillHost(a_vol, seed_a);
    const auto b_host = FillHost(b_vol, seed_b);
    const int st_ref = reference_run(a_host, b_host);
    (void)st_ref;

    FillPatternKernel<<<static_cast<unsigned>((a_vol + 255) / 256), 256>>>(da, a_vol, seed_a);
    FillPatternKernel<<<static_cast<unsigned>((b_vol + 255) / 256), 256>>>(db, b_vol, seed_b);
    if (hipDeviceSynchronize() != hipSuccess) return FailLast("fill sync");

    // Before anything reads A or B back: are the GPU's stores in DRAM? A dirty
    // L2 line that has not been written back is what would be invisible to the
    // NPU, so this has to be the first check after the fill.
    const std::size_t dram_a = DramCompare(fd_a, a_host);
    const std::size_t dram_b = DramCompare(fd_b, b_host);
    if (hipMemcpy(got_a.data(), da, a_vol, hipMemcpyDeviceToHost) != hipSuccess)
      return FailLast("readback a");
    if (hipMemcpy(got_b.data(), db, b_vol, hipMemcpyDeviceToHost) != hipSuccess)
      return FailLast("readback b");

    auto run = kernel(kErtOpcode, bo_instr, instr.size(), bo_a, bo_b, bo_c);
    const int st = static_cast<int>(run.wait());
    CopyKernel<<<static_cast<unsigned>((c_vol + 255) / 256), 256>>>(dc, view, c_vol);
    if (hipDeviceSynchronize() != hipSuccess) return FailLast("copy sync");
    if (hipMemcpy(got.data(), view, c_vol, hipMemcpyDeviceToHost) != hipSuccess)
      return FailLast("hipMemcpy");
    const std::size_t gpu_diff = Compare(got, c_ref);
    const std::size_t dram_c = DramCompare(fd_c, c_ref);
    // If the output DMA outlived run.wait(), the immediate read would be short
    // and a delayed one would not. Compare again after the next run has had
    // time to drain.
    usleep(20000);
    const std::size_t dram_c_late = DramCompare(fd_c, c_ref);
    const std::size_t fill_a = Compare(got_a, a_host);
    const std::size_t fill_b = Compare(got_b, b_host);
    std::printf(
        "iter %d state=%d fill(A/B)=%zu/%zu dram(A/B)=%zu/%zu C: gpu=%zu dram=%zu late=%zu\n",
        iter, st, fill_a, fill_b, dram_a, dram_b, gpu_diff, dram_c, dram_c_late);
    if (fill_a || fill_b || dram_a || dram_b || gpu_diff || dram_c || dram_c_late)
      ++failures;
  }
  std::printf("stress %d iterations, %d with mismatches\n", stress, failures);

  // Timing on the last iteration's data.
  auto timed_run = [&]() {
    const double t0 = NowMs();
    auto run = kernel(kErtOpcode, bo_instr, instr.size(), bo_a, bo_b, bo_c);
    const int st = static_cast<int>(run.wait());
    const double t1 = NowMs();
    return std::pair<double, int>(t1 - t0, st);
  };
  for (int i = 0; i < 3; ++i) timed_run();
  double best = 1e30, total = 0.0;
  for (int i = 0; i < reps; ++i) {
    if (gap_ms > 0) usleep(static_cast<unsigned>(gap_ms) * 1000);
    auto [ms, st] = timed_run();
    if (st != 4) std::printf("  rep %d state=%d\n", i, st);
    if (ms < best) best = ms;
    total += ms;
  }
  const double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) *
                       static_cast<double>(K);
  std::printf("npu: best %.3f ms  mean %.3f ms  -> %.2f TFLOPS (best) gap=%dms\n",
              best, total / reps, flops / (best / 1000.0) / 1e12, gap_ms);
  std::printf("NPU_RUN %s\n", failures == 0 ? "PASS" : "FAIL");

  if (host_alloc) {
    (void)hipHostFree(ka); (void)hipHostFree(kb); (void)hipHostFree(kc);
  } else {
    (void)hipFree(da); (void)hipFree(db); (void)hipFree(dc);
  }
  (void)hipFree(view);
  return failures == 0 ? 0 : 1;
}
