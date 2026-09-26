// Probe: can an XRT BO wrap the iGPU's own hipHostMalloc memory?
//
//   atb_userptr <xclbin> <insts.bin> <M> <K> <N>
//
// Runs the ATB kernel twice with identical A and B bytes. The reference run uses
// XRT-managed host-only BOs filled by host memcpy. The userptr run allocates A,
// B and C with hipHostMalloc and wraps those same pointers as XRT BOs, then
// fills A and B with a GPU kernel, so the data path is GPU write -> NPU read ->
// GPU read with no copy anywhere. Byte equality of the two C buffers means the
// NPU and the iGPU really do see one allocation, which is what lets the engine
// drop the staging copies entirely.
//
// Every XRT call is inside a try/catch and every HIP call is checked: an
// earlier unchecked import left a null pointer that a kernel dereferenced and
// took the ring down.
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

}  // namespace

int main(int argc, char** argv) {
  if (argc < 6) {
    std::fprintf(stderr, "usage: %s <xclbin> <insts.bin> <M> <K> <N>\n", argv[0]);
    return 2;
  }
  const std::size_t M = std::strtoull(argv[3], nullptr, 10);
  const std::size_t K = std::strtoull(argv[4], nullptr, 10);
  const std::size_t N = std::strtoull(argv[5], nullptr, 10);
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
  std::printf("instr %zu words, A=%zu B=%zu C=%zu bytes\n", instr.size(), a_vol,
              b_vol, c_vol);

  const std::vector<unsigned char> a_host = FillHost(a_vol, 0x9e3779b9u);
  const std::vector<unsigned char> b_host = FillHost(b_vol, 0x85ebca6bu);

  xrt::device device(0);
  auto xclbin = xrt::xclbin(std::string(argv[1]));
  const std::string node = "MLIR_AIE";
  const auto kernels = xclbin.get_kernels();
  auto it = kernels.begin();
  for (; it != kernels.end(); ++it) {
    if (it->get_name().rfind(node, 0) == 0) break;
  }
  if (it == kernels.end()) {
    std::fprintf(stderr, "kernel %s not found\n", node.c_str());
    return 1;
  }
  const std::string kernel_name = it->get_name();
  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, kernel_name);
  // The ERT start-CU opcode for these AIE xclbins is the literal 3, not
  // group_id(0). Passing the group id silently leaves C untouched, which made
  // an earlier handoff probe compare two all-zero buffers and "pass".
  const unsigned int opcode = 3;

  std::vector<unsigned char> c_ref;
  {  // Reference: XRT-managed host-only BOs.
    auto bo_instr = xrt::bo(device, instr.size() * sizeof(std::uint32_t),
                            XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    auto bo_a = xrt::bo(device, a_vol, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    auto bo_b = xrt::bo(device, b_vol, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
    auto bo_c = xrt::bo(device, c_vol, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
    std::memcpy(bo_instr.map<void*>(), instr.data(),
                instr.size() * sizeof(std::uint32_t));
    std::memcpy(bo_a.map<void*>(), a_host.data(), a_vol);
    std::memcpy(bo_b.map<void*>(), b_host.data(), b_vol);
    std::memset(bo_c.map<void*>(), 0, c_vol);
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto run = kernel(opcode, bo_instr, instr.size(), bo_a, bo_b, bo_c);
    std::printf("reference run state=%d\n", static_cast<int>(run.wait()));
    bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    const auto* p = bo_c.map<const unsigned char*>();
    c_ref.assign(p, p + c_vol);
  }
  std::size_t nonzero = 0;
  for (unsigned char v : c_ref) if (v != 0) ++nonzero;
  std::printf("reference C: bytes=%zu nonzero=%zu first9:", c_ref.size(), nonzero);
  for (int i = 0; i < 9; ++i) std::printf(" %u", c_ref[i]);
  std::printf("\n");
  if (nonzero == 0) {
    std::fprintf(stderr, "reference produced an all-zero C; the probe is void\n");
    return 1;
  }

  // Userptr: the iGPU's own host allocations wrapped as XRT BOs.
  unsigned char* ua = nullptr;
  unsigned char* ub = nullptr;
  unsigned char* uc = nullptr;
  if (hipHostMalloc(&ua, a_vol) != hipSuccess) return FailLast("hipHostMalloc a");
  if (hipHostMalloc(&ub, b_vol) != hipSuccess) return FailLast("hipHostMalloc b");
  if (hipHostMalloc(&uc, c_vol) != hipSuccess) return FailLast("hipHostMalloc c");
  std::memset(uc, 0, c_vol);
  // A and B are written by the GPU, never the host, so the NPU must observe
  // GPU-side writes through the shared mapping.
  FillPatternKernel<<<static_cast<unsigned>((a_vol + 255) / 256), 256>>>(ua, a_vol, 0x9e3779b9u);
  FillPatternKernel<<<static_cast<unsigned>((b_vol + 255) / 256), 256>>>(ub, b_vol, 0x85ebca6bu);
  if (hipDeviceSynchronize() != hipSuccess) return FailLast("fill sync");

  std::printf("userptr host ptrs a=%p b=%p c=%p\n", (void*)ua, (void*)ub, (void*)uc);
  struct FlagOpt { const char* name; unsigned int f; };
  const FlagOpt flag_opts[] = {
      {"host_only", XRT_BO_FLAGS_HOST_ONLY},
      {"normal", XRT_BO_FLAGS_NONE},
      {"device_only", XRT_BO_FLAGS_DEV_ONLY},
      {"svm", XRT_BO_FLAGS_SVM},
      {"cacheable", XRT_BO_FLAGS_CACHEABLE},
      {"host_only|cacheable", XRT_BO_FLAGS_HOST_ONLY | XRT_BO_FLAGS_CACHEABLE},
  };
  int chosen = -1;
  for (unsigned i = 0; i < sizeof(flag_opts) / sizeof(flag_opts[0]); ++i) {
    try {
      auto probe = xrt::bo(context, ua, a_vol,
                          static_cast<xrt::bo::flags>(flag_opts[i].f),
                          kernel.group_id(3));
      std::printf("userptr flags %-20s -> accepted\n", flag_opts[i].name);
      chosen = static_cast<int>(i);
      break;
    } catch (const std::exception& e) {
      std::printf("userptr flags %-20s -> %s\n", flag_opts[i].name, e.what());
    }
  }
  if (chosen < 0) {
    std::printf("USERPTR UNSUPPORTED: no flag accepted a userptr buffer\n");
    return 4;
  }
  try {
    auto bo_instr_u = xrt::bo(device, instr.size() * sizeof(std::uint32_t),
                              XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    const auto f = static_cast<xrt::bo::flags>(flag_opts[chosen].f);
    auto bo_a_u = xrt::bo(context, ua, a_vol, f, kernel.group_id(3));
    auto bo_b_u = xrt::bo(context, ub, b_vol, f, kernel.group_id(4));
    auto bo_c_u = xrt::bo(context, uc, c_vol, f, kernel.group_id(5));
    std::memcpy(bo_instr_u.map<void*>(), instr.data(),
                instr.size() * sizeof(std::uint32_t));
    bo_instr_u.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto run = kernel(opcode, bo_instr_u, instr.size(), bo_a_u, bo_b_u, bo_c_u);
    std::printf("userptr run state=%d\n", static_cast<int>(run.wait()));
    std::printf("userptr run complete\n");
  } catch (const std::exception& e) {
    std::printf("USERPTR ABORT: %s\n", e.what());
    return 4;
  }

  std::size_t diff = 0;
  std::size_t nz_u = 0;
  for (std::size_t i = 0; i < c_vol; ++i) {
    if (uc[i] != c_ref[i]) ++diff;
    if (uc[i] != 0) ++nz_u;
  }
  std::printf("userptr C: nonzero=%zu differing=%zu first9:", nz_u, diff);
  for (int i = 0; i < 9; ++i) std::printf(" %u", uc[i]);
  std::printf("\n");

  // Round trip through the GPU: proves the iGPU can read the NPU's writes back
  // out of the same allocation.
  unsigned char* d_view = nullptr;
  if (hipMalloc(&d_view, c_vol) != hipSuccess) return FailLast("hipMalloc");
  CopyKernel<<<static_cast<unsigned>((c_vol + 255) / 256), 256>>>(uc, d_view, c_vol);
  if (hipDeviceSynchronize() != hipSuccess) return FailLast("copy sync");
  std::vector<unsigned char> back(c_vol);
  if (hipMemcpy(back.data(), d_view, c_vol, hipMemcpyDeviceToHost) != hipSuccess)
    return FailLast("hipMemcpy");
  std::size_t diff2 = 0;
  for (std::size_t i = 0; i < c_vol; ++i) if (back[i] != c_ref[i]) ++diff2;
  std::printf("gpu readback differing=%zu\n", diff2);

  const bool ok = (diff == 0 && diff2 == 0);
  std::printf("USERPTR %s\n", ok ? "SHARED-MEMORY CONFIRMED" : "MISMATCH");
  (void)hipFree(d_view);
  (void)hipHostFree(ua);
  (void)hipHostFree(ub);
  (void)hipHostFree(uc);
  return ok ? 0 : 1;
}
