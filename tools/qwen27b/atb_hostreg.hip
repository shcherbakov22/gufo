// Probe: can the iGPU address an XRT host-only BO directly?
//
//   atb_hostreg <xclbin> <insts.bin> <M> <K> <N>
//
// The NPU's dma-buf cannot be imported into HIP, and amdxdna rejects userptr
// BOs, so the shared-memory question is whether hipHostRegister on an XRT BO's
// own mapping yields a device pointer onto that same memory. If it does, the
// GPU writes A and B straight into the NPU's buffers and reads C straight back
// out of them: no staging copy in either direction.
//
// Two runs with identical A and B bytes are compared. The reference uses the
// managed host-only BOs and host memcpy. The hostreg run registers the BO
// mappings with hipHostRegister and fills A and B through the resulting device
// pointers, so the NPU must observe GPU-side writes and the GPU must observe
// NPU-side writes through one allocation.
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

// The ERT start-CU opcode for these AIE xclbins is the literal 3, not
// group_id(0). Passing the group id leaves C untouched.
constexpr unsigned int kErtOpcode = 3;

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

  auto bo_instr = xrt::bo(device, instr.size() * sizeof(std::uint32_t),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_a = xrt::bo(device, a_vol, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_b = xrt::bo(device, b_vol, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_c = xrt::bo(device, c_vol, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
  std::memcpy(bo_instr.map<void*>(), instr.data(),
              instr.size() * sizeof(std::uint32_t));

  std::vector<unsigned char> c_ref;
  {  // Reference: host memcpy in, host read out.
    std::memcpy(bo_a.map<void*>(), a_host.data(), a_vol);
    std::memcpy(bo_b.map<void*>(), b_host.data(), b_vol);
    std::memset(bo_c.map<void*>(), 0, c_vol);
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto run = kernel(kErtOpcode, bo_instr, instr.size(), bo_a, bo_b, bo_c);
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

  // Hostreg: register the BO mappings so HIP hands back device pointers onto
  // the NPU's own buffers.
  void* ha = bo_a.map<void*>();
  void* hb = bo_b.map<void*>();
  void* hc = bo_c.map<void*>();
  std::printf("xrt host ptrs a=%p b=%p c=%p\n", ha, hb, hc);
  const unsigned int reg_flags = hipHostRegisterMapped;
  hipError_t e = hipHostRegister(ha, a_vol, reg_flags);
  std::printf("hipHostRegister a -> %s (%d)\n", hipGetErrorString(e), (int)e);
  if (e != hipSuccess) return 4;
  e = hipHostRegister(hb, b_vol, reg_flags);
  std::printf("hipHostRegister b -> %s (%d)\n", hipGetErrorString(e), (int)e);
  if (e != hipSuccess) return 4;
  e = hipHostRegister(hc, c_vol, reg_flags);
  std::printf("hipHostRegister c -> %s (%d)\n", hipGetErrorString(e), (int)e);
  if (e != hipSuccess) return 4;

  void* da = nullptr;
  void* db = nullptr;
  void* dc = nullptr;
  e = hipHostGetDevicePointer(&da, ha, 0);
  if (e != hipSuccess || da == nullptr) return Fail("hipHostGetDevicePointer a", e);
  e = hipHostGetDevicePointer(&db, hb, 0);
  if (e != hipSuccess || db == nullptr) return Fail("hipHostGetDevicePointer b", e);
  e = hipHostGetDevicePointer(&dc, hc, 0);
  if (e != hipSuccess || dc == nullptr) return Fail("hipHostGetDevicePointer c", e);
  std::printf("hip device ptrs a=%p b=%p c=%p\n", da, db, dc);

  // GPU writes A and B, never the host: the NPU has to observe GPU stores.
  FillPatternKernel<<<static_cast<unsigned>((a_vol + 255) / 256), 256>>>(
      static_cast<unsigned char*>(da), a_vol, 0x9e3779b9u);
  FillPatternKernel<<<static_cast<unsigned>((b_vol + 255) / 256), 256>>>(
      static_cast<unsigned char*>(db), b_vol, 0x85ebca6bu);
  if (hipDeviceSynchronize() != hipSuccess) return FailLast("fill sync");

  auto run = kernel(kErtOpcode, bo_instr, instr.size(), bo_a, bo_b, bo_c);
  std::printf("hostreg run state=%d\n", static_cast<int>(run.wait()));

  std::size_t diff = 0;
  std::size_t nz = 0;
  const auto* ch = static_cast<const unsigned char*>(hc);
  for (std::size_t i = 0; i < c_vol; ++i) {
    if (ch[i] != c_ref[i]) ++diff;
    if (ch[i] != 0) ++nz;
  }
  std::printf("hostreg host view of C: nonzero=%zu differing=%zu\n", nz, diff);

  // And the GPU reading C back through its own pointer, which is the direction
  // the engine actually needs.
  unsigned char* d_view = nullptr;
  if (hipMalloc(&d_view, c_vol) != hipSuccess) return FailLast("hipMalloc");
  CopyKernel<<<static_cast<unsigned>((c_vol + 255) / 256), 256>>>(
      static_cast<const unsigned char*>(dc), d_view, c_vol);
  if (hipDeviceSynchronize() != hipSuccess) return FailLast("copy sync");
  std::vector<unsigned char> back(c_vol);
  if (hipMemcpy(back.data(), d_view, c_vol, hipMemcpyDeviceToHost) != hipSuccess)
    return FailLast("hipMemcpy");
  std::size_t diff2 = 0;
  for (std::size_t i = 0; i < c_vol; ++i) if (back[i] != c_ref[i]) ++diff2;
  std::printf("gpu view of C differing=%zu\n", diff2);

  const bool ok = (diff == 0 && diff2 == 0);
  std::printf("HOSTREG %s\n", ok ? "SHARED-MEMORY CONFIRMED" : "MISMATCH");
  (void)hipFree(d_view);
  return ok ? 0 : 1;
}
