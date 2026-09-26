// Can the NPU consume the iGPU's own allocation through a dma-buf?
//
//   atb_dmabuf <xclbin> <insts.bin> <M> <K> <N>
//
// An XRT host-only BO is slow memory for HIP (11 GB/s device-to-host) and
// hipHostRegister refuses its mapping, so a copy handoff costs more than the
// split saves. This tests the other direction: allocate A, B and C with
// hipMalloc, export each as an amdgpu dma-buf with hsa_amd_portable_export_dmabuf,
// and hand the resulting file descriptors to XRT as imported BOs. If the ATB
// kernel accepts them and produces the reference C, the GPU and the NPU share
// one allocation and neither direction needs a copy.
//
// The reference run uses the managed host-only BOs and host memcpy, so the C
// bytes are the ground truth for the same A and B inputs.
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
  std::memcpy(bo_instr.map<void*>(), instr.data(),
              instr.size() * sizeof(std::uint32_t));
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  std::vector<unsigned char> c_ref;
  {  // Reference: XRT-managed host-only BOs, host memcpy in and out.
    auto bo_a = xrt::bo(device, a_vol, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    auto bo_b = xrt::bo(device, b_vol, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
    auto bo_c = xrt::bo(device, c_vol, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
    std::memcpy(bo_a.map<void*>(), a_host.data(), a_vol);
    std::memcpy(bo_b.map<void*>(), b_host.data(), b_vol);
    std::memset(bo_c.map<void*>(), 0, c_vol);
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

  hsa_status_t hs = hsa_init();
  if (hs != HSA_STATUS_SUCCESS) {
    std::fprintf(stderr, "hsa_init -> %d\n", static_cast<int>(hs));
    return 3;
  }

  unsigned char* da = nullptr;
  unsigned char* db = nullptr;
  unsigned char* dc = nullptr;
  if (hipMalloc(&da, a_vol) != hipSuccess) return FailLast("hipMalloc a");
  if (hipMalloc(&db, b_vol) != hipSuccess) return FailLast("hipMalloc b");
  if (hipMalloc(&dc, c_vol) != hipSuccess) return FailLast("hipMalloc c");
  if (hipMemcpy(da, a_host.data(), a_vol, hipMemcpyHostToDevice) != hipSuccess)
    return FailLast("hipMemcpy a");
  if (hipMemcpy(db, b_host.data(), b_vol, hipMemcpyHostToDevice) != hipSuccess)
    return FailLast("hipMemcpy b");
  if (hipMemset(dc, 0, c_vol) != hipSuccess) return FailLast("hipMemset c");

  int fd_a = -1, fd_b = -1, fd_c = -1;
  std::uint64_t off_a = 0, off_b = 0, off_c = 0;
  const auto export_one = [&](void* ptr, std::size_t sz, int* fd, std::uint64_t* off,
                              const char* what) -> bool {
    const hsa_status_t s =
        hsa_amd_portable_export_dmabuf(ptr, sz, fd, off);
    std::printf("export %s -> %d fd=%d offset=%lld\n", what, static_cast<int>(s),
                *fd, static_cast<long long>(*off));
    return s == HSA_STATUS_SUCCESS && *fd >= 0;
  };
  if (!export_one(da, a_vol, &fd_a, &off_a, "A")) return 4;
  if (!export_one(db, b_vol, &fd_b, &off_b, "B")) return 4;
  if (!export_one(dc, c_vol, &fd_c, &off_c, "C")) return 4;

  try {
    auto bo_a = xrt::bo(device, static_cast<xrt::bo::export_handle>(fd_a));
    std::printf("imported A bo ok\n");
    auto bo_b = xrt::bo(device, static_cast<xrt::bo::export_handle>(fd_b));
    std::printf("imported B bo ok\n");
    auto bo_c = xrt::bo(device, static_cast<xrt::bo::export_handle>(fd_c));
    std::printf("imported C bo ok\n");
    auto run = kernel(kErtOpcode, bo_instr, instr.size(), bo_a, bo_b, bo_c);
    std::printf("dmabuf run state=%d\n", static_cast<int>(run.wait()));
  } catch (const std::exception& e) {
    std::printf("DMABUF ABORT: %s\n", e.what());
    return 5;
  }

  std::vector<unsigned char> got(c_vol);
  if (hipMemcpy(got.data(), dc, c_vol, hipMemcpyDeviceToHost) != hipSuccess)
    return FailLast("hipMemcpy c");
  std::size_t diff = 0;
  std::size_t nz = 0;
  for (std::size_t i = 0; i < c_vol; ++i) {
    if (got[i] != c_ref[i]) ++diff;
    if (got[i] != 0) ++nz;
  }
  std::printf("dmabuf C in GPU memory: nonzero=%zu differing=%zu first9:", nz, diff);
  for (int i = 0; i < 9; ++i) std::printf(" %u", got[i]);
  std::printf("\n");
  const bool ok = (diff == 0);
  std::printf("DMABUF %s\n", ok ? "GPU-MEMORY SHARING CONFIRMED" : "MISMATCH");
  return ok ? 0 : 1;
}
