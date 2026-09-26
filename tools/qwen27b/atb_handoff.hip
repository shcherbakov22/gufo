// Minimal XRT + HIP probe for the ATB NPU kernel's output handoff.
//
//   atb_handoff <xclbin> <insts.bin> <M> <K> <N>
//
// Runs the ATB kernel once on the NPU, then reads its output buffer from the GPU
// through a dma-buf import and compares the GPU's bytes against the host's view
// of the same buffer. Equality means the handoff is a mapping rather than a
// transfer, which is what the iGPU+NPU split rests on.
//
// Every call is checked before anything is launched. The first version of this
// did not, and a failed import left a null pointer that a GPU kernel then
// dereferenced: IOMMU page faults and a full ring timeout. ATB_HANDOFF_SKIP_GPU
// stops after the import and map so the API path can be probed without running
// any kernel at all.
#include <algorithm>
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

__global__ void AtbHandoffCopyKernel(const unsigned char* __restrict__ src,
                                     unsigned char* __restrict__ dst,
                                     unsigned long long n) {
  const unsigned long long i =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    dst[i] = src[i];
  }
}

int Fail(const char* what, hipError_t e) {
  std::fprintf(stderr, "HIP %s -> %s (%d)\n", what, hipGetErrorString(e),
               static_cast<int>(e));
  return 3;
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
    if (!in) {
      std::fprintf(stderr, "cannot read %s\n", argv[2]);
      return 1;
    }
    const std::streamsize n = in.tellg();
    in.seekg(0);
    instr.resize(static_cast<std::size_t>(n) / sizeof(std::uint32_t));
    in.read(reinterpret_cast<char*>(instr.data()), n);
  }
  std::printf("instr %zu words, A=%zu B=%zu C=%zu bytes\n", instr.size(), a_vol,
              b_vol, c_vol);

  xrt::device device(0);
  auto xclbin = xrt::xclbin(std::string(argv[1]));
  const std::string node = "MLIR_AIE";
  auto kernels = xclbin.get_kernels();
  auto it = std::find_if(kernels.begin(), kernels.end(),
                         [&](xrt::xclbin::kernel& k) {
                           return k.get_name().rfind(node, 0) == 0;
                         });
  if (it == kernels.end()) {
    std::fprintf(stderr, "kernel %s not found\n", node.c_str());
    return 1;
  }
  const std::string kernel_name = it->get_name();
  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, kernel_name);
  const auto opcode = kernel.group_id(0);

  auto bo_instr = xrt::bo(device, instr.size() * sizeof(std::uint32_t),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_a = xrt::bo(device, a_vol, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_b = xrt::bo(device, b_vol, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_out = xrt::bo(device, c_vol, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

  std::memcpy(bo_instr.map<void*>(), instr.data(),
              instr.size() * sizeof(std::uint32_t));
  std::memset(bo_a.map<void*>(), 0x11, a_vol);
  std::memset(bo_b.map<void*>(), 0x11, b_vol);
  std::memset(bo_out.map<void*>(), 0, c_vol);
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto run = kernel(opcode, bo_instr, instr.size(), bo_a, bo_b, bo_out);
  run.wait();
  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  std::printf("NPU run complete\n");

  unsigned char* host_view = bo_out.map<unsigned char*>();

  const auto handle = bo_out.export_buffer();
  std::printf("export -> fd %d\n", static_cast<int>(handle));
  if (static_cast<int>(handle) < 0) {
    std::fprintf(stderr, "HANDOFF ABORT: export_buffer failed\n");
    return 3;
  }
  hipExternalMemoryHandleDesc hd{};
  hd.type = hipExternalMemoryHandleTypeOpaqueFd;
  hd.handle.fd = static_cast<int>(handle);
  hd.size = c_vol;
  hipExternalMemory_t ext = nullptr;
  hipError_t e = hipImportExternalMemory(&ext, &hd);
  std::printf("import -> %s (%d)\n", hipGetErrorString(e),
              static_cast<int>(e));
  if (e != hipSuccess) {
    std::fprintf(stderr, "HANDOFF ABORT at import\n");
    return 3;
  }
  hipExternalMemoryBufferDesc bd{};
  bd.offset = 0;
  bd.size = c_vol;
  void* mapped = nullptr;
  e = hipExternalMemoryGetMappedBuffer(&mapped, ext, &bd);
  std::printf("map -> %s (%d), ptr=%p\n", hipGetErrorString(e),
              static_cast<int>(e), mapped);
  if (e != hipSuccess || mapped == nullptr) {
    std::fprintf(stderr, "HANDOFF ABORT at map\n");
    return 3;
  }
  if (std::getenv("ATB_HANDOFF_SKIP_GPU") != nullptr) {
    std::printf("HANDOFF SKIP_GPU: import and map both succeeded\n");
    return 0;
  }

  void* d_dst = nullptr;
  e = hipMalloc(&d_dst, c_vol);
  if (e != hipSuccess) return Fail("hipMalloc", e);
  AtbHandoffCopyKernel<<<static_cast<unsigned>((c_vol + 255) / 256), 256>>>(
      static_cast<const unsigned char*>(mapped),
      static_cast<unsigned char*>(d_dst), c_vol);
  e = hipDeviceSynchronize();
  if (e != hipSuccess) return Fail("hipDeviceSynchronize", e);
  std::vector<unsigned char> got(c_vol);
  e = hipMemcpy(got.data(), d_dst, c_vol, hipMemcpyDeviceToHost);
  if (e != hipSuccess) return Fail("hipMemcpy", e);

  std::size_t diff = 0;
  for (std::size_t i = 0; i < c_vol; ++i) {
    if (got[i] != host_view[i]) ++diff;
  }
  std::printf("HANDOFF bytes=%zu differing=%zu\n", c_vol, diff);
  std::printf("first bytes host:");
  for (int i = 0; i < 9 && static_cast<std::size_t>(i) < c_vol; ++i)
    std::printf(" %u", static_cast<unsigned>(host_view[i]));
  std::printf("\nfirst bytes gpu :");
  for (int i = 0; i < 9 && static_cast<std::size_t>(i) < c_vol; ++i)
    std::printf(" %u", static_cast<unsigned>(got[i]));
  std::printf("\n");
  hipFree(d_dst);
  hipDestroyExternalMemory(ext);
  return diff == 0 ? 0 : 1;
}
