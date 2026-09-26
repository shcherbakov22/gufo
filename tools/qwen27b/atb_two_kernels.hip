// Can two ATB xclbins be resident in one process?
//
//   atb_two_kernels <xclbin1> <insts1> <M1> <K1> <N1> <xclbin2> <insts2> <M2> <K2> <N2>
//
// One xclbin per shape is the constraint the split currently lives with: an
// instruction stream from another shape aborts the device queue, so a process
// holds exactly one K and N. If two hardware contexts can coexist, the down
// projection can be offloaded alongside gate and up and the GPU stops waiting
// on the NPU before it can start down.
//
// State 4 is ERT_CMD_STATE_COMPLETED. Nonzero C from both runs means both
// kernels really executed.
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

constexpr unsigned int kErtOpcode = 3;

std::vector<std::uint32_t> ReadInstr(const char* path) {
  std::vector<std::uint32_t> instr;
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) return instr;
  const std::streamsize bytes = in.tellg();
  in.seekg(0);
  instr.resize(static_cast<std::size_t>(bytes) / sizeof(std::uint32_t));
  in.read(reinterpret_cast<char*>(instr.data()), bytes);
  return instr;
}

struct Slice {
  std::size_t m, k, n;
  xrt::xclbin xclbin;
  xrt::hw_context context;
  xrt::kernel kernel;
  xrt::bo instr;
  xrt::bo a, b, c;
  unsigned words{0};
};

bool Prepare(xrt::device& device, const char* xclbin_path, const char* insts_path,
             std::size_t m, std::size_t k, std::size_t n, Slice* out,
             const char* label) {
  out->m = m;
  out->k = k;
  out->n = n;
  const auto instr = ReadInstr(insts_path);
  if (instr.empty()) {
    std::fprintf(stderr, "%s: cannot read %s\n", label, insts_path);
    return false;
  }
  try {
    out->xclbin = xrt::xclbin(std::string(xclbin_path));
    const auto kernels = out->xclbin.get_kernels();
    auto it = kernels.begin();
    for (; it != kernels.end(); ++it) {
      if (it->get_name().rfind("MLIR_AIE", 0) == 0) break;
    }
    if (it == kernels.end()) {
      std::fprintf(stderr, "%s: no MLIR_AIE kernel\n", label);
      return false;
    }
    device.register_xclbin(out->xclbin);
    out->context = xrt::hw_context(device, out->xclbin.get_uuid());
    out->kernel = xrt::kernel(out->context, it->get_name());
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s: context failed: %s\n", label, e.what());
    return false;
  }
  out->instr = xrt::bo(device, instr.size() * sizeof(std::uint32_t),
                       XCL_BO_FLAGS_CACHEABLE, out->kernel.group_id(1));
  std::memcpy(out->instr.map<void*>(), instr.data(),
              instr.size() * sizeof(std::uint32_t));
  out->instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  out->words = static_cast<unsigned>(instr.size());
  const std::size_t a_bytes = m * k * 9 / 8;
  const std::size_t b_bytes = n * k * 9 / 8;
  const std::size_t c_bytes = m * n * 9 / 8;
  try {
    out->a = xrt::bo(device, a_bytes, XRT_BO_FLAGS_HOST_ONLY, out->kernel.group_id(3));
    out->b = xrt::bo(device, b_bytes, XRT_BO_FLAGS_HOST_ONLY, out->kernel.group_id(4));
    out->c = xrt::bo(device, c_bytes, XRT_BO_FLAGS_HOST_ONLY, out->kernel.group_id(5));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s: bo failed: %s\n", label, e.what());
    return false;
  }
  // Constant bytes are numerically degenerate here: a repeated 0x21 exponent
  // byte means a 2^-94 shared exponent, so every product underflows and C is
  // zero whether or not the kernel ran. Use the pseudorandom pattern instead.
  const auto fill = [](void* dst, std::size_t bytes, unsigned int seed) {
    auto* p = static_cast<unsigned char*>(dst);
    for (std::size_t i = 0; i < bytes; ++i) {
      unsigned int s = seed ^ static_cast<unsigned int>(i * 2654435761u);
      s = s * 1664525u + 1013904223u;
      s = s * 1664525u + 1013904223u;
      p[i] = static_cast<unsigned char>(s >> 24);
    }
  };
  fill(out->a.map<void*>(), a_bytes, 0x9e3779b9u);
  fill(out->b.map<void*>(), b_bytes, 0x85ebca6bu);
  std::memset(out->c.map<void*>(), 0, c_bytes);
  out->a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  out->b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  out->c.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  std::printf("%s: prepared M=%zu K=%zu N=%zu\n", label, m, k, n);
  return true;
}

bool Run(Slice* s, const char* label) {
  try {
    auto run = s->kernel(kErtOpcode, s->instr, s->words, s->a, s->b, s->c);
    const int state = static_cast<int>(run.wait());
    s->c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    const auto* p = s->c.map<const unsigned char*>();
    const std::size_t bytes = s->m * s->n * 9 / 8;
    std::size_t nonzero = 0;
    for (std::size_t i = 0; i < bytes; ++i) if (p[i] != 0) ++nonzero;
    std::printf("%s: state=%d C nonzero=%zu/%zu\n", label, state, nonzero, bytes);
    return state == 4 && nonzero == bytes;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s: run failed: %s\n", label, e.what());
    return false;
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 11) {
    std::fprintf(stderr,
                 "usage: %s <xclbin1> <insts1> <M1> <K1> <N1> <xclbin2> <insts2> "
                 "<M2> <K2> <N2>\n",
                 argv[0]);
    return 2;
  }
  xrt::device device(0);
  Slice first;
  Slice second;
  if (!Prepare(device, argv[1], argv[2], std::strtoull(argv[3], nullptr, 10),
               std::strtoull(argv[4], nullptr, 10),
               std::strtoull(argv[5], nullptr, 10), &first, "gu")) {
    return 3;
  }
  if (!Prepare(device, argv[6], argv[7], std::strtoull(argv[8], nullptr, 10),
               std::strtoull(argv[9], nullptr, 10),
               std::strtoull(argv[10], nullptr, 10), &second, "dn")) {
    std::printf("SECOND_CONTEXT_REFUSED\n");
    return 4;
  }
  const bool ok_first = Run(&first, "gu");
  const bool ok_second = Run(&second, "dn");
  const bool ok_first_again = Run(&first, "gu-again");
  std::printf("two xclbins %s\n",
              (ok_first && ok_second && ok_first_again) ? "COEXIST"
                                                        : "FAILED");
  return (ok_first && ok_second && ok_first_again) ? 0 : 1;
}
