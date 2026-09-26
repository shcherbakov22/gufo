// Host-side owner of the NPU half of the FFN split.
//
// Everything here exists because of one measurement: an XRT host-only BO costs
// 11 GB/s to read back through HIP, while a hipMalloc allocation exported as an
// amdgpu dma-buf and imported by XRT is memory both engines address directly.
// So A, B and C live in the iGPU's allocations and the NPU reads and writes
// them in place. tools/qwen27b/atb_npu_run.hip is the standalone proof, and it
// is also where the ERT opcode (the literal 3) and the flag rejections below
// were found.
#include "src/models/qwen/hip/atb_npu.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#if defined(GUFO_ENABLE_NPU)
#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <unistd.h>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#endif

namespace gufo::hip {
namespace {

// An ATB kernel owns the whole 8x4 array, so one xclbin fixes K and N. Two
// hardware contexts coexist on this device, which is what lets gate/up and down
// hold different shapes at the same time.
struct AtbShape {
  const char* xclbin_env;
  const char* insts_env;
  const char* nslice_env;
  const char* label;
  std::size_t k;
  std::size_t n_full;
};

[[nodiscard]] const AtbShape& ShapeFor(AtbRole role) {
  static const AtbShape gate_up{"GUFO_ATB_GU_XCLBIN",
                                "GUFO_ATB_GU_INSTS",
                                "GUFO_ATB_GU_NSLICE",
                                "gate/up",
                                5120,
                                17408};
  static const AtbShape down{"GUFO_ATB_DN_XCLBIN",
                             "GUFO_ATB_DN_INSTS",
                             "GUFO_ATB_DN_NSLICE",
                             "down",
                             17408,
                             5120};
  return role == AtbRole::kGateUp ? gate_up : down;
}

constexpr std::size_t kAtbL1N = 128;
constexpr std::size_t kAtbL1K = 64;

// The ERT start-CU opcode for these xclbins. It is not group_id(0); passing the
// group id leaves C untouched and the split would silently read stale data.
constexpr unsigned int kErtOpcode = 3;

std::size_t AlignDown(std::size_t value, std::size_t quantum) {
  return value - value % quantum;
}

std::size_t EnvSize(const char* name) {
  const char* text = std::getenv(name);
  if (text == nullptr) {
    return 0;
  }
  return static_cast<std::size_t>(std::strtoull(text, nullptr, 10));
}

}  // namespace

#if defined(GUFO_ENABLE_NPU)

namespace {

// XRT's run::wait() polls command completion in a tight loop of ioctls, which
// costs a full core of system time for the entire wait. On this box CPU power
// is drawn from the same SoC budget as the GPU and the NPU, so that polling
// directly competes with the work it is waiting for. Checking the state and
// sleeping between checks keeps the same completion guarantee for a small
// fraction of the cycles.
// Values from ert.h; they are an ABI constant of the command buffer.
constexpr int kErtNew = 1;
constexpr int kErtQueued = 2;
constexpr int kErtRunning = 3;
constexpr int kErtCompleted = 4;
constexpr int kErtSubmitted = 7;

bool WaitSleeping(xrt::run& run) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(20);
  for (;;) {
    const int state = static_cast<int>(run.state());
    if (state == kErtCompleted) {
      return true;
    }
    if (state != kErtNew && state != kErtQueued && state != kErtRunning &&
        state != kErtSubmitted) {
      return false;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(200));
  }
  return static_cast<int>(run.wait()) == kErtCompleted;
}

std::vector<std::uint32_t> ReadInstr(const char* path) {
  std::vector<std::uint32_t> instr;
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) {
    return instr;
  }
  const std::streamsize bytes = in.tellg();
  if (bytes <= 0 ||
      bytes % static_cast<std::streamsize>(sizeof(std::uint32_t))) {
    return instr;
  }
  in.seekg(0);
  instr.resize(static_cast<std::size_t>(bytes) / sizeof(std::uint32_t));
  in.read(reinterpret_cast<char*>(instr.data()), bytes);
  if (!in) {
    instr.clear();
  }
  return instr;
}

}  // namespace

struct AtbNpuOffload::Impl {
  xrt::device device{0};
  xrt::xclbin xclbin;
  xrt::hw_context context;
  xrt::kernel kernel;
  xrt::bo bo_instr;
  xrt::bo bo_a;
  xrt::bo bo_b_gate;
  xrt::bo bo_b_up;
  xrt::bo bo_c_gate;
  xrt::bo bo_c_up;
  xrt::run run_gate;
  xrt::run run_up;
  void* a{nullptr};
  void* b_gate{nullptr};
  void* b_up{nullptr};
  void* c_gate{nullptr};
  void* c_up{nullptr};
  int fd_a{-1};
  int fd_b_gate{-1};
  int fd_b_up{-1};
  int fd_c_gate{-1};
  int fd_c_up{-1};
  unsigned int instr_words{0};
};

AtbNpuOffload::AtbNpuOffload() = default;

AtbNpuOffload::~AtbNpuOffload() {
  if (impl_ == nullptr) {
    return;
  }
  // The BOs own their import; release the exporter side in the reverse order.
  delete impl_;
  impl_ = nullptr;
}

void* AtbNpuOffload::activations() const {
  return impl_ == nullptr ? nullptr : impl_->a;
}
void* AtbNpuOffload::gate_weights() const {
  return impl_ == nullptr ? nullptr : impl_->b_gate;
}
void* AtbNpuOffload::up_weights() const {
  return impl_ == nullptr ? nullptr : impl_->b_up;
}
void* AtbNpuOffload::gate_output() const {
  return impl_ == nullptr ? nullptr : impl_->c_gate;
}
void* AtbNpuOffload::up_output() const {
  return impl_ == nullptr ? nullptr : impl_->c_up;
}

bool AtbNpuOffload::Init(AtbRole role) {
  const AtbShape& shape = ShapeFor(role);
  const char* xclbin_path = std::getenv(shape.xclbin_env);
  const char* insts_path = std::getenv(shape.insts_env);
  if (xclbin_path == nullptr || insts_path == nullptr) {
    return false;
  }
  const std::size_t n_slice = EnvSize(shape.nslice_env);
  if (n_slice == 0 || n_slice % kAtbL1N != 0 || n_slice >= shape.n_full) {
    std::fprintf(stderr,
                 "atb-npu: %s must be a positive multiple of %zu below %zu\n",
                 shape.nslice_env, kAtbL1N, shape.n_full);
    return false;
  }
  const std::size_t n_gpu = shape.n_full - n_slice;
  if (n_gpu % kAtbL1N != 0) {
    std::fprintf(stderr, "atb-npu: the GPU share %zu is not tile aligned\n",
                 n_gpu);
    return false;
  }
  const std::size_t batch = EnvSize("GUFO_ATB_BATCH");
  if (batch == 0 || batch % 512 != 0) {
    std::fprintf(stderr,
                 "atb-npu: GUFO_ATB_BATCH must be a positive multiple "
                 "of 512\n");
    return false;
  }

  const auto instr = ReadInstr(insts_path);
  if (instr.empty()) {
    std::fprintf(stderr, "atb-npu: cannot read %s\n", insts_path);
    return false;
  }

  geometry_.batch = batch;
  geometry_.gu_k = shape.k;
  geometry_.gu_n_full = shape.n_full;
  geometry_.gu_n_gpu = n_gpu;

  const std::size_t a_bytes = batch * shape.k * 9 / 8;
  const std::size_t b_bytes = n_slice * shape.k * 9 / 8;
  const std::size_t c_bytes = batch * n_slice * 9 / 8;

  try {
    impl_ = new Impl();
    Impl& p = *impl_;
    p.xclbin = xrt::xclbin(std::string(xclbin_path));
    const auto kernels = p.xclbin.get_kernels();
    auto chosen = kernels.begin();
    for (; chosen != kernels.end(); ++chosen) {
      if (chosen->get_name().rfind("MLIR_AIE", 0) == 0) {
        break;
      }
    }
    if (chosen == kernels.end()) {
      std::fprintf(stderr, "atb-npu: no MLIR_AIE kernel in %s\n", xclbin_path);
      return false;
    }
    p.device.register_xclbin(p.xclbin);
    p.context = xrt::hw_context(p.device, p.xclbin.get_uuid());
    p.kernel = xrt::kernel(p.context, chosen->get_name());

    p.bo_instr = xrt::bo(p.device, instr.size() * sizeof(std::uint32_t),
                         XCL_BO_FLAGS_CACHEABLE, p.kernel.group_id(1));
    std::memcpy(p.bo_instr.map<void*>(), instr.data(),
                instr.size() * sizeof(std::uint32_t));
    p.bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    p.instr_words = static_cast<unsigned int>(instr.size());

    const auto import = [&](const std::size_t bytes, const int group,
                            void** host, int* fd, xrt::bo* bo) -> bool {
      if (hipMalloc(host, bytes) != hipSuccess) {
        std::fprintf(stderr, "atb-npu: hipMalloc %zu failed\n", bytes);
        return false;
      }
      std::uint64_t offset = 0;
      if (hsa_amd_portable_export_dmabuf(*host, bytes, fd, &offset) !=
          HSA_STATUS_SUCCESS) {
        std::fprintf(stderr, "atb-npu: dma-buf export of %zu bytes failed\n",
                     bytes);
        return false;
      }
      *bo = xrt::bo(p.device, static_cast<xrt::bo::export_handle>(*fd));
      (void)group;
      return true;
    };
    if (hsa_init() != HSA_STATUS_SUCCESS) {
      std::fprintf(stderr, "atb-npu: hsa_init failed\n");
      return false;
    }
    if (!import(a_bytes, 3, &p.a, &p.fd_a, &p.bo_a) ||
        !import(b_bytes, 4, &p.b_gate, &p.fd_b_gate, &p.bo_b_gate) ||
        !import(b_bytes, 4, &p.b_up, &p.fd_b_up, &p.bo_b_up) ||
        !import(c_bytes, 5, &p.c_gate, &p.fd_c_gate, &p.bo_c_gate) ||
        !import(c_bytes, 5, &p.c_up, &p.fd_c_up, &p.bo_c_up)) {
      return false;
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "atb-npu: %s\n", e.what());
    return false;
  } catch (...) {
    // XRT can also fail from pieces that are not std::exception. A
    // misconfigured NPU has to degrade to the GPU path, never take the process
    // down.
    std::fprintf(stderr, "atb-npu: unknown failure during init\n");
    return false;
  }

  std::fprintf(stderr,
               "atb-npu: %s active, batch=%zu k=%zu n_full=%zu n_npu=%zu "
               "n_gpu=%zu A=%.1f B=%.1f C=%.1f MB\n",
               shape.label, batch, shape.k, shape.n_full, n_slice, n_gpu,
               static_cast<double>(a_bytes) / 1048576.0,
               static_cast<double>(b_bytes) / 1048576.0,
               static_cast<double>(c_bytes) / 1048576.0);
  return true;
}

bool AtbNpuOffload::Launch() {
  if (impl_ == nullptr) {
    return false;
  }
  Impl& p = *impl_;
  try {
    p.run_gate = p.kernel(kErtOpcode, p.bo_instr, p.instr_words, p.bo_a,
                          p.bo_b_gate, p.bo_c_gate);
    p.run_up = p.kernel(kErtOpcode, p.bo_instr, p.instr_words, p.bo_a,
                        p.bo_b_up, p.bo_c_up);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "atb-npu: launch failed: %s\n", e.what());
    return false;
  }
  return true;
}

bool AtbNpuOffload::Wait() {
  if (impl_ == nullptr) {
    return false;
  }
  Impl& p = *impl_;
  int state_gate = 0;
  int state_up = 0;
  try {
    // Gate and up are queued together and run back to back, so the total
    // blocking time is the same; only the polling strategy changes.
    state_gate = WaitSleeping(p.run_gate) ? kErtCompleted : 0;
    state_up = WaitSleeping(p.run_up) ? kErtCompleted : 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "atb-npu: wait failed: %s\n", e.what());
    return false;
  }
  // 4 is ERT_CMD_STATE_COMPLETED.
  if (state_gate != 4 || state_up != 4) {
    std::fprintf(stderr, "atb-npu: unexpected run state gate=%d up=%d\n",
                 state_gate, state_up);
    return false;
  }
  return true;
}

AtbNpuOffload* AtbNpuOffload::Get(AtbRole role) {
  const auto create = [](AtbRole r) -> AtbNpuOffload* {
    auto* candidate = new AtbNpuOffload();
    if (!candidate->Init(r)) {
      delete candidate;
      return nullptr;
    }
    return candidate;
  };
  static AtbNpuOffload* const gate_up = create(AtbRole::kGateUp);
  static AtbNpuOffload* const down = create(AtbRole::kDown);
  return role == AtbRole::kGateUp ? gate_up : down;
}

#else

struct AtbNpuOffload::Impl {};
AtbNpuOffload::AtbNpuOffload() = default;
AtbNpuOffload::~AtbNpuOffload() = default;
bool AtbNpuOffload::Init(AtbRole) {
  return false;
}
void* AtbNpuOffload::activations() const {
  return nullptr;
}
void* AtbNpuOffload::gate_weights() const {
  return nullptr;
}
void* AtbNpuOffload::up_weights() const {
  return nullptr;
}
void* AtbNpuOffload::gate_output() const {
  return nullptr;
}
void* AtbNpuOffload::up_output() const {
  return nullptr;
}
bool AtbNpuOffload::Launch() {
  return false;
}
bool AtbNpuOffload::Wait() {
  return false;
}
AtbNpuOffload* AtbNpuOffload::Get(AtbRole) {
  return nullptr;
}

#endif
}  // namespace gufo::hip
