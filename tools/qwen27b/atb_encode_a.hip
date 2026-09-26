// GPU encoder for the ATB A operand: an FP16 activation [rows, K] into the
// bfp16 A L1 layout the NPU design consumes.
//
//   atb_encode_a <act_fp16> <M> <K> [m_tile] [k_tile] [cpu_packed.bin]
//
// The layout is layout_A_L1_2x1_8x8block from the vendored header: L1 tiles of
// (m_tile, k_tile) emitted row-major, each shuffled into 2x1 vertically-stacked
// super-blocks of 8x8 row-major sub-blocks, then bfp16 with one shared exponent
// per 8 consecutive elements. Groups of 8 are consecutive k within one row, so a
// group never straddles a row and the shared exponent is the same set the NPU
// sees. tools/qwen27b/atb_act_bench.cpp is the CPU oracle for this.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <vector>

namespace {

constexpr unsigned kL1M = 128;
constexpr unsigned kL1K = 64;
constexpr unsigned kGroupsPerTile = (kL1M * kL1K) / 8;  // 1024

__global__ void AtbEncodeAKernel(const __half* __restrict__ a, int k,
                                 std::size_t l1_cols, std::size_t groups,
                                 std::uint8_t* __restrict__ out) {
  const std::size_t g =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (g >= groups) {
    return;
  }
  const std::size_t tile = g / kGroupsPerTile;
  const std::size_t gt = g % kGroupsPerTile;
  const std::size_t l1r = tile / l1_cols;
  const std::size_t l1c = tile % l1_cols;
  const unsigned rib = static_cast<unsigned>(gt % 8);
  const unsigned t1 = static_cast<unsigned>(gt / 8);
  const unsigned bis = t1 % 2;
  const unsigned t2 = t1 / 2;
  const unsigned sbc = t2 % 8;
  const unsigned si = t2 / 8;

  const unsigned r = (2 * si + bis) * 8 + rib;
  const std::size_t row = l1r * kL1M + r;
  const __half* src = a + row * static_cast<std::size_t>(k) + l1c * kL1K + sbc * 8;

  float value[8];
  float amax = 0.0F;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    value[i] = __half2float(src[i]);
    amax = fmaxf(amax, fabsf(value[i]));
  }
  std::uint8_t* dst = out + g * 9;
  if (amax == 0.0F) {
    for (int i = 0; i < 9; ++i) dst[i] = 0;
    return;
  }
  int exponent = 0;
  (void)frexpf(amax, &exponent);
  dst[0] = static_cast<std::uint8_t>(exponent + 126);
  const float step = ldexpf(1.0F, exponent - 7);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int magnitude = static_cast<int>(floorf(value[i] / step + 0.5F));
    dst[1 + i] = static_cast<std::uint8_t>(static_cast<std::int8_t>(magnitude));
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr,
                 "usage: %s <act_fp16> <M> <K> [m_tile] [k_tile] [cpu_packed.bin]\n",
                 argv[0]);
    return 2;
  }
  const std::size_t rows = std::strtoull(argv[2], nullptr, 10);
  const std::size_t k = std::strtoull(argv[3], nullptr, 10);
  const int m_tile = argc > 4 ? std::atoi(argv[4]) : 128;
  const int k_tile = argc > 5 ? std::atoi(argv[5]) : 64;
  if (m_tile != static_cast<int>(kL1M) || k_tile != static_cast<int>(kL1K)) {
    std::fprintf(stderr, "only the %ux%u tile is implemented\n", kL1M, kL1K);
    return 2;
  }
  const std::size_t count = rows * k;
  const std::size_t groups = count / 8;
  const std::size_t packed_bytes = groups * 9;
  const std::size_t l1_cols = k / kL1K;

  std::vector<__half> host(count);
  {
    std::FILE* fp = std::fopen(argv[1], "rb");
    if (fp == nullptr ||
        std::fread(host.data(), sizeof(__half), count, fp) != count) {
      std::fprintf(stderr, "cannot read %s\n", argv[1]);
      return 1;
    }
    std::fclose(fp);
  }

  __half* d_a = nullptr;
  std::uint8_t* d_out = nullptr;
  if (hipMalloc(&d_a, count * sizeof(__half)) != hipSuccess) return 1;
  if (hipMalloc(&d_out, packed_bytes) != hipSuccess) return 1;
  if (hipMemcpy(d_a, host.data(), count * sizeof(__half),
                hipMemcpyHostToDevice) != hipSuccess) {
    return 1;
  }
  AtbEncodeAKernel<<<static_cast<unsigned>((groups + 255) / 256), 256>>>(
      d_a, static_cast<int>(k), l1_cols, groups, d_out);
  if (hipDeviceSynchronize() != hipSuccess) return 1;
  std::vector<std::uint8_t> packed(packed_bytes);
  if (hipMemcpy(packed.data(), d_out, packed_bytes, hipMemcpyDeviceToHost) !=
      hipSuccess) {
    return 1;
  }

  if (argc > 6) {
    std::FILE* cf = std::fopen(argv[6], "rb");
    if (cf == nullptr) {
      std::fprintf(stderr, "cannot read %s\n", argv[6]);
      return 1;
    }
    std::vector<std::uint8_t> cpu(packed_bytes);
    const std::size_t got = std::fread(cpu.data(), 1, cpu.size(), cf);
    std::fclose(cf);
    if (got != cpu.size()) {
      std::fprintf(stderr, "oracle size mismatch %zu vs %zu\n", got, cpu.size());
      return 1;
    }
    std::size_t diff = 0;
    std::size_t exp_diff = 0;
    for (std::size_t i = 0; i < packed.size(); ++i) {
      if (packed[i] != cpu[i]) {
        ++diff;
        if (i % 9 == 0) ++exp_diff;
      }
    }
    std::printf(
        "A encode %zu x %zu: bytes=%zu mismatches=%zu exponent_mismatches=%zu\n",
        rows, k, packed.size(), diff, exp_diff);
  } else {
    std::fwrite(packed.data(), 1, packed.size(), stdout);
  }
  hipFree(d_a);
  hipFree(d_out);
  return 0;
}
