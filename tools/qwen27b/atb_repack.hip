// Runtime repack of a quantised weight tensor into the ATB bfp16 B operand,
// on the GPU, per layer. This is what removes the need to hold a packed copy of
// the model: the source stays the shipped quantised tensor and the packed tiles
// are produced into a small reusable staging buffer instead.
//
//   atb_repack <w_raw> <type_id> <N> <K> [cpu_packed.bin]
//
// The CPU packer (tools/qwen27b/atb_pack) is the oracle: it is byte-identical
// to the vendored mlir-aie host, so any disagreement here is this kernel's bug.
// With the optional fifth argument the packed bytes are compared and the
// mismatch count is reported; without it, the output goes to stdout.
#define ENGINE_ENABLE_HIP 1
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "src/models/qwen/hip/quant_ops.hpp"

namespace {

using gufo::core::GgmlType;

// config3's L1 tile: 64 reduction rows by 128 output columns.
constexpr unsigned kL1K = 64;
constexpr unsigned kL1N = 128;
constexpr unsigned kGroupsPerTile = (kL1K * kL1N) / 8;  // 1024

// One thread per bfp16 group of 8 packed bytes. A group is 8 consecutive
// reduction rows (k) at one fixed output column (n) -- which is exactly what the
// packer's shuffle makes contiguous, so the write is 9 contiguous bytes and the
// read is 8 consecutive weights of one row.
__global__ void RepackKernel(const void* __restrict__ w, std::size_t row_bytes,
                             int type_id, std::size_t l1_rows,
                             std::size_t groups, int bits,
                             std::uint8_t* __restrict__ out) {
  const std::size_t g = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
  if (g >= groups) {
    return;
  }
  const std::size_t tile = g / kGroupsPerTile;
  const std::size_t gt = g % kGroupsPerTile;
  const std::size_t l1c = tile / l1_rows;
  const std::size_t l1r = tile % l1_rows;
  const unsigned cib = static_cast<unsigned>(gt % 8);
  const unsigned t1 = static_cast<unsigned>(gt / 8);
  const unsigned bis = t1 % 2;
  const unsigned t2 = t1 / 2;
  const unsigned sbr = t2 % 8;
  const unsigned sb = t2 / 8;
  const unsigned sbc = 2 * sb;

  const std::size_t n = l1c * kL1N + (sbc + bis) * 8 + cib;
  const std::size_t k0 = l1r * kL1K + sbr * 8;

  const void* row = static_cast<const std::uint8_t*>(w) + n * row_bytes;
  const GgmlType type = static_cast<GgmlType>(type_id);
  // Decode the 16-element sub-block once. k0 is a multiple of 8, so the 8 values
  // this group needs are either the low or the high half of one sub-block --
  // never split across two -- and decoding per element would repeat this 8x.
  gufo::hip::QuantSub16 sub;
  gufo::hip::DecodeQuantSub16(type, row, k0 / 16, sub);
  const unsigned within = static_cast<unsigned>(k0 % 16);
  float value[8];
  float amax = 0.0F;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    value[i] = sub.scale * static_cast<float>(sub.q[within + i]) - sub.offset;
    amax = fmaxf(amax, fabsf(value[i]));
  }
  std::uint8_t* dst = out + g * 9;
  if (amax == 0.0F) {
    for (int i = 0; i < 9; ++i) {
      dst[i] = 0;
    }
    return;
  }
  int exponent = 0;
  (void)frexpf(amax, &exponent);
  // Matches the host encoder: the stored exponent byte is the biased exponent
  // of the block maximum, and the grid step is 2^(exponent - bits).
  dst[0] = static_cast<std::uint8_t>(exponent + 126);
  const float step = ldexpf(1.0F, exponent - bits);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    // Round half up, as the host does with its add-half-then-shift.
    const int magnitude = static_cast<int>(floorf(value[i] / step + 0.5F));
    dst[1 + i] = static_cast<std::uint8_t>(static_cast<std::int8_t>(magnitude));
  }
}

std::size_t RowBytes(GgmlType type, std::size_t k) {
  switch (type) {
    case GgmlType::kQ3_K:
      return (k / 256) * 110;
    case GgmlType::kQ4_K:
      return (k / 256) * 144;
    case GgmlType::kQ5_K:
      return (k / 256) * 176;
    case GgmlType::kQ6_K:
      return (k / 256) * 210;
    case GgmlType::kIQ2_XXS:
      return (k / 256) * 66;
    case GgmlType::kIQ2_XS:
      return (k / 256) * 74;
    case GgmlType::kIQ2_S:
      return (k / 256) * 82;
    case GgmlType::kIQ3_XXS:
      return (k / 256) * 98;
    case GgmlType::kIQ3_S:
      return (k / 256) * 110;
    case GgmlType::kIQ4_XS:
      return (k / 256) * 136;
    default:
      return 0;
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr, "usage: %s <w_raw> <type_id> <N> <K> [cpu_packed.bin]\n",
                 argv[0]);
    return 2;
  }
  const char* raw_path = argv[1];
  const int type_id = std::atoi(argv[2]);
  const std::size_t n_out = std::strtoull(argv[3], nullptr, 10);
  const std::size_t k = std::strtoull(argv[4], nullptr, 10);
  const GgmlType type = static_cast<GgmlType>(type_id);
  const std::size_t row_bytes = RowBytes(type, k);
  if (row_bytes == 0 || k % kL1K != 0 || n_out % kL1N != 0) {
    std::fprintf(stderr, "unsupported type id %d or geometry\n", type_id);
    return 1;
  }
  const std::size_t groups = (n_out * k) / 8;
  const std::size_t packed_bytes = groups * 9;

  std::vector<std::uint8_t> raw(row_bytes * n_out);
  std::FILE* fp = std::fopen(raw_path, "rb");
  if (fp == nullptr || std::fread(raw.data(), 1, raw.size(), fp) != raw.size()) {
    std::fprintf(stderr, "cannot read %s\n", raw_path);
    return 1;
  }
  std::fclose(fp);

  void* d_w = nullptr;
  std::uint8_t* d_out = nullptr;
  if (hipMalloc(&d_w, raw.size()) != hipSuccess) return 1;
  if (hipMalloc(&d_out, packed_bytes) != hipSuccess) return 1;
  if (hipMemcpy(d_w, raw.data(), raw.size(), hipMemcpyHostToDevice) != hipSuccess) {
    return 1;
  }
  const unsigned blocks = static_cast<unsigned>((groups + 255) / 256);
  RepackKernel<<<blocks, 256>>>(d_w, row_bytes, type_id, k / kL1K, groups, 7, d_out);
  if (hipDeviceSynchronize() != hipSuccess) return 1;

  std::vector<std::uint8_t> packed(packed_bytes);
  if (hipMemcpy(packed.data(), d_out, packed_bytes, hipMemcpyDeviceToHost) !=
      hipSuccess) {
    return 1;
  }

  // Time it: enough repetitions to cover the launch, at the real geometry.
  const int reps = argc > 5 ? 20 : 20;
  const auto start = std::chrono::high_resolution_clock::now();
  for (int r = 0; r < reps; ++r) {
    RepackKernel<<<blocks, 256>>>(d_w, row_bytes, type_id, k / kL1K, groups, 7,
                                  d_out);
  }
  if (hipDeviceSynchronize() != hipSuccess) return 1;
  const auto stop = std::chrono::high_resolution_clock::now();
  const double ms =
      std::chrono::duration<double, std::milli>(stop - start).count() / reps;
  std::fprintf(stderr,
               "repack type=%d N=%zu K=%zu: %.2f ms/layer, %.1f GB/s written, "
               "%.1f G weights/s\n",
               type_id, n_out, k, ms,
               static_cast<double>(packed_bytes) / ms / 1e6,
               static_cast<double>(n_out * k) / ms / 1e6);

  if (argc > 5) {
    std::FILE* cf = std::fopen(argv[5], "rb");
    if (cf == nullptr) {
      std::fprintf(stderr, "cannot read %s\n", argv[5]);
      return 1;
    }
    std::vector<std::uint8_t> cpu(packed_bytes);
    const std::size_t got = std::fread(cpu.data(), 1, cpu.size(), cf);
    std::fclose(cf);
    if (got != cpu.size()) {
      std::fprintf(stderr, "oracle size mismatch: %zu vs %zu\n", got, cpu.size());
      return 1;
    }
    std::size_t diff = 0, worst = 0;
    for (std::size_t i = 0; i < packed.size(); ++i) {
      if (packed[i] != cpu[i]) {
        ++diff;
        const int d = std::abs(static_cast<int>(packed[i]) - static_cast<int>(cpu[i]));
        if (static_cast<std::size_t>(d) > worst) worst = static_cast<std::size_t>(d);
      }
    }
    std::printf("type=%d N=%zu K=%zu  bytes=%zu  mismatches=%zu  max|delta|=%zu\n",
                type_id, n_out, k, packed.size(), diff, worst);
  } else {
    std::fwrite(packed.data(), 1, packed.size(), stdout);
  }
  hipFree(d_w);
  hipFree(d_out);
  return 0;
}
