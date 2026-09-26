// GPU decoder from the ATB C layout back to the engine's FP32 FFN rows.
//
//   atb_decode_c <packed_c.bin> <M> <N> [reference_fp32.bin]
//
// The device writes C as bfp16ebs8 in L1 tiles of (512, 128) laid out row-major,
// each tile 2x2 super-blocks of 8x8 row-major sub-blocks. Consecutive 8 packed
// elements are therefore 8 consecutive output columns of one row, so one thread
// per 8-element group decodes 9 bytes and writes 8 floats.
//
// The oracle is the vendored host's own path: bfp16ebs8ToFloat followed by
// layout_inverse_C_L1_2x2_8x8block, in tools/qwen27b/atb_c_oracle.cpp.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hip/hip_runtime.h>
#include <vector>

namespace {

constexpr unsigned kL1M = 512;
constexpr unsigned kL1N = 128;
constexpr unsigned kGroupsPerTile = (kL1M * kL1N) / 8;  // 8192

__device__ __forceinline__ float Bfp16DecodeByte(std::uint8_t byte,
                                                 float multiplier) {
  const bool negative = (byte & 0x80u) != 0;
  float magnitude;
  if (negative) {
    magnitude = static_cast<float>(static_cast<std::uint8_t>(~(byte - 1)));
    return -(magnitude * multiplier);
  }
  magnitude = static_cast<float>(byte);
  return magnitude * multiplier;
}

__global__ void AtbDecodeCKernel(const std::uint8_t* __restrict__ packed,
                                 int n, std::size_t l1_cols, std::size_t groups,
                                 float* __restrict__ out) {
  const std::size_t g =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (g >= groups) {
    return;
  }
  const std::size_t tile = g / kGroupsPerTile;
  const std::size_t gi = g % kGroupsPerTile;
  const std::size_t l1r = tile / l1_cols;
  const std::size_t l1c = tile % l1_cols;

  const unsigned rib = static_cast<unsigned>(gi % 8);
  const unsigned t1 = static_cast<unsigned>(gi / 8);
  const unsigned block = t1 % 4;
  const unsigned t2 = t1 / 4;
  const unsigned sbc_half = t2 % 8;
  const unsigned sbr_half = t2 / 8;
  const unsigned br = block / 2;
  const unsigned bc = block % 2;

  const std::size_t row = l1r * kL1M + (2 * sbr_half + br) * 8 + rib;
  const std::size_t col0 = l1c * kL1N + (2 * sbc_half + bc) * 8;

  const std::uint8_t* src = packed + g * 9;
  const std::uint8_t shared = src[0];
  float multiplier;
  if (shared >= 127) {
    multiplier = static_cast<float>(1u << (shared - 127));
  } else {
    multiplier = 1.0F / static_cast<float>(1u << (127 - shared));
  }
  multiplier /= 64.0F;
  float* dst = out + row * static_cast<std::size_t>(n) + col0;
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    dst[k] = Bfp16DecodeByte(src[1 + k], multiplier);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s <packed_c.bin> <M> <N> [ref_fp32.bin]\n",
                 argv[0]);
    return 2;
  }
  const std::size_t M = std::strtoull(argv[2], nullptr, 10);
  const std::size_t N = std::strtoull(argv[3], nullptr, 10);
  if (M % kL1M != 0 || N % kL1N != 0) {
    std::fprintf(stderr, "M must be a multiple of %u and N of %u\n", kL1M, kL1N);
    return 2;
  }
  const std::size_t count = M * N;
  const std::size_t groups = count / 8;
  const std::size_t packed_bytes = groups * 9;
  const std::size_t l1_cols = N / kL1N;

  std::vector<std::uint8_t> host(packed_bytes);
  {
    std::FILE* fp = std::fopen(argv[1], "rb");
    if (fp == nullptr || std::fread(host.data(), 1, packed_bytes, fp) != packed_bytes) {
      std::fprintf(stderr, "cannot read %s\n", argv[1]);
      return 1;
    }
    std::fclose(fp);
  }

  std::uint8_t* d_packed = nullptr;
  float* d_out = nullptr;
  if (hipMalloc(&d_packed, packed_bytes) != hipSuccess) return 1;
  if (hipMalloc(&d_out, count * sizeof(float)) != hipSuccess) return 1;
  if (hipMemcpy(d_packed, host.data(), packed_bytes, hipMemcpyHostToDevice) !=
      hipSuccess) {
    return 1;
  }
  AtbDecodeCKernel<<<static_cast<unsigned>((groups + 255) / 256), 256>>>(
      d_packed, static_cast<int>(N), l1_cols, groups, d_out);
  if (hipDeviceSynchronize() != hipSuccess) return 1;
  std::vector<float> out(count);
  if (hipMemcpy(out.data(), d_out, count * sizeof(float),
                hipMemcpyDeviceToHost) != hipSuccess) {
    return 1;
  }

  if (argc > 4) {
    std::FILE* rf = std::fopen(argv[4], "rb");
    if (rf == nullptr) {
      std::fprintf(stderr, "cannot read %s\n", argv[4]);
      return 1;
    }
    std::vector<float> ref(count);
    const std::size_t got = std::fread(ref.data(), sizeof(float), count, rf);
    std::fclose(rf);
    if (got != count) {
      std::fprintf(stderr, "oracle size mismatch %zu vs %zu\n", got, count);
      return 1;
    }
    std::size_t diff = 0;
    double worst = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
      if (out[i] != ref[i]) {
        ++diff;
        const double d = (double)out[i] - (double)ref[i];
        if (d > worst) worst = d;
        if (-d > worst) worst = -d;
      }
    }
    std::printf("C decode %zux%zu: elements=%zu exact_mismatches=%zu max_abs=%.6g\n",
                M, N, count, diff, worst);
  } else {
    std::fwrite(out.data(), sizeof(float), count, stdout);
  }
  hipFree(d_packed);
  hipFree(d_out);
  return 0;
}
