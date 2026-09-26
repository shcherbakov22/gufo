// Checks the engine's ATB slice transforms against their full-tensor forms and
// against a CPU re-implementation of the A encoder.
//
// The NPU operand layouts were each verified against the vendored host code in
// tools/qwen27b (atb_pack.cpp, atb_act_bench.cpp, atb_c_oracle.cpp). The split
// adds offsets and a fused SwiGLU decode on top, so what has to be checked is
// the offset arithmetic, not the layout again: a slice pack must equal the
// corresponding byte range of the full pack, and a slice decode must equal the
// corresponding columns of the full decode.
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen/hip/kernels/prefill_fp16.hip"
#include "src/models/qwen/hip/ops/prefill_fp16.hpp"

namespace {

using gufo::core::GgmlType;
using gufo::hip::LaunchAtbDecodeCFp32;
using gufo::hip::LaunchAtbDecodeSwiGLUFp16;
using gufo::hip::LaunchAtbEncodeAFp16;
using gufo::hip::LaunchAtbRepackBfp16;
using gufo::hip::LaunchAtbRepackBfp16Slice;

int Fail(const char* what) {
  std::fprintf(stderr, "FAIL %s\n", what);
  return 1;
}

// A Q4_K block is 144 bytes holding 256 weights: an FP16 d and dmin, twelve
// packed 6-bit scale bytes, and 128 quant bytes. Writing d = 1, dmin = 0 and 1
// everywhere else keeps every decoded value finite and small, which is all this
// test needs: both paths decode the same bytes, so the comparison is about
// indexing.
void FillWeights(std::vector<std::uint8_t>* w) {
  std::fill(w->begin(), w->end(), 1);
  for (std::size_t base = 0; base + 144 <= w->size(); base += 144) {
    (*w)[base + 0] = 0x00;
    (*w)[base + 1] = 0x3C;  // f16 1.0
    (*w)[base + 2] = 0x00;
    (*w)[base + 3] = 0x00;  // f16 0.0
  }
}

std::vector<std::uint8_t> EncodeAOnCpu(const std::vector<__half>& act,
                                       std::size_t rows, std::size_t k) {
  constexpr unsigned L1M = 128, L1K = 64;
  constexpr unsigned per_tile = (L1M * L1K) / 8;
  const std::size_t groups = rows * k / 8;
  const std::size_t l1_cols = k / L1K;
  std::vector<std::uint8_t> out(groups * 9);
  for (std::size_t g = 0; g < groups; ++g) {
    const std::size_t tile = g / per_tile;
    const std::size_t gt = g % per_tile;
    const std::size_t l1r = tile / l1_cols;
    const std::size_t l1c = tile % l1_cols;
    const unsigned rib = static_cast<unsigned>(gt % 8);
    const unsigned t1 = static_cast<unsigned>(gt / 8);
    const unsigned bis = t1 % 2;
    const unsigned t2 = t1 / 2;
    const unsigned sbc = t2 % 8;
    const unsigned si = t2 / 8;
    const unsigned r = (2 * si + bis) * 8 + rib;
    const std::size_t row = l1r * L1M + r;
    const std::size_t col = l1c * L1K + sbc * 8;
    float value[8];
    float amax = 0.0F;
    for (int i = 0; i < 8; ++i) {
      const float v = static_cast<float>(act[row * k + col + i]);
      value[i] = v;
      amax = std::fmax(amax, std::fabs(v));
    }
    std::uint8_t* dst = &out[g * 9];
    if (amax == 0.0F) continue;
    int exponent = 0;
    (void)std::frexp(amax, &exponent);
    dst[0] = static_cast<std::uint8_t>(exponent + 126);
    const float step = std::ldexp(1.0F, exponent - 7);
    for (int i = 0; i < 8; ++i) {
      const int magnitude = static_cast<int>(std::floor(value[i] / step + 0.5F));
      dst[1 + i] = static_cast<std::uint8_t>(static_cast<std::int8_t>(magnitude));
    }
  }
  return out;
}

}  // namespace

int main() {
  constexpr std::size_t kK = 5120;
  constexpr std::size_t kRows = 512;
  constexpr std::size_t kN = 1024;
  constexpr std::size_t kNTiles = 4;      // 512 columns packed as a slice
  constexpr std::size_t kNOffset = 512;   // starting at column 512

  // ---- B: a slice pack must be the matching byte range of the full pack.
  const std::size_t row_bytes = gufo::quant::QuantizedRowBytes(
      GgmlType::kQ4_K, kK);
  if (row_bytes == 0) return Fail("QuantizedRowBytes");
  std::vector<std::uint8_t> weights(kN * row_bytes);
  FillWeights(&weights);
  const std::size_t full_bytes = kN * kK * 9 / 8;
  const std::size_t slice_bytes = kNTiles * 128 * kK * 9 / 8;

  std::uint8_t* d_w = nullptr;
  std::uint8_t* d_full = nullptr;
  std::uint8_t* d_slice = nullptr;
  if (hipMalloc(&d_w, weights.size()) != hipSuccess) return Fail("hipMalloc w");
  if (hipMalloc(&d_full, full_bytes) != hipSuccess) return Fail("hipMalloc full");
  if (hipMalloc(&d_slice, slice_bytes) != hipSuccess) return Fail("hipMalloc slice");
  if (hipMemcpy(d_w, weights.data(), weights.size(), hipMemcpyHostToDevice) !=
      hipSuccess)
    return Fail("hipMemcpy w");
  LaunchAtbRepackBfp16(d_w, GgmlType::kQ4_K, kN, kK, d_full, nullptr);
  LaunchAtbRepackBfp16Slice(d_w, GgmlType::kQ4_K, kN, kK, kNOffset, kNTiles,
                            d_slice, nullptr);
  if (hipDeviceSynchronize() != hipSuccess) return Fail("repack sync");
  std::vector<std::uint8_t> full(full_bytes), slice(slice_bytes);
  if (hipMemcpy(full.data(), d_full, full_bytes, hipMemcpyDeviceToHost) != hipSuccess)
    return Fail("read full");
  if (hipMemcpy(slice.data(), d_slice, slice_bytes, hipMemcpyDeviceToHost) !=
      hipSuccess)
    return Fail("read slice");

  // Tiles run (n outer, k inner), so an aligned range of n tiles is contiguous.
  const std::size_t l1_rows = kK / 64;
  const std::size_t base = (kNOffset / 128) * l1_rows * 1024 * 9;
  std::size_t b_diff = 0;
  std::size_t b_nonzero = 0;
  if (base + slice_bytes > full_bytes) return Fail("B base out of range");
  for (std::size_t i = 0; i < slice_bytes; ++i) {
    if (slice[i] != full[base + i]) ++b_diff;
    if (slice[i] != 0) ++b_nonzero;
  }
  std::printf("B slice: bytes=%zu nonzero=%zu mismatches_vs_full=%zu\n",
              slice_bytes, b_nonzero, b_diff);
  if (b_diff != 0) return Fail("B slice");
  if (b_nonzero == 0) return Fail("B slice all zero");

  // ---- A: the engine encoder against a CPU copy of the same transform.
  const std::size_t a_bytes = kRows * kK * 9 / 8;
  std::vector<__half> act(kRows * kK);
  for (std::size_t i = 0; i < act.size(); ++i) {
    const float v = static_cast<float>((static_cast<int>(i % 97) - 48)) * 0.017F;
    act[i] = __float2half(v);
  }
  __half* d_act = nullptr;
  std::uint8_t* d_a = nullptr;
  if (hipMalloc(&d_act, a_bytes / 9 * 8 * 2) != hipSuccess) return Fail("hipMalloc act");
  if (hipMalloc(&d_a, a_bytes) != hipSuccess) return Fail("hipMalloc a");
  if (hipMemcpy(d_act, act.data(), act.size() * sizeof(__half),
                hipMemcpyHostToDevice) != hipSuccess)
    return Fail("hipMemcpy act");
  LaunchAtbEncodeAFp16(d_act, kRows, kK, d_a, nullptr);
  if (hipDeviceSynchronize() != hipSuccess) return Fail("encode sync");
  std::vector<std::uint8_t> a_gpu(a_bytes);
  if (hipMemcpy(a_gpu.data(), d_a, a_bytes, hipMemcpyDeviceToHost) != hipSuccess)
    return Fail("read a");
  const std::vector<std::uint8_t> a_cpu = EncodeAOnCpu(act, kRows, kK);
  std::size_t a_diff = 0;
  std::size_t a_exp_diff = 0;
  std::size_t a_nonzero = 0;
  for (std::size_t i = 0; i < a_bytes; ++i) {
    if (a_gpu[i] != a_cpu[i]) {
      ++a_diff;
      if (i % 9 == 0) ++a_exp_diff;
    }
    if (a_gpu[i] != 0) ++a_nonzero;
  }
  std::printf("A encode: bytes=%zu nonzero=%zu mismatches_vs_cpu=%zu exponent=%zu\n",
              a_bytes, a_nonzero, a_diff, a_exp_diff);
  if (a_diff != 0) return Fail("A encode");
  if (a_nonzero == 0) return Fail("A encode all zero");

  // ---- C: a slice decode must equal the matching columns of the full decode.
  constexpr std::size_t cRows = 512;
  constexpr std::size_t cFull = 1024;
  constexpr std::size_t cSlice = 256;
  constexpr std::size_t cOffset = 384;
  const std::size_t c_full_bytes = cRows * cFull * 9 / 8;
  const std::size_t c_slice_bytes = cRows * cSlice * 9 / 8;
  std::vector<std::uint8_t> packed(c_full_bytes);
  for (std::size_t i = 0; i < packed.size(); ++i) {
    packed[i] = static_cast<std::uint8_t>((i * 37 + 11) % 251);
  }
  // Slice bytes must be the slice's own tile range, so extract them from the
  // full image the same way the packer orders tiles.
  std::vector<std::uint8_t> packed_slice(c_slice_bytes);
  {
    const std::size_t groups_full = cRows * cFull / 8;
    const std::size_t groups_slice = cRows * cSlice / 8;
    const std::size_t per_tile = 512 * 128 / 8;
    const std::size_t l1_cols_full = cFull / 128;
    for (std::size_t g = 0; g < groups_slice; ++g) {
      const std::size_t tile = g / per_tile;
      const std::size_t gi = g % per_tile;
      const std::size_t l1r = tile / (cSlice / 128);
      const std::size_t l1c = tile % (cSlice / 128);
      const std::size_t src = (l1r * l1_cols_full + (l1c + cOffset / 128)) * per_tile + gi;
      std::memcpy(&packed_slice[g * 9], &packed[src * 9], 9);
    }
    (void)groups_full;
  }
  std::uint8_t* d_packed = nullptr;
  std::uint8_t* d_packed_slice = nullptr;
  float* d_full_c = nullptr;
  float* d_slice_c = nullptr;
  if (hipMalloc(&d_packed, c_full_bytes) != hipSuccess) return Fail("hipMalloc pc");
  if (hipMalloc(&d_packed_slice, c_slice_bytes) != hipSuccess) return Fail("hipMalloc ps");
  if (hipMalloc(&d_full_c, cRows * cFull * sizeof(float)) != hipSuccess)
    return Fail("hipMalloc fc");
  if (hipMalloc(&d_slice_c, cRows * cFull * sizeof(float)) != hipSuccess)
    return Fail("hipMalloc sc");
  if (hipMemcpy(d_packed, packed.data(), c_full_bytes, hipMemcpyHostToDevice) != hipSuccess)
    return Fail("copy pc");
  if (hipMemcpy(d_packed_slice, packed_slice.data(), c_slice_bytes,
                hipMemcpyHostToDevice) != hipSuccess)
    return Fail("copy ps");
  LaunchAtbDecodeCFp32(d_packed, cRows, cFull, cFull, 0, d_full_c, nullptr);
  LaunchAtbDecodeCFp32(d_packed_slice, cRows, cSlice, cFull, cOffset, d_slice_c,
                       nullptr);
  if (hipDeviceSynchronize() != hipSuccess) return Fail("decode sync");
  std::vector<float> full_c(cRows * cFull), slice_c(cRows * cFull);
  if (hipMemcpy(full_c.data(), d_full_c, full_c.size() * sizeof(float),
                hipMemcpyDeviceToHost) != hipSuccess)
    return Fail("read fc");
  if (hipMemcpy(slice_c.data(), d_slice_c, slice_c.size() * sizeof(float),
                hipMemcpyDeviceToHost) != hipSuccess)
    return Fail("read sc");
  std::size_t c_diff = 0;
  std::size_t c_checked = 0;
  std::size_t c_nonzero = 0;
  for (std::size_t r = 0; r < cRows; ++r) {
    for (std::size_t c = 0; c < cSlice; ++c) {
      const float want = full_c[r * cFull + cOffset + c];
      const float got = slice_c[r * cFull + cOffset + c];
      ++c_checked;
      if (want != got) ++c_diff;
      if (got != 0.0F) ++c_nonzero;
    }
  }
  std::printf("C slice decode: checked=%zu nonzero=%zu mismatches=%zu\n", c_checked,
              c_nonzero, c_diff);
  if (c_diff != 0) return Fail("C slice decode");
  if (c_nonzero == 0) return Fail("C slice all zero");

  // ---- Fused SwiGLU decode against the FP32 decode plus an explicit silu.
  float* d_up_c = nullptr;
  __half* d_swiglu = nullptr;
  if (hipMalloc(&d_up_c, cRows * cFull * sizeof(float)) != hipSuccess)
    return Fail("hipMalloc uc");
  if (hipMalloc(&d_swiglu, cRows * cFull * sizeof(__half)) != hipSuccess)
    return Fail("hipMalloc sw");
  // A second packed image standing in for the up projection.
  std::vector<std::uint8_t> packed2(packed.size());
  for (std::size_t i = 0; i < packed2.size(); ++i) {
    packed2[i] = static_cast<std::uint8_t>((i * 53 + 7) % 251);
  }
  std::uint8_t* d_packed2 = nullptr;
  if (hipMalloc(&d_packed2, c_full_bytes) != hipSuccess) return Fail("hipMalloc p2");
  if (hipMemcpy(d_packed2, packed2.data(), c_full_bytes, hipMemcpyHostToDevice) !=
      hipSuccess)
    return Fail("copy p2");
  std::vector<std::uint8_t> packed2_slice(c_slice_bytes);
  {
    const std::size_t per_tile = 512 * 128 / 8;
    const std::size_t l1_cols_full = cFull / 128;
    for (std::size_t g = 0; g < cRows * cSlice / 8; ++g) {
      const std::size_t tile = g / per_tile;
      const std::size_t gi = g % per_tile;
      const std::size_t l1r = tile / (cSlice / 128);
      const std::size_t l1c = tile % (cSlice / 128);
      const std::size_t src =
          (l1r * l1_cols_full + (l1c + cOffset / 128)) * per_tile + gi;
      std::memcpy(&packed2_slice[g * 9], &packed2[src * 9], 9);
    }
  }
  if (hipMemcpy(d_packed2, packed2.data(), c_full_bytes, hipMemcpyHostToDevice) !=
      hipSuccess)
    return Fail("copy p2b");
  // Slice form of the up image must go through its own buffer to exercise the
  // same path the engine uses.
  std::uint8_t* d_packed2_slice = nullptr;
  if (hipMalloc(&d_packed2_slice, c_slice_bytes) != hipSuccess) return Fail("hipMalloc p2s");
  if (hipMemcpy(d_packed2_slice, packed2_slice.data(), c_slice_bytes,
                hipMemcpyHostToDevice) != hipSuccess)
    return Fail("copy p2s");
  LaunchAtbDecodeCFp32(d_packed2, cRows, cFull, cFull, 0, d_up_c, nullptr);
  LaunchAtbDecodeSwiGLUFp16(d_packed_slice, d_packed2_slice, cRows, cSlice, cFull,
                            cOffset, d_swiglu, nullptr);
  if (hipDeviceSynchronize() != hipSuccess) return Fail("swiglu sync");
  std::vector<float> up_c(cRows * cFull);
  std::vector<__half> swiglu(cRows * cFull);
  if (hipMemcpy(up_c.data(), d_up_c, up_c.size() * sizeof(float),
                hipMemcpyDeviceToHost) != hipSuccess)
    return Fail("read uc");
  if (hipMemcpy(swiglu.data(), d_swiglu, swiglu.size() * sizeof(__half),
                hipMemcpyDeviceToHost) != hipSuccess)
    return Fail("read sw");
  std::size_t s_diff = 0;
  for (std::size_t r = 0; r < cRows; ++r) {
    for (std::size_t c = 0; c < cSlice; ++c) {
      const std::size_t idx = r * cFull + cOffset + c;
      const float g = full_c[idx];
      const float u = up_c[idx];
      // The kernel writes FP16, so the reference has to be rounded the same
      // way before comparing; comparing a half against an unrounded float
      // measures the format, not the kernel.
      const float want = __half2float(__float2half_rn((g / (1.0F + std::exp(-g))) * u));
      const float got = __half2float(swiglu[idx]);
      if (got != want) ++s_diff;
    }
  }
  std::printf("fused SwiGLU: elements=%zu mismatches=%zu\n", cRows * cSlice, s_diff);
  if (s_diff != 0) return Fail("fused SwiGLU");

  std::printf("ATB_SLICE_CHECK OK\n");
  return 0;
}
