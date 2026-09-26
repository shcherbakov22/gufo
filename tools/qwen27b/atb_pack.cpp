#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "tools/qwen27b/atb_quant_types.hpp"

// Offline repack of a quantised FFN weight tensor into the ATB bfp16 B operand.
//
//   atb_pack <w_raw> <type_id> <W_rows> <W_cols> [k_tile] [n_tile] > out.bin
//
// W_raw is W as raw quantisation blocks, row-major [W_rows, W_cols].  The GEMM
// consumes B as row-major [K, N] with B[k][n] = W[n][k], tiled into (k_tile,
// n_tile) L1 tiles emitted column-major, each tile shuffled into 1x2
// super-blocks of 8x8 column-major sub-blocks, then encoded to bfp16 with a
// shared exponent per 8 elements.  The encoder rounds to nearest rather than
// truncating, which is a free choice: the stored value is an int8 magnitude
// times a shared exponent, so any encoding the device can read is legal.
//
// The output is byte-identical to what the mlir-aie ATB host produces at
// runtime from the same floats; that equality is the validation.

namespace {

// Shuffle a (rows x cols) row-major float matrix into 1x2 row-major
// super-blocks of 8x8 column-major sub-blocks. Port of
// gemm_atb::layout_transpose_1x2_8x8block.
void Shuffle1x2(const float* in, int rows, int cols, int row0, int col0, int ld,
                std::vector<float>& out, std::size_t& oi) {
  const int block_rows = rows / 8;
  const int block_cols = cols / 8;
  for (int sbc = 0; sbc < block_cols; sbc += 2) {
    for (int sbr = 0; sbr < block_rows; sbr++) {
      const int nbis = std::min(2, block_cols - sbc);
      for (int bis = 0; bis < nbis; bis++) {
        const int cbr = sbr;
        const int cbc = sbc + bis;
        for (int cib = 0; cib < 8; cib++) {
          for (int rib = 0; rib < 8; rib++) {
            const int r = row0 + cbr * 8 + rib;
            const int c = col0 + cbc * 8 + cib;
            out[oi++] = in[(std::size_t)r * ld + c];
          }
        }
      }
    }
  }
}

// Port of gemm_atb::layout_transpose_L1_1x2_8x8block: column-major L1 tiles.
std::vector<float> ShuffleB(const std::vector<float>& in, int rows, int cols,
                            int l1_k, int l1_n) {
  std::vector<float> out((std::size_t)rows * cols);
  const int l1_rows = rows / l1_k;
  const int l1_cols = cols / l1_n;
  std::size_t oi = 0;
  for (int l1c = 0; l1c < l1_cols; l1c++)
    for (int l1r = 0; l1r < l1_rows; l1r++)
      Shuffle1x2(in.data(), l1_k, l1_n, l1r * l1_k, l1c * l1_n, cols, out, oi);
  return out;
}

// Port of floatToBfp16 from block_datatypes/helper.h with the round-to-nearest
// branch: 7 magnitude bits, one shared exponent per 8 elements.
std::vector<std::uint8_t> Bfp16Encode(const std::vector<float>& a,
                                      int block = 8) {
  const std::size_t size = a.size();
  std::vector<std::uint8_t> res(size + size / 8);
  const int mbits = 7;
  std::size_t start = 0, ci = 1;
  for (;;) {
    const std::size_t end = std::min(start + (std::size_t)block, size);
    std::uint32_t max_exp = 0;
    for (std::size_t i = start; i < end; i++) {
      std::uint32_t x;
      std::memcpy(&x, &a[i], 4);
      max_exp = std::max(max_exp, (x >> 23) & 0xFFu);
    }
    for (std::size_t i = start; i < end; i++) {
      std::uint32_t x;
      std::memcpy(&x, &a[i], 4);
      const std::uint32_t sign = x & 0x80000000u;
      const std::uint32_t exp = (x >> 23) & 0xFFu;
      std::uint32_t mantissa = x & 0x007FFFFFu;
      if (exp)
        mantissa |= 0x00800000u;
      if (exp >= 255)
        continue;  // inf/NaN: the reference leaves it unwritten
      const std::int32_t m =
          sign ? (std::int32_t)(~mantissa + 1u) : (std::int32_t)mantissa;
      const std::int8_t s8 =
          (std::int8_t)(std::uint8_t)((std::uint32_t)m >> (23 - mbits + 1));
      const int shift = (int)max_exp - (int)exp;
      std::uint8_t v;
      if (shift >= 32) {
        v = sign ? 0xFFu : 0x00u;
      } else if (shift > 0) {
        v = (std::uint8_t)(std::int8_t)(((int)s8 + (1 << (shift - 1))) >>
                                        shift);
      } else {
        v = (std::uint8_t)(std::int8_t)((int)s8 >> shift);
      }
      res[ci++] = v;
    }
    res[ci - 9] = (std::uint8_t)max_exp;
    // The reference helper advances the write cursor past the exponent slot
    // here (helper.h's "currentIndex++" after "res[currentIndex - 9]"), which
    // is what makes the stride 9 bytes rather than 8.
    ci++;
    start = end;
    if (start >= size)
      break;
  }
  return res;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5 || argc > 7) {
    std::fprintf(
        stderr,
        "usage: %s <w_raw> <type_id> <W_rows> <W_cols> [k_tile] [n_tile] > "
        "out.bin\n",
        argv[0]);
    return 2;
  }
  const char* raw_path = argv[1];
  const int type_id = std::atoi(argv[2]);
  const std::size_t n_out = std::strtoull(argv[3], nullptr, 10);  // W_rows = N
  const std::size_t k = std::strtoull(argv[4], nullptr, 10);      // W_cols = K
  const int l1_k = argc > 5 ? std::atoi(argv[5]) : 64;
  const int l1_n = argc > 6 ? std::atoi(argv[6]) : 128;

  const atb_pack::QuantType* qt = atb_pack::LookupType(type_id);
  if (qt == nullptr) {
    std::fprintf(stderr, "unsupported type id %d\n", type_id);
    return 1;
  }
  if (k % qt->block != 0 || k % (std::size_t)l1_k != 0 ||
      n_out % (std::size_t)l1_n != 0) {
    std::fprintf(stderr,
                 "K must be a multiple of %zu and of k_tile; N of n_tile\n",
                 qt->block);
    return 1;
  }

  const std::size_t row_bytes = (k / qt->block) * qt->block_bytes;
  std::vector<std::uint8_t> raw(row_bytes * n_out);
  std::FILE* fp = std::fopen(raw_path, "rb");
  if (!fp) {
    std::perror("open");
    return 1;
  }
  if (std::fread(raw.data(), 1, raw.size(), fp) != raw.size())
    return 1;
  std::fclose(fp);

  std::vector<float> w(n_out * k);
  for (std::size_t r = 0; r < n_out; ++r)
    qt->dequant(raw.data() + r * row_bytes, w.data() + r * k, k);

  // B row-major [K, N] = W^T.
  std::vector<float> b(k * n_out);
  for (std::size_t kk = 0; kk < k; ++kk)
    for (std::size_t n = 0; n < n_out; ++n)
      b[kk * n_out + n] = w[n * k + kk];

  const std::vector<float> bs = ShuffleB(b, (int)k, (int)n_out, l1_k, l1_n);
  const std::vector<std::uint8_t> packed = Bfp16Encode(bs);
  if (std::fwrite(packed.data(), 1, packed.size(), stdout) != packed.size())
    return 1;
  std::fprintf(
      stderr,
      "packed %s %zu x %zu (B %zu x %zu) -> %zu bytes, k_tile=%d n_tile=%d\n",
      qt->name, n_out, k, k, n_out, packed.size(), l1_k, l1_n);
  return 0;
}
