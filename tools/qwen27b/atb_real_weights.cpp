#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "tools/qwen27b/atb_quant_types.hpp"

// Produce the ATB GEMM operand files for a real quantised FFN weight tensor.
//
//   atb_real_weights <w_raw> <type_id> <W_rows> <W_cols> <M> <out_A.f32>
//   <out_B.f32>
//
// w_raw holds W as raw quantisation blocks, row-major [W_rows, W_cols].  The
// GEMM consumes B as row-major [K, N] with B[k][n] = W[n][k], so the
// dequantised weight is written transposed.  A is a fixed-seed standard normal
// [M, K]; the bfp16 error is a per-element relative effect, so the output error
// is insensitive to A's scale.
int main(int argc, char** argv) {
  if (argc != 8) {
    std::fprintf(
        stderr,
        "usage: %s <w_raw> <type_id> <W_rows> <W_cols> <M> <out_A.f32> "
        "<out_B.f32>\n",
        argv[0]);
    return 2;
  }
  const char* raw_path = argv[1];
  const int type_id = std::atoi(argv[2]);
  const std::size_t n_rows = std::strtoull(argv[3], nullptr, 10);  // W_rows = N
  const std::size_t k = std::strtoull(argv[4], nullptr, 10);       // W_cols = K
  const std::size_t m = std::strtoull(argv[5], nullptr, 10);
  const char* a_path = argv[6];
  const char* b_path = argv[7];

  const atb_pack::QuantType* qt = atb_pack::LookupType(type_id);
  if (qt == nullptr) {
    std::fprintf(stderr, "unsupported type id %d\n", type_id);
    return 1;
  }
  if (k % qt->block != 0) {
    std::fprintf(stderr, "K must be a multiple of %zu\n", qt->block);
    return 1;
  }
  const std::size_t row_bytes = (k / qt->block) * qt->block_bytes;
  std::vector<std::uint8_t> raw(row_bytes * n_rows);
  FILE* fp = std::fopen(raw_path, "rb");
  if (!fp) {
    std::perror("open w_raw");
    return 1;
  }
  if (std::fread(raw.data(), 1, raw.size(), fp) != raw.size()) {
    std::fprintf(stderr, "short read on %s\n", raw_path);
    return 1;
  }
  std::fclose(fp);

  std::vector<float> w(n_rows * k);
  for (std::size_t r = 0; r < n_rows; ++r)
    qt->dequant(raw.data() + r * row_bytes, w.data() + r * k, k);

  double sum2 = 0, mx = 0;
  for (float v : w) {
    sum2 += (double)v * v;
    mx = std::max(mx, (double)std::fabs(v));
  }
  std::printf("W %zu x %zu (%s) dequantised: max|w|=%.5g RMS|w|=%.5g\n", n_rows,
              k, qt->name, mx, std::sqrt(sum2 / (double)w.size()));

  // B is row-major [K, N] = W^T.
  {
    FILE* bf = std::fopen(b_path, "wb");
    if (!bf) {
      std::perror("open out_B");
      return 1;
    }
    const std::size_t chunk = 4096;
    std::vector<float> buf(std::min(chunk, k) * n_rows);
    for (std::size_t k0 = 0; k0 < k; k0 += chunk) {
      const std::size_t kk = std::min(chunk, k - k0);
      for (std::size_t i = 0; i < kk; ++i)
        for (std::size_t n = 0; n < n_rows; ++n)
          buf[i * n_rows + n] = w[n * k + (k0 + i)];
      std::fwrite(buf.data(), sizeof(float), kk * n_rows, bf);
    }
    std::fclose(bf);
  }

  // A: fixed-seed standard normal [M, K].
  {
    std::mt19937_64 rng(0x5eed1234ULL);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    FILE* af = std::fopen(a_path, "wb");
    if (!af) {
      std::perror("open out_A");
      return 1;
    }
    std::vector<float> a(1u << 20);
    std::size_t left = m * k;
    while (left) {
      const std::size_t take = std::min(left, a.size());
      for (std::size_t i = 0; i < take; ++i)
        a[i] = nd(rng);
      std::fwrite(a.data(), sizeof(float), take, af);
      left -= take;
    }
    std::fclose(af);
  }
  std::printf("wrote %s (%zu x %zu) and %s (%zu x %zu)\n", b_path, k, n_rows,
              a_path, m, k);
  return 0;
}
