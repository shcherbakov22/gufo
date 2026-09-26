#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "src/core/quant/ggml_dequant.hpp"

// Does bfp16 (8 elements sharing one exponent, 8-bit magnitudes) preserve
// dequantised IQ3_XXS weights?  Measures the round-trip error the NPU
// conversion would add on real weights.
int main(int argc, char** argv) {
  const std::size_t k = 5120, rows = 512;
  const std::size_t row_bytes = (k / 256) * 98;
  const char* path = argc > 1 ? argv[1] : "/tmp/iq3xxs_real.bin";
  FILE* fp = std::fopen(path, "rb");
  if (!fp) { std::perror("open"); return 1; }
  std::vector<std::uint8_t> raw(row_bytes * rows);
  if (std::fread(raw.data(), 1, raw.size(), fp) != raw.size()) return 1;
  std::fclose(fp);
  std::vector<float> w(k * rows);
  for (std::size_t r = 0; r < rows; ++r)
    gufo::quant::DequantizeIQ3_XXS(raw.data() + r * row_bytes,
                                   w.data() + r * k, k);
  double sum2 = 0, err2 = 0, maxrel = 0, maxabs = 0;
  for (float v : w) { sum2 += (double)v * v; if (std::fabs(v) > maxabs) maxabs = std::fabs(v); }
  for (int bits = 7; bits <= 8; ++bits) {
    const double levels = std::ldexp(1.0, bits) - 1.0;
    err2 = 0; maxrel = 0;
    for (std::size_t i = 0; i < w.size(); i += 8) {
      double m = 0;
      for (std::size_t j = 0; j < 8; ++j) m = std::max(m, (double)std::fabs(w[i + j]));
      if (m == 0) continue;
      int e; std::frexp(m, &e);
      const double scale = std::ldexp(1.0, e) / (levels + 1.0);
      for (std::size_t j = 0; j < 8; ++j) {
        const double x = w[i + j];
        double mag = std::round(std::fabs(x) / scale);
        if (mag > levels) mag = levels;
        const double xr = std::copysign(mag * scale, x);
        const double d = x - xr;
        err2 += d * d;
        if (std::fabs(x) > 1e-8) maxrel = std::max(maxrel, std::fabs(d) / std::fabs(x));
      }
    }
    const double rms = std::sqrt(err2 / w.size());
    const double rel = rms / std::sqrt(sum2 / w.size());
    std::printf("bfp16 magnitude_bits=%d  RMS abs err=%.6g  RMS/||w||=%.4f%%  max rel=%.4f%%\n",
                bits, rms, 100.0 * rel, 100.0 * maxrel);
  }
  std::printf("weights: %zu rows x %zu, max|w|=%.4g, RMS|w|=%.4g\n",
              rows, k, maxabs, std::sqrt(sum2 / w.size()));
  return 0;
}