#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

// A-operand packing benchmark: shuffle a (batch x hidden) activation into the
// ATB A L1 layout and encode it to bfp16, serially and with N threads. The
// runtime path needs this per prefill call, so its cost is what decides whether
// the NPU FFN path can be fed from the host at all.
//
//   atb_act_bench <act_fp16> <M> <K> [m_tile] [k_tile]

namespace {

// gemm_atb::layout_A_2x1_8x8block: 2x1 vertically-stacked super-blocks of 8x8
// row-major sub-blocks.
void ShuffleA2x1(const float* in, int rows, int cols, int row0, int col0,
                 int ld, float* out) {
  const int block_cols = cols / 8;
  int oi = 0;
  for (int sbr = 0; sbr < rows / 8; sbr += 2)
    for (int sbc = 0; sbc < block_cols; sbc++)
      for (int bis = 0; bis < 2; bis++) {
        const int cbr = sbr + bis;
        for (int rib = 0; rib < 8; rib++)
          for (int cib = 0; cib < 8; cib++)
            out[oi++] = in[(std::size_t)(row0 + cbr * 8 + rib) * ld + col0 +
                           sbc * 8 + cib];
      }
}

// gemm_atb::layout_A_L1_2x1_8x8block: row-major L1 tiles.
void ShuffleARange(const std::vector<float>& in, int M, int K, int l1_m,
                   int l1_k, int t0, int t1, std::vector<float>& out) {
  const int l1_cols = K / l1_k;
  for (int t = t0; t < t1; t++) {
    const int l1r = t / l1_cols, l1c = t % l1_cols;
    ShuffleA2x1(in.data(), l1_m, l1_k, l1r * l1_m, l1c * l1_k, K,
                out.data() + (std::size_t)t * l1_m * l1_k);
  }
}

// bfp16 encode with round-to-nearest, shared exponent per 8. Block-independent,
// so ranges split cleanly.
void EncodeRange(const float* a, std::size_t start, std::size_t end,
                 std::uint8_t* res) {
  const int mbits = 7;
  for (std::size_t b = start; b < end; b++) {
    const std::size_t lo = b * 8, hi = lo + 8;
    std::uint32_t max_exp = 0;
    for (std::size_t i = lo; i < hi; i++) {
      std::uint32_t x;
      std::memcpy(&x, &a[i], 4);
      max_exp = std::max(max_exp, (x >> 23) & 0xFFu);
    }
    std::uint8_t* dst = res + b * 9;
    dst[0] = (std::uint8_t)max_exp;
    for (std::size_t i = lo; i < hi; i++) {
      std::uint32_t x;
      std::memcpy(&x, &a[i], 4);
      const std::uint32_t sign = x & 0x80000000u;
      const std::uint32_t exp = (x >> 23) & 0xFFu;
      std::uint32_t mantissa = x & 0x007FFFFFu;
      if (exp)
        mantissa |= 0x00800000u;
      if (exp >= 255)
        continue;
      const std::int32_t m =
          sign ? (std::int32_t)(~mantissa + 1u) : (std::int32_t)mantissa;
      const std::int8_t s8 =
          (std::int8_t)(std::uint8_t)((std::uint32_t)m >> (23 - mbits + 1));
      const int shift = (int)max_exp - (int)exp;
      int v;
      if (shift >= 32)
        v = sign ? -1 : 0;
      else if (shift > 0)
        v = ((int)s8 + (1 << (shift - 1))) >> shift;
      else
        v = (int)s8 >> shift;
      dst[1 + (i - lo)] = (std::uint8_t)(std::int8_t)v;
    }
  }
}

template<typename F>
double TimeIt(F&& fn, int reps = 3) {
  double best = 1e30;
  for (int r = 0; r < reps; r++) {
    const auto t0 = std::chrono::high_resolution_clock::now();
    fn();
    const auto t1 = std::chrono::high_resolution_clock::now();
    best = std::min(best,
                    std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  return best;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s <act_fp16> <M> <K> [m_tile] [k_tile]\n",
                 argv[0]);
    return 2;
  }
  const int M = std::atoi(argv[2]);
  const int K = std::atoi(argv[3]);
  const int l1_m = argc > 4 ? std::atoi(argv[4]) : 128;
  const int l1_k = argc > 5 ? std::atoi(argv[5]) : 64;

  std::FILE* fp = std::fopen(argv[1], "rb");
  if (!fp) {
    std::perror("open");
    return 1;
  }
  std::vector<std::uint16_t> half((std::size_t)M * K);
  if (std::fread(half.data(), 2, half.size(), fp) != half.size())
    return 1;
  std::fclose(fp);
  std::vector<float> a((std::size_t)M * K);
  for (std::size_t i = 0; i < a.size(); i++) {  // fp16 -> fp32
    const std::uint32_t h = half[i];
    const std::uint32_t s = (h & 0x8000u) << 16, e = (h >> 10) & 0x1Fu,
                        m = h & 0x3FFu;
    std::uint32_t bits = 0;
    if (e == 0)
      bits = s;  // subnormal/flush
    else if (e == 31)
      bits = s | 0x7F800000u | (m << 13);
    else
      bits = s | ((e + 112) << 23) | (m << 13);
    std::memcpy(&a[i], &bits, 4);
  }

  const int tiles = (M / l1_m) * (K / l1_k);
  std::vector<float> shuffled(a.size());
  std::vector<std::uint8_t> packed(a.size() + a.size() / 8);

  const unsigned hw = std::thread::hardware_concurrency();
  std::printf(
      "activation %dx%d, L1 tile %dx%d, %d tiles, %u hardware threads\n", M, K,
      l1_m, l1_k, tiles, hw);
  std::printf("%8s %14s %14s %14s\n", "threads", "shuffle ms", "encode ms",
              "total ms");
  for (unsigned nt : {1u, 2u, 4u, 8u, 16u, 32u}) {
    if (nt > hw * 2)
      continue;
    const double sh = TimeIt([&] {
      std::vector<std::thread> th;
      for (unsigned t = 0; t < nt; t++) {
        const int lo = (int)((long long)tiles * t / nt);
        const int hi = (int)((long long)tiles * (t + 1) / nt);
        th.emplace_back([&, lo, hi] {
          ShuffleARange(a, M, K, l1_m, l1_k, lo, hi, shuffled);
        });
      }
      for (auto& x : th)
        x.join();
    });
    const std::size_t blocks = a.size() / 8;
    const double en = TimeIt([&] {
      std::vector<std::thread> th;
      for (unsigned t = 0; t < nt; t++) {
        const std::size_t lo = blocks * t / nt, hi = blocks * (t + 1) / nt;
        th.emplace_back([&, lo, hi] {
          EncodeRange(shuffled.data(), lo, hi, packed.data());
        });
      }
      for (auto& x : th)
        x.join();
    });
    std::printf("%8u %14.2f %14.2f %14.2f\n", nt, sh, en, sh + en);
  }
  return 0;
}
