// Oracle for atb_decode_c: the vendored host's own C path.
//
//   atb_c_oracle <packed_c.bin> <M> <N> <out_fp32.bin>
//
// bfp16ebs8ToFloat then layout_inverse_C_L1_2x2_8x8block, exactly as the mlir-aie
// ATB host verifies its own results. If the GPU decoder disagrees with this, the
// GPU decoder is wrong.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../helper.h"
#include "gemm_atb_layout.h"

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr, "usage: %s <packed_c.bin> <M> <N> <out_fp32.bin>\n",
                 argv[0]);
    return 2;
  }
  const int M = std::atoi(argv[2]);
  const int N = std::atoi(argv[3]);
  const std::size_t count = static_cast<std::size_t>(M) * N;
  const std::size_t packed_bytes = count * 9 / 8;

  std::vector<std::uint8_t> packed(packed_bytes);
  std::FILE* fp = std::fopen(argv[1], "rb");
  if (fp == nullptr || std::fread(packed.data(), 1, packed_bytes, fp) != packed_bytes) {
    std::fprintf(stderr, "cannot read %s\n", argv[1]);
    return 1;
  }
  std::fclose(fp);

  std::vector<float> decoded =
      bfp16ebs8ToFloat(static_cast<int>(packed_bytes), packed.data(), 0);
  std::vector<float> out =
      gemm_atb::layout_inverse_C_L1_2x2_8x8block(decoded, M, N, 512, 128);

  std::FILE* of = std::fopen(argv[4], "wb");
  if (of == nullptr) {
    std::fprintf(stderr, "cannot write %s\n", argv[4]);
    return 1;
  }
  std::fwrite(out.data(), sizeof(float), count, of);
  std::fclose(of);
  std::printf("oracle wrote %zu floats\n", count);
  return 0;
}
