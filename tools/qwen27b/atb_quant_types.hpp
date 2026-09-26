#pragma once

#include <cstddef>
#include <cstring>

#include "src/core/quant/ggml_dequant.hpp"

// Quantisation types the ATB repack understands, with the block geometry each
// dequantiser expects. Every (block, block_bytes) pair here was checked against
// the shipped shard's own tensor byte spans before being relied on.
namespace atb_pack {

struct QuantType {
  int id;
  const char* name;
  std::size_t block;        // weights per block
  std::size_t block_bytes;  // bytes per block
  void (*dequant)(const void*, float*, std::size_t);
};

inline const QuantType* LookupType(int id) {
  static const QuantType kTypes[] = {
      {11, "Q3_K", 256, 110, &gufo::quant::DequantizeQ3_K},
      {12, "Q4_K", 256, 144, &gufo::quant::DequantizeQ4_K},
      {13, "Q5_K", 256, 176, &gufo::quant::DequantizeQ5_K},
      {14, "Q6_K", 256, 210, &gufo::quant::DequantizeQ6_K},
      {16, "IQ2_XXS", 256, 66, &gufo::quant::DequantizeIQ2_XXS},
      {17, "IQ2_XS", 256, 74, &gufo::quant::DequantizeIQ2_XS},
      {18, "IQ3_XXS", 256, 98, &gufo::quant::DequantizeIQ3_XXS},
      {21, "IQ3_S", 256, 110, &gufo::quant::DequantizeIQ3_S},
      {22, "IQ2_S", 256, 82, &gufo::quant::DequantizeIQ2_S},
      {23, "IQ4_XS", 256, 136, &gufo::quant::DequantizeIQ4_XS},
  };
  for (const QuantType& t : kTypes)
    if (t.id == id)
      return &t;
  return nullptr;
}

}  // namespace atb_pack
