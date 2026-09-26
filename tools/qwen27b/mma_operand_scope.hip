// Does the WMMA instruction read the upper half-warp's operand registers?
//
// `v_wmma_f32_16x16x16_f16` declares 8 VGPRs per operand on wave32: 16 halves
// per lane = 1024 B for a 512 B matrix, 2x the data. LoadSwizzled produces
// exactly that duplication (lanes 0-15 and 16-31 read the same LDS rows). If the
// instruction ignores the upper half-warp's operand registers the fragment load
// can drop from two ds_load_b128 to one, halving LDS traffic.
//
// A correctness question, so the answer is exact: zero the upper half-warp's
// operands and compare the accumulator bit for bit.
#include <cstdio>
#include <cstring>
#include <hip/hip_runtime.h>

using v16bf = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__global__ void Mma(const _Float16* __restrict__ a,
                    const _Float16* __restrict__ b,
                    float* __restrict__ out, int break_a, int break_b) {
  const int tid = (int)threadIdx.x;
  const int lane = tid & 31;
  const int row = tid & 15;
  union {
    v16bf f;
    uint4 q[2];
  } av, bv;
  // Lanes 0-15 and 16-31 load the same 32 bytes, as LoadSwizzled does.
  __builtin_memcpy(&av, a + row * 16, 32);
  __builtin_memcpy(&bv, b + row * 16, 32);
  if (break_a != 0 && lane >= 16) av = {};
  if (break_b != 0 && lane >= 16) bv = {};
  v8f acc = {};
  acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(av.f, bv.f, acc);
  for (int i = 0; i < 8; ++i) out[tid * 8 + i] = acc[i];
}

int main() {
  _Float16 *a = nullptr, *b = nullptr;
  float* out = nullptr;
  if (hipMalloc(&a, 512) != hipSuccess) return 1;
  if (hipMalloc(&b, 512) != hipSuccess) return 1;
  if (hipMalloc(&out, 32 * 8 * 4) != hipSuccess) return 1;
  _Float16 ha[256], hb[256];
  for (int i = 0; i < 256; ++i) {
    ha[i] = (_Float16)((i % 17) - 8) * (_Float16)0.125F;
    hb[i] = (_Float16)(((i * 3) % 13) - 6) * (_Float16)0.25F;
  }
  hipMemcpy(a, ha, sizeof(ha), hipMemcpyHostToDevice);
  hipMemcpy(b, hb, sizeof(hb), hipMemcpyHostToDevice);
  float r[4][32 * 8];
  const int ba[4] = {0, 1, 0, 1}, bb[4] = {0, 0, 1, 1};
  for (int k = 0; k < 4; ++k) {
    Mma<<<1, 32>>>(a, b, out, ba[k], bb[k]);
    hipDeviceSynchronize();
    hipMemcpy(r[k], out, sizeof(r[k]), hipMemcpyDeviceToHost);
  }
  const char* label[4] = {"baseline           ", "A upper zeroed     ",
                          "B upper zeroed     ", "both upper zeroed  "};
  for (int k = 0; k < 4; ++k) {
    int diff = 0;
    for (int i = 0; i < 32 * 8; ++i)
      diff += std::memcmp(&r[k][i], &r[0][i], 4) != 0;
    std::printf("%s differing values = %d\n", label[k], diff);
  }
  return 0;
}