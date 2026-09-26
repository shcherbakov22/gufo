# Why the K_S-derived quant is slow in Gufo (and Q4_K_XL is not)

Filed from the prefill-routing investigation. The numbers below are tied to the
named artifacts in `/home/q/models/gufo-sweep/` and were not re-measured after
the later prefill kernel work; the routing weakness they describe -- un-quantized
`ssm_alpha`/`ssm_beta` taking a slow small-n path -- is still open.

Answer: **it is not the Q4_K/Q5_K mix, and not the Q6_K content. It is 96 tiny GDN
gate tensors — 48 × `ssm_alpha.weight` left in F32 and 48 × `ssm_beta.weight` in
Q8_0 — that the UD-Q4_K_S-derived artifact carries un-quantized. Gufo runs those
on a slow path and they cost 27% of prefill. llama.cpp does not care (−1%).**

## Quant type lists (element-weighted)

| file | mix | size |
|---|---|---|
| `UD-Q4_K_S-requant294-Q4K` | Q4_K 91.22%, Q6_K 8.63%, Q8_0 0.10%, F32 0.05% | 14.95 GiB |
| `UD-Q4_K_S` | IQ4_XS 43.37%, Q4_K 19.06%, Q5_K 13.39%, Q3_K 8.57%, Q6_K 6.78%, IQ3_S 4.89%, IQ3_XXS 1.63%, IQ4_NL 1.50%, IQ2_XS/IQ2_S 0.33% ea, Q8_0 0.14% | 14.29 GiB |
| `UD-Q4_K_XL` (Gufo target) | tensor counts: F32 360, Q5_K 191, Q8_0 110, IQ4_XS 70, Q4_K 69, Q6_K 56, IQ4_NL 6, Q3_K 3, IQ3_S 1 | 16.35 GiB |

`UD-Q4_K_XL` has **360 F32 tensors — exactly the standard count** — so its
`ssm_alpha`/`ssm_beta` are quantized. The K_S-derived requant has **408 F32**.

## Measurements (Gufo `bench -p 2048 -n 1 -r 1`, same box, same engine)

| file | change vs pure Q4_K | size | Gufo pp2048 | llama.cpp pp2048 |
|---|---|---:|---:|---:|
| `base_q4kpure` | — (uniform Q4_K, 506 Q4_K + 360 F32) | 14.33 GiB | **586.46** | 385.51 |
| `ssm_f32q8` | **only** `ssm_alpha`→F32, `ssm_beta`→Q8_0 (96 tensors) | 14.37 GiB | **427.24** | 381.73 |
| `q6_big` | 34 big tensors →Q6_K (incl. `output`, `attn_output`) | 14.88 GiB | 579.71 | 368.61 |
| `ffn_q5k` | 195 FFN tensors →Q5_K | 16.35 GiB | 564.76 | — |
| `ffn_q6k` | 195 FFN tensors →Q6_K | 18.50 GiB | 459.39 | — |
| `requant294-Q4K` (original) | Unsloth UD mix | 14.95 GiB | **428.51** | 375.61 |

The ssm ablation (+0.04 GiB) reproduces the entire deficit: **586.46 → 427.24**
vs the real artifact's 428.51. Q6_K on the big matmuls costs 1%.

Also excluded:
* prompt length / chunking — Gufo pp is flat: 512→407.30, 1024→431.73,
  2048→428.51, 4096→423.31 t/s on the requant.
* workload mismatch — Gufo's own `bench.json` is
  `prompt_tokens: 2048, output_tokens: 128, temperature: 0, seed: 1`.
* layer routing — `GUFO_DISPATCH_TELEMETRY` shows **identical** per-layer route
  fingerprints for the requant and the pure file (32×attention fp6, 64×attention
  fp7, 96×ssm fp0, 192×ssm fp1), so the difference is inside the per-GEMM
  dispatch, not the layer policy.

## Interpretation

Per GDN layer Gufo must compute two (2048×5120)·(5120×48) projections. There are
48 GDN layers, so 96 of them per prefill. Total compute is trivial (~48 GFLOP),
but the deficit is ~1.3 s of a 4.8 s prefill — i.e. ~13 ms per tiny projection.
That is a fixed per-op cost (dense hipBLASLt / fallback path for F32 and Q8_0
weights with n=48), not arithmetic.

llama.cpp's Vulkan backend is insensitive to the same change (−1.0%), so this is
a Gufo kernel-routing weakness rather than a property of the quant.

**Corollary:** Gufo's claimed 656 t/s pp2048 on `UD-Q4_K_XL` is plausible — a
plain uniform Q4_K of the same model already reaches 586 on this box, and XL's
gates are quantized.

## Fixes

1. For this artifact: force `ssm_alpha`/`ssm_beta` to Q4_K.  `base_q4kpure.gguf`
   is exactly that and runs at 586 t/s (+37% vs the requant, +55% vs llama.cpp
   on the same file). Note it is a double re-quant, so quality is not endorsed.
2. Upstream-style fix in Gufo: give the F32 and Q8_0 small-*n* prefill GEMMs
   the same int8-WMMA quant-direct treatment (or fuse the two GDN gate
   projections into one kernel), so any artifact with un-quantized gates is fast.

## Artifacts

`/home/q/models/gufo-sweep/` (79 GiB): `base_q4kpure.gguf`, `ssm_f32q8.gguf`,
`q6_big.gguf`, `ffn_q5k.gguf`, `ffn_q6k.gguf` + `.quant.log` files.
Reproduce with `llama-quantize --pure --allow-requantize [--tensor-type ...] IN OUT Q4_K`

## Correction: the mechanism is a global gate, not the 96 tiny projections

The attribution above -- "96 tiny GDN projections on a slow path, ~13 ms each" --
is wrong, and measurement says so.

`UseQwen27bFp16Prefill` is **global and all-or-nothing**. It requires every GEMM
tensor role in every layer to be Q8_0 or `IsNativeWmmaQuant`, and that includes
`ssm_alpha`/`ssm_beta`. A single F32 `ssm_alpha` does not make 48 projections
slow; it takes the *whole model* off the fp16 WMMA path in
[prefill_chunk.cpp](../../../src/models/qwen/hip/prefill_chunk.cpp). The FFN dual
fused kernel is launched only `if (half_prefill)`, and once `half_prefill` is
false every native quant GEMM drops to the int8-activation route -- measured at
1.4-1.5x slower per FFN GEMM (14.8-17.5 ms against 11.5-13.0 ms).

The ablations' own telemetry already pointed here: per-layer route fingerprints
were identical between the requant and the pure file, so "the difference is
inside the per-GEMM dispatch". `half_prefill` is exactly such a per-GEMM
dispatch switch, and it does not appear in the layer route plan.

**Measured.** Relaxing the gate so `ssm_alpha`/`ssm_beta` do not trip it, and
routing `gemm_weight` per tensor instead of per model, on `ssm_f32q8.gguf`,
`-p 2048 -n 1`:

| | pp2048 t/s |
| --- | --- |
| `base_q4kpure` (all native, gate on) | 555 / 589 / 567 |
| `ssm_f32q8` before (gate off) | **424.46 +/- 0.68** |
| `ssm_f32q8` after (per-tensor gate) | **579.61 +/- 7.98** |

That is +36.6%, back to parity with the all-native file.

**But that configuration is numerically invalid, and must not be shipped as it
stands.** `--validate-prefill 2048`:

| file | cosine | max_abs_diff |
| --- | --- | --- |
| `base_q4kpure` (unaffected control) | 0.99999970 | 0.0076 |
| `ssm_f32q8` with the per-tensor gate | **0.98234493** | **1.67** |

The cause: under `half_prefill` the RMSNorm writes only the BF16 staging buffer.
The FP32 buffer `d_normed` is never populated on that path, but the F32 route
(`kHipPrefillF32Blas`) reads `fp32_input = d_normed`. So the relaxed gate fed
`ssm_alpha` a stale buffer and still measured fast. A *more* precise alpha (F32
rather than Q4_K) cannot explain a 0.982 cosine against a 0.9999997 control; only
wrong input can.

**So the fix is two parts, not one:**

1. dispatch `gemm_weight` per tensor rather than per model, so one foreign type
   costs its own projection instead of the whole model's fast path; and
2. make the fp16 norm path emit the FP32 normed row whenever any consumer of that
   row is routed off the fp16 path, so the F32 route has valid input.

Part 1 alone is worth roughly +36% on such artifacts and is what the numbers
above measure; part 2 is what makes it correct. Both are small, and together they
replace the two fixes originally proposed here (requantizing the artifact, or
writing new int8-WMMA small-n kernels / fusing the GDN gate projections).

### Implemented, and the trap in part 2

The first attempt at part 2 reused `LaunchBatchedRMSNormFp16` and passed
`arena_.d_normed` in its fourth parameter, on the assumption that the argument
after the weight was an FP32 output (the sibling `LaunchBatchedRMSNorm` takes
`(x, weight, out_fp32, out_bf16, ...)`). It is not: the fp16 kernel's signature
is `(input, residual, weight, float* sum_out, void* output, ...)`, and the
fourth argument is the **reduction sum**. That silently wrote a sum buffer into
`d_normed`, which `ssm_alpha` then read. It still measured 579.61 t/s, and it
still passed `top1_match=yes finite=yes` -- only the cosine gave it away, at
0.9934 against a pre-change 0.99936. The fp16 norm kernel cannot produce an FP32
normed row at all, so the correct part 2 is a second pass with
`LaunchBatchedRMSNorm`, paid only by artifacts that have a foreign gate type.

Final state, `-p 2048 -n 1`, `--validate-prefill 2048`:

| `ssm_f32q8` | cosine | max_abs_diff | pp2048 t/s |
| --- | --- | --- | --- |
| before | 0.99935561 | 0.3754 | 424.46 +/- 0.68 |
| per-tensor gate, no FP32 row (invalid) | 0.98234493 | 1.67 | 579.61 +/- 7.98 |
| per-tensor gate, sum buffer reused (invalid) | 0.99340147 | 1.0870 | -- |
| **per-tensor gate + FP32 norm pass** | **0.99999970** | **0.0089** | **504.31 +/- 24.46** |

`base_q4kpure` as the no-regression control: 0.99999970 / 0.0076, 588.42 +/- 8.32
against a 555-589 baseline.

So the fix is +18.8% and *better* numerics than the original, which makes sense:
the fp16 WMMA path is more accurate than the int8-activation path the gate was
forcing every large GEMM onto. The all-native production artifacts take neither
new code path -- `ssm_gate_needs_fp32` is false, so they pay nothing.
followed by `gufo bench -m OUT -p 2048 -n 1 -r 1`.
