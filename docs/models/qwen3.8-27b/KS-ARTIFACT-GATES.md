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
followed by `gufo bench -m OUT -p 2048 -n 1 -r 1`.
