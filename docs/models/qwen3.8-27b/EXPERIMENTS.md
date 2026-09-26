# Qwen3.8 27B experiments

| Experiment | Decision / qualification |
| --- | --- |
| Quantized verification row groups | Retained per shape; scalar FP32 bits, full target logits and private acceptance/RNG must match. |
| Shared DFlash2 body/context injection | Retained across requests; independent attention, convolution, history and selector state. |
| Partial verification after rejection | Retained with the complete original proposal and unchanged consumed-prefix feedback. |
| BF16 draft gate/up reuse and tiled argmax | Retained; complete head, finite filtering and lowest-ID ties; no extra persistent buffer. |
| Large-chunk KV packing | Retained at shallow and long depths in idle FFN scratch with bounded head groups; exact attention output/log-sum-exp, cache bytes unchanged. |
| BF16 target projection reduction | Fixed per-row FP32 order retained for chunk/cache/continued-image equivalence. |
| Register-cached vision softmax | Retained; byte-identical embeddings with the shared Q4/Q8 projector, unchanged reduction order and memory allocation. |
| 4096-patch vision attention tiles | Retained; byte-identical full embeddings, lower latency and 24 MiB less attention scratch. |
| Alternate tiles/waves/pipeline depths (2026-09-19) | Rejected: no release throughput improvement. |
| Dynamic verification chunk/controller alternatives | No new default retained; seeded private-acceptance policy remains. |
| Parallel mapped-weight reads and larger DFlash packing chunks | Retained: faster cold startup, unchanged encoded weights. |
| Compact active/saved recurrence and valid-prefix KV | Retained: unused attention-layer state and future KV rows excluded; Q4/Q8 sampling, rollback, image and disk replay pass. |
| Verification queries grouped by KV partition | Retained: unchanged arithmetic and storage; eight-token attention 2.95× faster at 32K and 3.66× at 64K. Matched d32K C1 HTTP TG improves 15.7% on Q4_K_XL and 20.2% on Q8_K_XL with Q4 DFlash2; shallow TG and PP remain comparable. |
| Split-K two-position KV prefetch and DPP score reductions | Rejected: small single-row component gains did not consistently help eight-row verification. |
| Shallow attention paired KV loads and DPP reduction | Retained: byte-exact; 12.6% less attention time in the complete AR profile. Matched d0 pp2048/tg128 controls improve TG by 1.4–2.6% across Q4/Q8 AR and Q4 DFlash2, with unchanged outputs and acceptance; PP remains comparable. No extra allocation. |
| Parallel attention from 128 tokens | Retained for Q4 C1 AR and Q4 DFlash2: greedy outputs match; lower error against FP64 in all 12 component controls. Other targets/concurrency remain to qualify. |
| Wider shallow pipeline, removed broadcast and separate score/value passes | Rejected: worse cold-KV or eight-row costs than the retained two-position pipeline. |
| Additional GEMV format specialization | Not promoted: Q8 cold components unchanged; Q6 gains at most 3% in the cold component test. |
| Paired-row activation reuse and shared input sums | Not promoted: dominant Q5_K cold components were unchanged or slower. |
| Two-block Q5 weight prefetch | Rejected: byte-exact, but cold-weight throughput is 8–12% lower. |
| Seven-row Q5 token scheduling and output-row groups | Not promoted: byte-exact across three projection shapes and input scales, but cold-weight gains are at most 1.5%; several variants regress. |
| Compact seven-row Q5 activation staging | Rejected: byte-exact and lower register/LDS use, but cold-weight gains are under 1% on gate/up; down and attention projections regress. Token-step and wave-count variants do not recover a useful gain. |
| Seven-row Q5 row-first FMA scheduling | Rejected: byte-exact on three cold-weight projection shapes; most variants are 1–3% slower. |
| Seven-row Q5 wave32 and row-first scheduling | Rejected: byte-exact on three cold-weight shapes and three input scales; no consistent useful gain over wave64. |
| Fused SSM format specialization and larger output-row groups | Not promoted: no useful cold-weight gain; larger groups were slower despite byte-exact outputs. |
| 64–256 attention partitions | Rejected: no improvement at 32K, with different FP32 rounding. |
| Split-attention graph replay | Rejected: matched Q4 AR C1 d0 remains 11.90 tok/s. The existing launch path is already 97.5% GPU-busy. |
| Graph identity includes target feature taps | Correctness fix: changing captured layers or their order must replace the graph's feature-copy operations. |
| Residual/RMSNorm fusion and fewer reduction barriers | Rejected: byte-exact component gains did not survive the full model. Q4 AR C1 remains 11.90 tok/s and profile GPU time slightly increases. |
| Multiple attention heads per thread block | Not promoted: byte-exact through 32K, but cold-KV gains are at most 1.5%. |
| Shared four-position KV tile across six query heads | Retained: byte-exact; Q4 AR C1 d32K TG improves 2.9%, with shallow TG and PP retained. Existing scalar/batched and full-logit replay checks pass. |
| Shared KV tile for verification rows | Retained: byte-exact; Q4 DFlash2 C1 d32K TG improves 4.4%. Shallow TG, PP, output hashes and acceptance counts are retained. |
| Two 16-lane verification heads per wave | Retained for multi-row attention: preserves both original lane partials and their reduction tree. Q4 DFlash2 C1 d32K TG improves 5.8%, with shallow TG, PP, greedy output and acceptance retained. Scalar AR keeps 32 lanes. |
| Draft-attention value prefetch and paired query heads | Retained: 32-value prefetch preserves the original FMA order; wider blocks share K/V across two query heads. Byte-exact through 32K; C1 d0/d32K TG improves 0.8–1.0%, with unchanged outputs/acceptance and comparable PP. No persistent allocation. |
| Wider draft prefetch and four query heads per block | Rejected: 64/128-value prefetch and four-head sharing lose to the retained 32-value/two-head layout. |
| Context-aware Q4 draft cost and censored full acceptance | Retained: measured attention growth improves C1 d32K TG by 3.0%; shallow TG, PP, greedy AR agreement and repetitive throughput are retained. Choices remain deterministic from private history and position. |
| Width-weighted acceptance feedback | Retained: removes block-width bias under the controller's geometric model. Q4 C1 d0 TG improves 7.7%, with modest d32K/d64K gains, matched PP and exact greedy/cached sampled replay. Repetition retains full acceptance. |
| Fixed three-proposal default | Rejected: helps the ordinary d32K prompt but slows d0. The adaptive controller remains the default. |
| Context cost without handling saturated acceptance | Rejected: improves ordinary d32K output but slows perfect-acceptance repetition; saturation must allow wider-block probes. |
| Eight lanes per verification head | Not promoted: higher register use and less consistent gains across draft widths than the 16-lane variant. |
| KV sharing between adjacent verification rows | Rejected: byte-exact, but slower at 2K/32K than independent row tiles. |
| Eight-position verification tiles and LDS-only barriers | Not promoted: byte-exact, but gains are small or mixed across 3/7/8 rows, with shallow regressions. |
| Next-tile KV prefetch | Rejected: 144 byte-exact controls through 64K; scalar AR is flat and verification is slower. |
| Query-row workgroup ordering | Not promoted: 144 byte-exact controls; component gains are at most 1.5%, and padded row groups regress. |
| FP32 KV staging in shared memory | Rejected: 192 byte-exact controls; sharing conversions does not offset the extra shared-memory cost. |
| Distributed encoded K/V loads | Rejected: 144 byte-exact partial/output comparisons through 64K; plane-, token- and vector-interleaved loads are slower. |
| Sixteen-row Q5 activation staging in smaller token groups | Rejected: byte-exact on three cold-weight shapes; halving or quartering shared memory adds more work than it saves. |
| Sixteen-row Q5 row-first FMA operand schedules | Rejected: byte-exact on three cold-weight shapes; no throughput gain. |
| Sixteen-row Q5 packed or lane-broadcast headers and scalar scales | Rejected: byte-exact on three cold-weight shapes, but slower than the existing shared-scale decoder. |
| Unsigned Q4/Q5 coefficient conversion | Rejected: byte-exact on three sixteen-row cold-weight shapes, but no useful speed gain or register reduction. |
| Sixteen-row Q5 wave32 | Rejected: byte-exact on three cold-weight shapes, but substantially slower than wave64; the smaller row group also spills registers. |
| Native 32-row Q5 tiles | Rejected: byte-exact on three cold-weight shapes, but slower than paired 16-row tiles across one-to-four output rows per lane; wider accumulators also spill or use private memory. |
| Compact sixteen-row Q5 staging and register activation sums | Rejected: byte-exact on three cold-weight shapes; smaller LDS allocations and alternate bank layouts do not offset the extra work. |
| Batched DFlash2 selector | Retained: exact private token chains/probabilities, two launches per saturated C4 cycle instead of 56, and reused FFN scratch. The focused release pair gains 1.3% TG; C1 PP/TG and output are retained. |
| Six fixed drafts at Q4 C2 | Rejected on perfect-acceptance repetition: exact output, but slower than adaptive. |
| Paired Q4 greedy verification costs | Retained: accounts for the projection cost jump above eight rows. C2 generation improves 20.0% on the Italian/Chinese pair, 12.1% on the pangram/train pair and 21.3% at d32K; repetition, AR equality, C1 PP/TG and private sampled replay are retained. |
| Four-request Q4 greedy verification costs | Retained: accounts for the projection jump above sixteen rows. Ordinary C4 generation improves 26.9% and 13.0%; repetition, d32K PP/TG, AR output and private sampled replay are retained. C1 DFlash2 remains unchanged. |
| Joint C4 draft allocation | Rejected: the Italian/Chinese control retains AR output but proposes 603 tokens for 297 accepted, versus 527 previously, and runs slower. Private sampled/RNG/snapshot tests pass; no production change retained. |
| Two interleaved ten-row Q5 groups | Rejected: byte-exact with production compiler settings, but slower than the existing sixteen-plus-four split on cold-weight FFN projections. |
| IQ4_XS staging both phases and compact shared memory | Not promoted: byte-exact on cold-weight FFN projections, but no consistent gain; the down projection regresses. |
| Exact FP16 integer coefficients with mixed FP32 FMA | Rejected: 2,097,152 instruction controls and both 32-row Q5 FFN projections are byte-exact, but packing coefficients and using mixed FMA runs 4–6% slower. Activations and accumulation remain FP32; no production change. |
| Draft blocks wider than seven proposals | Not attempted: the cached Q4 DFlash2 artifact declares an eight-token block, including the anchor. Its metadata limit is retained. |
| Six-request Q4 greedy verification costs | Retained: ordinary C6 controls improve 6.3% and 16.5%, with a focused 4.3% d32K gain. PP, full acceptance, AR output and private sampled replay are retained; saturated history keeps the full-block probe. |
| Eight-request Q4 greedy verification costs | Retained: ordinary C8 controls improve 10.4% and 13.6%; repetition, d32K, PP and AR output are retained. Other cohort/mode decisions and sampled replay are unchanged. AR remains faster on the harder pair, so profitable fallback still needs work. |
| Three fixed drafts at Q4 concurrency | Rejected as a default: helps C2 low acceptance but slows the higher-acceptance pair and C4 repetition. |
| Two/three query heads per shared KV tile | Rejected: byte-exact, but slower than six-head sharing at 2K and 32K. |
| Wave64 scalar projections and uniform row addresses | Not promoted: cold-weight gains are flat or mixed; some Q4/Q6 shapes regress. |
| Copied weights, including a complete GGUF allocation | Not promoted: the complete-copy control has 3–8% lower cold-weight throughput than mapped weights, with identical outputs. |
| Exact zero-exponent fast path, scalar score broadcast and LDS-only barriers | Not promoted: byte-exact, but no useful gain after shared KV staging. |
| Register-cached decode RMSNorm, tiled C1 argmax and snapshot sizing | Retained with explicit fused square accumulation after expanded tests caught a contraction difference. Existing exact argmax and no logits download merely to count them; full-logit replay now passes. |
| Streaming Q4/Q5 payload loads | Rejected: byte-exact cold-kernel gains regress full-model AR by about 1%. |
| Q6 format specialization and paired/split SSM projections | Not promoted: byte-exact, but negligible gains or regressions; split SSM adds launches. |
| Precomputed affine input sums | Rejected: byte-exact, but no consistent cold-projection benefit after including the preparation pass. |
| Gate/up input sharing, including four simultaneous dots | Not retained: exact kernel outputs and verification checks, but the paired implementation stays flat in full-model Q4 C1 AR despite cold-kernel gains. |
| Huge-page-backed immutable weights | Retained for Q4 C1 AR/DFlash2: identical encoded bytes and greedy/cache results, approximately 2% more AR TG at d0/d32K, comparable PP and unchanged process RSS. Warm AR readiness stays below one second; cold loading remains unmeasured; Q8 execution checks now pass. |
| Residual/RMSNorm fusion after register caching | Rejected: byte-exact and faster as a component, but full-model AR is slower. |
| Separate gate/up waves, staged inputs, fixed FFN geometry and `-O3` | Not retained: byte-exact controls, but no useful isolated mapped-weight gain. Stage weights as production does; device-allocated microbenchmarks overstated earlier gains. |
| Cached small-batch RMSNorm and shared attention partition scales | Retained: explicit FMA and unchanged sum trees pass independent FP64 and byte-exact controls. Q4 C1 DFlash2 gains about 1.1% at d0/d32K; AR, PP, greedy output and sampled cache replay are retained. |
| C1 batched selector and FFN scratch reuse | Retained as simplification: exact token chains/probabilities, fewer launches and 15,616 fewer allocated bytes. The focused profile does not show an isolated selector speed gain. |
| Decode fallback for missing verification replay records | Correctness fix: reuse recorded rows and decode unrecorded rows with the original arithmetic. Full mixed-context and C1–C8 target replay checks pass; the complete recorded fast path is unchanged. |
| Generated-frontier replay and cancellation rollback | Correctness fix: retain decode arithmetic for pending output IDs; reuse the existing verification backup when cancellation interrupts publication. Q4/Q8 live, disk and seeded continuation checks pass without adding a normal-cycle GPU copy. |
| Q8 scalar row interleaving and packed coefficients | Rejected: byte-exact but slower than the existing read-only mapped-weight kernel; static type specialization is speed-neutral. |
| Q8 C1 measured width costs | Rejected: exact d0/d32K output, but no end-to-end gain; the existing policy remains. |
| Q8 verification staging four/eight activation rows | Rejected: byte-exact on 32-row gate/up and down projections, but extra staging phases outweigh the smaller shared-memory allocation. |
| Private Q8 C4 width costs and per-request full-block probes | Rejected: private costs slow repetition; per-request probes recover it but regress ordinary output. |
| Shared Q8 C4 greedy width | Retained: summed private acceptance histories avoid costly ragged verification. Difficult-pair TG improves 12.3%, deep full-checkpoint TG 13.2%; full acceptance, AR output, private sampled replay and C1 pp/tg are retained. |
| Shared Q8 C8 greedy width | Retained: complete measured cycles account for ragged projection costs. Difficult-pair TG improves 21.5% and d32K generated-history forks 17.5%; repetition, AR output and private sampled replay are retained. C1 PP ranges overlap in forward/reverse controls, with TG around 16.1 tok/s. |
| Shared Q8 C2/C6 greedy widths | Retained: measured complete-cycle costs improve difficult-pair TG by 6.1%/28.9% and d32K generated-history forks by 12.9%/10.2%. Repetition, AR output, private sampled replay and C1 pp/tg are retained. |
| Ragged Q8 projection groups and odd-cohort width selection | Retained: 17–36 rows share weight reads with bounded final-group loads/stores. C8 mixed reaches 65.96 tok/s, repetition 116.90 and d32K forks 41.52; outputs match AR. Scalar bits, full logits/features, rollback and private sampled state pass; C1 is retained. C6 remains sensitive to admission order. |
| Q4 costs for shrinking odd cohorts | Retained: avoids oversized verification after a peer finishes. Mixed C4/C6/C8 reaches 71.76/79.78/85.27 tok/s, C8 repetition 121.68 and d32K C4 39.87. Outputs, private sampled replay and C1 behavior are retained. |
| Saturated-cycle cost retune without accounting for partial verification | Rejected: faster full blocks increased wasted proposals and slowed ordinary workloads. |
| Neutral-history probes after isolated short successes | Rejected: recovers repetition but can over-propose on difficult C6 prompts. Retained probes require stronger established acceptance histories. |
| Native 38-row groups using 13/14-token tiles | Rejected: negligible gains or slower cold-weight projections; only the qualified 17–36-row groups are retained. |
| Concurrent generated-reply forks | Retained: freeze the live generated checkpoint before branching, within the snapshot budget. C1 avoids the copy; Q4/Q8 greedy/seeded forks, disk restoration and cancellation pass. |
| Visible causal attention tails | Retained: FP32 FMA over visible tail keys removes chunk-boundary drift while complete tiles keep WMMA. Q8 full logits/features match at 8K across scheduling budgets; four 32K DFlash2 continuations match isolated AR. Later packed V loads and row-first scale reuse recover PP speed without spills; large shallow chunks use the same packed route. |
| Separate causal-tail passes, helper functions and deeper V prefetch | Rejected: exact outputs, but extra work or register pressure makes attention slower. |

Q4 C1 AR leads the pinned d0/d32K controls; Q8 C1 AR matches them.
DFlash2 retains AR output in the focused shallow/deep and concurrent checks.
Current comparisons and unmeasured cells are in [benchmarks](BENCHMARKS.md);
[quality](QUALITY.md) records the quality scope.
Keep the work on one PR branch with incremental, reviewable commits.

Overall target: match llama.cpp generation speed, aiming for a further 10%,
on Q4_K_XL and Q8_K_XL with AR and **Q4_K_M DFlash2** at C1/C2/C4/C6/C8.
Cover both shallow (d0–d16K) and long-context (d32K–d128K) token generation:
screen d0 and d32K, then expand where needed to isolate or qualify the change.
Preserve prompt-processing speed, greedy AR/speculative agreement and sampled
replay. Check cross-engine differences against each engine's AR output.

Iterate with one affected shape and one control. Do not refresh the
full benchmark sweep during exploration. Repeat only to resolve noise or a
failure. Qualify the other configurations after the focused optimization phase,
or earlier only when a specific correctness concern requires it. Remote GPU time
is limited.

Published workload numbers live only in [benchmarks](BENCHMARKS.md); source and
model qualification live in [quality](QUALITY.md).


## Deferred: INT4 WMMA doubles matrix throughput

Measured 2026-09-25 with a register-fed issue loop (operands in registers, eight
independent accumulator chains, non-zero operands so the loop cannot be folded):

| kernel | TMAC/s | TFLOPS |
| --- | --- | --- |
| `wmma_i32_16x16x16_iu8_w32` | 25.6 | 51.1 |
| `wmma_i32_16x16x16_iu4_w32` | 49.6 | 99.2 |

The int4 form has the same 16x16x16 shape but runs at **2x** the int8 rate, so the
matrix ceiling is ~99 TFLOPS rather than the ~51-59 TFLOPS the int8 path reaches.
A pp2048 dense prefill would floor at 1.13 s (1815 t/s) instead of 2.10 s
(976 t/s). Reproduced at two chain depths (1.89x and 1.94x).

Availability on gfx1151 is limited to that one path. `wmma_i32_16x16x32_iu4`
(the larger int4 shape), `wmma_i32_16x16x64_iu8`, the fp8 and `f8f6f4` shapes
and `f32_32x16x128_f4` are gfx12 only, and the VALU int4 dots (`sdot8`,
`sudot8`, `udot8`) need the gfx12 `dot1-insts` target feature.

The 2x is an issue-rate effect: at the same 16x16x16 shape, iu4 retires about
twice the instructions per second of iu8 and its operand fragments are half the
size (2-int vs 4-int per lane), so register and LDS pressure drop as well.

What can feed it is decided by `*linearity*`, not bit width, and the K-quants
and the IQ-quants fall on opposite sides despite similar names. Integer WMMA
multiplies the stored code by the activation, so the code must be proportional
to the value.

Linear, therefore eligible: Q4_K (0..15), **Q3_K** (already decoded to -4..3 with
a per-16 scale and no offset), Q2_K and Q2_0 (0..3), ternary (TQ1_0/TQ2_0),
Q1-style. Q4_K's constant folds into the affine offset it already carries, so a
q-8 remap costs nothing new; Q3_K needs nothing at all, since
`PerHalfScale` already covers its per-16 scale and its values are already
signed. Q4_K plus Q3_K are 25.7% of UD-Q4_K_S and 15.6% of the 3.84 bpw shard,
and neither needs new correction machinery.

Codebook, therefore ineligible at any width: IQ4_XS/IQ4_NL (4-bit grid indices
whose decoded span is -127..113), IQ2_XXS/IQ2_XS/IQ2_S (grid magnitudes reach
43), IQ3_S/IQ3_XXS (reach 62). Q5_K and Q6_K exceed four bits and Q6_K also
carries a per-16 constant.


Both operands are int4, so this also means 4-bit *activations* rather than the
current q8_1. That is a prefill precision change needing quality validation, not
just a kernel change, and it is the main reason the 2x is not free.

Every format is decoded to int8 (`QuantSub16::q[16]`) and multiplied on the int8
WMMA today, so the comparison that matters is int4 against **int8**, not against
some hypothetical narrower path. Any format that can be re-encoded as signed
4-bit values therefore takes the full 2x immediately, whatever its storage width:
Q4_K, Q3_K, Q2_K, Q1_0, Q2_0 and the ternary formats. There is no int2 or int1
WMMA, so the int4 rate is the ceiling, but sub-4-bit formats are currently paying
int8 rates, so moving them to int4 is a 2x in its own right rather than merely a
bandwidth win.

The dividing line is linear versus codebook. No integer WMMA can consume a grid
code, so IQ2, IQ3 and IQ4 stay on int8 permanently. Linear formats are therefore
architecturally favoured on this part, which inverts the usual "IQ is better per
bit" preference: an IQ-heavy shard gets the 2x only on its linear fraction. The
3.84 bpw shard is about 46% IQ3 by bytes, so its int4 gain is capped at the
Q4_K+Q3_K share (15.6%) unless the quantisation mix moves toward linear formats -
a model decision whose value is now quantified.

Deferred until the GEMM delivery work lands. On the current shards the eligible share is Q4_K plus Q3_K (25.7% of UD-Q4_K_S,
15.6% of the 3.84 bpw shard), so expect a low-double-digit percent of prefill
from those two alone, and they need no new offset support. The full 2x needs weights that are linear and at
most 4 bits throughout, which is a model/quantization decision. It is also
orthogonal to the Q2_K chunked-path work: sub-4-bit linear formats become
*eligible* for the int4 path, but their per-16 affine minimum still needs the
per-half activation sum, so that fix is not avoided by moving to int4.

### The int4 activation grid: measured, and the naive version is too lossy

The int4 MMA needs int4 for both operands, so the weight repack being lossless
buys nothing about the activation side. Measured before writing any int4 kernel,
with two environment-gated diagnostics:

* `GUFO_FORCE_Q8_PREFILL` drives the q8_1 activation path on artifacts that would
  otherwise take the fp16 path, so both baselines are reachable from one binary;
* `GUFO_INT4_ACT` pushes the already-quantized q8_1 codes onto the int4 grid in
  place (`scale *= 127/7`, codes `round(q*7/127)` clamped to +/-7, sum sidecar
  rebuilt) at the one call site every activation producer shares.

This is not a proxy. int4 is a subset of int8, and `iu8` and `iu4` share the
K=16 grouping and the int32 accumulator, so storing int4 values in the int8
operand makes the existing int8 WMMA compute *exactly* the integers an int4 WMMA
would. It answers the quality question with no int4 MMA in the build.

ByteShape 3.84 bpw, `-p 2048 -n 1 --validate-prefill 2048`:

| activations | cosine | max abs diff | rmse |
| --- | ---: | ---: | ---: |
| fp16 (the production path) | 0.99999923 | 0.0111 | 0.00239 |
| q8_1 int8 | 0.99965954 | 0.2703 | 0.0513 |
| int4 grid, per-32 symmetric | **0.96105802** | **2.7537** | **0.5274** |

fp16 -> int8 costs 3.4e-4 of cosine. int8 -> int4 costs **3.9e-2**, two orders of
magnitude more, with ten times the rmse. `top1_match` still holds at 2048
tokens, but a 0.961 logit cosine at 10x the error is the regime where the output
distribution is being reshaped rather than perturbed. **The 2x is not free and
the naive grid does not pay for it.**

Three caveats, because they bound what this does and does not say:

1. **This is W4A4 globally.** The instrument rewrites the shared q8_1 buffer, and
   every consumer reads that buffer, so the grid applies to every projection --
   not just the ~104 int4-eligible ones. The mixed design would be strictly
   better than this number, so 0.961 is a floor on quality for it and 0.99966 the
   ceiling. Isolating the mixed case needs per-GEMM activation quantization.
2. **per-32 symmetric, one scale per block.** This is the configuration the
   outlier argument predicts will fail: activations are heavy-tailed, so a block
   containing an outlier spends its scale on that element and leaves the other 31
   with a couple of effective bits. Finer granularity (per-16) or a rotation
   folded into the adjacent weights is the standard mitigation, and neither is
   measured here.
3. **No int4 performance number is implied.** There is no int4 MMA in this build;
   what runs is the int8 path plus an extra re-quantization pass.

So the next question is not "is int4 activations lossless" -- it clearly is not
-- but whether eligibility-scoped quantization plus finer granularity gets the
loss back to the int8 level, where the step from fp16 was only 3.4e-4.

### The end-to-end metric is chaotic at int4 precision

Chasing a discrepancy between two builds of the instrument turned up something
more important than the number. The two builds differed *only* in how the
per-block scale is rounded:

| build | cosine | top1 match |
| --- | ---: | --- |
| `scale = d * (127/7)` | 0.96105802 | yes |
| `scale = (d * 127) / 7` | 0.92354196 | **no** |

Each result reproduces to every printed digit across three runs. The maximum
relative difference between those two scale computations is **1.2e-7** -- about
two ULP, measured over the full exponent range. So a two-ULP change in one stored
scale moves the logit cosine by 0.037 and **flips the top-1 token**, an
amplification of roughly 3e5.

Before concluding it was the scale, the alternative explanations were eliminated:
the code formula does not matter (computing the code as `q * 7/127` or as the
reconstructed quotient `(q*d)/((d*127)/7)` gives bit-identical results from one
binary), the instrument is deterministic, and the source diff contains nothing
else. Whether the residual cause is that last ULP or a codegen difference in the
same kernel, the conclusion is the same: **two semantically equivalent builds of
the same grid disagree by 0.037 of cosine.**

That makes the end-to-end cosine the wrong instrument for this decision, for
three reasons:

* at int4 precision the output is **chaotic** in the perturbation, so "how much
  quality does 4-bit activation lose" has no stable answer at this granularity;
* `--validate-prefill`'s cosine is a *self-consistency* metric -- sequential
  against batched within one build -- so at this sensitivity it samples a chaotic
  distribution rather than measuring quality;
* by contrast the same instrument is stable at int8: fp16 -> int8 costs 3.4e-4,
  and the int8 builds are bit-identical across both (0.99965954 each time).

Clipping the scale does not rescue it, and is non-monotonic: alpha=3 gives
0.75796336 and alpha=5 gives 0.94447929, both with top1 broken. The activation
outliers are load-bearing and cannot be clipped away.

**So the int4 go/no-go needs different instruments:**

1. a **per-GEMM** error metric -- activation quantization error against that
   GEMM's output, which is stable because it does not amplify through 64 layers.
   The oracle suite already measures per-format error this way, so the harness
   exists; and
2. a real held-out **quality** metric (perplexity or KL on text), not
   sequential-vs-batched self-consistency.

Both of those are prerequisites for judging the mixed design, which remains
unmeasured and would perturb far less than the global W4A4 measured here.

### Per-GEMM: what a 4-bit activation actually costs

The stable instrument is offline. `GUFO_DUMP_ACTIVATION=<path>` writes one real
activation buffer once (the post-norm FFN input, fp16, batch x hidden), and
`tools/qwen27b/int4_activation_error.py` reads it alongside a real Q4_K weight
matrix and reports the per-GEMM relative error. No 64-layer amplification, no
self-consistency metric, deterministic.

Real weight `blk.3.attn_q.weight` (Q4_K, 12288x5120), real activation
(2048x5120, layer 0, 256 tokens used), weight side identical in every row:

| GEMM variant | per-GEMM relative error |
| --- | ---: |
| fp16/fp32 activations | 0.00000 (reference) |
| int8 per-32 activations | 0.00569 |
| int4 per-32 activations | **0.10983** |
| int4 per-16 activations | 0.08966 |
| int4 per-8 activations | 0.07153 |
| `weight 4-bit grid (for comparison)` | 0.09736 |

Per-32 block dynamic range (amax/rms; a Gaussian block sits near 2.5-3.5):

| | mean | median | p90 | max |
| --- | ---: | ---: | ---: | ---: |
| weights | 2.36 | 2.31 | 2.83 | 4.55 |
| activations | 2.66 | 2.50 | 3.51 | 5.65 |

Two results, and the first corrects an assumption that was being repeated
above and in the prose around it.

**The activation here is not the outlier problem it is usually assumed to be.**
Post-RMSNorm, the per-32 blocks are nearly as Gaussian as the weights -- 2.66
against 2.36, and the p90 is 3.51. The "activations are heavy-tailed, so the
scale is eaten by an outlier" argument does not apply at this point in the
network, which is why finer granularity buys so little: per-16 recovers 0.090 and
even per-8 only reaches 0.072. Neither is a fix.

**A 4-bit activation adds an error term the same size as the one already
present.** The weight side's 4-bit grid costs 0.097; a 4-bit activation grid
costs 0.110. As independent terms the GEMM's relative error goes from 0.097 to
sqrt(0.097^2 + 0.110^2) = 0.147, so **+51% on the eligible projections**. int8
activations add 0.0057, which is negligible against 0.097.

This is a much better position than the global W4A4 result, and consistently so:
globally the *high-precision* IQ projections -- whose own weight error is small --
each had an 0.11 activation error added on top, which is what broke top-1. The
mixed design never touches them.

So the price of the mixed design is now quantified and stable: **+51% GEMM error
on the ~104 int4-eligible projections, for a 2x MMA rate on them.** Whether the
model tolerates that is a genuine quality question that the sequential-vs-batched
cosine cannot answer -- it needs a held-out metric (perplexity or KL).

### The mixed design, measured

The design the whole exercise pointed at -- per-tensor, not per-model, activation
precision -- is implemented behind `GUFO_INT4_MIXED`. `gemm_weight` takes a
private copy of the Q8_1 activation, pushes that copy onto the int4 grid, and
hands it to the GEMM **only** when the weight type is int4-eligible
(Q4_0/Q4_1/Q4_K/Q3_K/Q2_K). Every other consumer keeps the unmodified int8
buffer, so the coarse grid never reaches a projection that did not opt in.

ByteShape 3.84 bpw, `-p 2048 -n 1 --validate-prefill 2048`, all on the q8_1 path
so the only difference between the last three rows is the activation grid:

| configuration | cosine | top1 match | max abs diff |
| --- | ---: | --- | ---: |
| fp16 activations (production path) | 0.99999923 | yes | 0.0111 |
| int8 q8_1 (the comparison baseline) | 0.99965954 | yes | 0.2703 |
| **mixed int4, eligible GEMMs only** | **0.97941464** | **yes** | 2.0899 |
| global int4, every GEMM | 0.92354196 | **no** | 3.2464 |

Reading:

* the mixed design is decisively better than global -- top-1 is preserved, and it
  takes about half the cosine damage (0.020 against 0.076);
* but against the step it replaces it is large: int8 costs 3.4e-4 relative to
  fp16, and mixed int4 costs a further 2.0e-2 on top of int8, roughly **60x the
  int8 step**;
* the per-GEMM result above predicted the direction (+51% error on ~104
  projections) but not that magnitude, which is a reminder that the model-level
  cosine is not a linear function of per-GEMM error.

**The caveat that must travel with this number:** the metric is
sequential-vs-batched *self-consistency*, not quality, and at the global int4
level it was measured above to be chaotic -- a 2-ULP change moved it by 0.037 and
flipped top-1. The mixed figure reproduces within this build, but whether 0.9794
is a stable characterisation or a draw from a chaotic distribution is **not**
established. `top1_match` is the only field in that table which survived the
earlier sensitivity test, and it is intact here.

State of the decision:

1. **weight repack: lossless** -- provable by capacity, nothing left to measure;
2. **activation: the cost** -- 0.110 per-GEMM against the weight side's own
   0.097, i.e. +51% on the eligible projections, and 2.0e-2 of model-level
   self-consistency;
3. **mixed beats global**, and keeps top-1;
4. **missing: a held-out quality metric.** Without one, "is 2.0e-2 acceptable"
   has no answer, and the int4 kernel work should not start on the strength of a
   self-consistency number.

Recommended next step: build the held-out metric *before* the kernel. If
perplexity or KL moves by less than the run-to-run spread of two fp16 seeds, the
int4 path is worth implementing; if it moves materially, the 2.2x is not
available at four activation bits and the direction closes.

### Is the int8 baseline global, and where the int4 opportunity actually is

Both worth checking, because the mixed comparison only means something if the
baseline is global and the grid really reaches the eligible projections.

**The int8 baseline is global.** On the forced q8 path `reads_q8_act` is true for
every predicate-role tensor in this shard -- the token-type enumeration found
zero non-native roles -- so every GEMM reads the q8_1 activation. The mixed test
then changes only the eligible GEMMs, so the two rows are like-for-like.

**Coverage is real, and it includes the big projections.** An instrumented run
(`GUFO_INT4_REPORT`) confirms the grid reaches every eligible role:

| type | m | k | calls | role |
| --- | ---: | ---: | ---: | --- |
| Q4_K | 10240 | 5120 | 92 | attn_qkv |
| Q4_K | 48 | 5120 | 84 | ssm_alpha |
| Q4_K | 5120 | 6144 | 52 | ssm_out |
| Q4_K | 6144 | 5120 | 44 | attn_gate |
| Q4_K | 12288 | 5120 | 32 | attn_q |
| Q3_K | 17408 | 5120 | 32 | ffn_gate |
| Q4_K | 1024 | 5120 | 28 | attn_k |
| Q4_K | 5120 | 17408 | 16 | ffn_down |
| Q3_K | 48 | 5120 | 12 | ssm_beta |
| Q4_K | 17408 | 5120 | 4 | ffn_up |
| Q2_K | 48 | 5120 | 4 | ssm_beta (the one Q2_K) |

**But the payoff is capped well below 2x, by the quantization mix rather than the
kernel.** Weight elements are proportional to GEMM FLOPs, so the eligible share
is the FLOP share:

| role | all (M el) | int4-eligible (M el) | share |
| --- | ---: | ---: | ---: |
| ffn_down | 5793.4 | 356.5 | 6.2% |
| ffn_gate | 5793.4 | 713.0 | 12.3% |
| ffn_up | 5793.4 | 89.1 | 1.5% |
| attn_qkv | 2516.6 | 1258.3 | 50.0% |
| attn_gate | 1509.9 | 408.9 | 27.1% |
| ssm_out | 1509.9 | 377.5 | 25.0% |
| attn_q | 1069.5 | 566.2 | 52.9% |
| attn_output | 534.8 | 31.5 | 5.9% |
| attn_k | 89.1 | 36.7 | 41.2% |
| ssm_alpha | 11.8 | 2.5 | 20.8% |
| ssm_beta | 11.8 | 3.7 | 31.2% |
| attn_v | 89.1 | 0.0 | 0.0% |
| **total** | **24722.8** | **3843.9** | **15.5%** |

Read that two ways:

* the eligible share is **15.5% of prefill FLOPs**, so at a 2.2x rate on that
  fraction the whole-model ceiling is `1 / (0.845 + 0.155/2.2) = 1.09` -- about
  **9% faster prefill**, not 2x;
* the reason is that the FFN is 70% of prefill and only **6.7%** of it is
  eligible, while attention/SSM is 30% of prefill and **36.6%** of it is. The
  shard's FFN is IQ3_S/IQ4_XS/Q5_K, which is excluded at any width or by width.

So the int4 kernel buys ~9% on this shard for a 2.0e-2 activation regression on
15.5% of the FLOPs. The larger lever is the quantisation mix: a shard whose FFN
were Q4_K rather than IQ would put ~70% of prefill on the eligible side and make
the 2.2x worth implementing. That is a model decision with its own quality
question, and it is where the value is -- which is what the eligibility analysis
at the top of this section already said, now with the FLOP share attached.

### Double-buffering the fp16 stage: blocked by registers, not by LDS

The reasoning that motivated this was sound and worth testing. At BK=4 the fp16
stage is *exactly* 64 KiB, which is why it is single-buffered, which is why
`commit()` sits in its own fenced window and why there are two barriers per
stage. At BK=2 the stage is 32 KiB, so **two** buffers fit exactly: the commit for
the next stage lands in the other parity, it can be scheduled inside the region
the MMAs occupy, and one barrier per stage suffices.

Implemented as a `DoubleBuffer` template arm and measured in one binary against
the BK=4 baseline. Every configuration is bit-exact (`mismatch=0`); medians of
three passes:

| variant | LDS | VGPRs | spills | ms | vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| 256x256 bk4 single (baseline) | 65536 | 192 | 0 | 11.79 | -- |
| 256x256 bk2 single | 32768 | 191 | 0 | 14.14 | +20% |
| 256x256 bk2 **double** | 65536 | 192 | **35** | 19.15 | **+62%** |
| 128x128 w4n2 bk4 single | 32768 | 170 | 0 | 12.75 | +8% |
| 128x128 w4n2 bk4 **double** | 65536 | 256 | **157** | 46.67 | **+296%** |
| 128x128 w4n2 bk2 double | 65536 | -- | -- | 14.80 | +26% |

The LDS arithmetic works and the occupancy confirms it -- the double-buffered arms
report exactly one block per CU, which is what two 32 KiB stages require. What
breaks is the register file. Overlapping the commit with the MMAs means the
accumulators (64 VGPRs), the MMA operand fragments (about 48), the raw weights
already fetched for the next stage and the decode's temporaries are all live at
the same instant. The compiler spills 35 registers at 256x256/bk2 and 157 at
128x128, and the spill cost (a fifth to a third of the runtime) swamps the barrier
saving it was meant to buy.

**This is the second independent confirmation of the same wall.** The decode hoist
failed identically -- 209 spills when the decode alone moved into the MMA region.
The staging work is VALU and LDS, not MMA, so overlapping it with the MMAs is
exactly what requires its registers to be live alongside the accumulators and
operands, and there is not room. The pairing is not "LDS limits, registers spare":
both are spent.

The arithmetic also says the idea would not have paid even spill-free, which is
worth recording so it is not retried. Comparing the two single-buffered arms
isolates the cost of stage granularity: bk2 carries 160 extra barriers **and** 80
extra stage boundaries for +2.35 ms. Halving the barriers back to 160 recovers
about 1.2 ms, and moving the commit (7.3% of the block, 0.86 ms) into the overlap
region recovers at most another 0.86 ms, landing near 12.1 ms -- about 2.5%
*worse* than the 11.79 ms baseline. Doubling the stage count costs more than the
fence it removes.

The only shape that keeps 80 stages is BK=4, and BK=4 at 256x256 is 128 KiB for
two buffers. So the fenced commit window is not recoverable this way either: the
kernel's structure is what 64 KiB of LDS *and* the accumulator-and-operand VGPR
budget jointly permit, and both attempts to beat it died on the register half of
that pair rather than the LDS half.

### Is the int8 kernel unoptimizable? No -- it has more headroom than fp16

It is worth separating two claims that were conflated in the prose above. What
was measured is that the int8 *path* is slower and that its smaller staging does
not convert into speed because the fp32 scale/offset/sum metadata fills the LDS
back up. What was **not** measured is whether the int8 kernel is near its own
limit. It is not.

`tools/qwen27b/prefill_gemm_bench.hip` already benches it. Interleaved with the
fp16 bench in one session, Q4_K, m=17408 k=5120 batch=2048, so both arms see the
same thermal state:

| kernel | ms | TFLOPS | share of its measured ceiling |
| --- | ---: | ---: | ---: |
| fp16 `HalfPrefillGemmKernel`, 256x256 bk4 | ~12.0 | 30.4 | **63%** of 48.35 |
| int8 `WKQuantA8BlockedWmmaGEMMKernel` | ~15.1 (14.3-16.4) | 24.2 | **48%** of 50.31 |

The int8 kernel is **15 percentage points further from its ceiling**. So the
honest answer to "is int8 at its limit" is no: it wastes about half of what the
hardware offers on this shape, and the levers are different from the fp16 ones --
its cost is per-element metadata arithmetic (four fp32 arrays staged as scales,
offsets, activation scales and activation sums, plus the affine offset
correction) and its epilogue, not the barrier structure.

But the headroom is worth more than the first draft of this section claimed, and
that draft's framing was wrong. It compared *instruction* ceilings -- 50.31
against 48.35, +4% -- and concluded there was nothing to win. The whole finding of
this investigation is that the **practical** ceiling is format-dependent: the fp16
kernel reaches only 63% of its instruction ceiling precisely because the 64 KiB
single-buffer rule forces the fenced commit and the two barriers. int8 uses
20-30 KiB and is not subject to that rule, so the comparison that matters is
fp16's 63% *practical* against whatever int8's practical ceiling turns out to be:

| int8 efficiency | TFLOPS | against fp16's achieved 30.4 |
| --- | ---: | ---: |
| 63% (fp16's own) | 31.7 | +4% |
| 75% | 37.7 | +24% |
| 85% | 42.8 | +41% |

At a measured quality cost of 3.4e-4 of cosine, that is a materially better
prospect than int4's ~9% at 2.0e-2 -- **if** the int8 kernel can get there, which
turns entirely on whether its 52% overhead is stalls or work.

The evidence leans toward work, not stalls. The mitigations a smaller footprint
buys are already in place: the int8 kernel reports 2+ blocks per CU and it
prefetches into registers (`kPrefetch`, `r_b`, `r_q0`, `r_q1`). It has the
spare capacity and it is still at 48%, so its overhead is probably the per-element
arithmetic -- four fp32 metadata arrays and the affine offset correction -- and
**halving the operand width does not shrink fp32 metadata.** That is exactly why
the stage did not halve: the weight side costs 1 byte of code plus 8 bytes of
scale and offset per 32 elements, so about 1.25 B/element against fp16's 2, not
0.5. The LDS *instruction* count does halve (one `ds_read_b128` per fragment
instead of two), which is a real saving, but the spare *capacity* is already spent
on the second resident block.

2. **The int8 path carries roughly 12% of non-GEMM overhead.** Model-level, the
   forced q8 path runs 390 t/s against 540 for fp16, about 28% slower, while the
   GEMM alone is only 26% slower -- the rest is the activation quantization passes
   and the fusion the fp16 path gets instead (`LaunchBatchedDualQuantGEMMSwiGLU`
   against the fp16 dual). Fixing the GEMM to parity would still leave the path
   behind.

And the one thing int8's spare LDS could buy -- a larger tile with a better
fragment ratio -- does not help, because the fp16 measurements already showed the
K loop sits at 100% of the MMA peak and 0.75 loads per MMA. A better ratio has
nothing to recover; the bottleneck is the format-independent staging, barriers and
epilogue, which is exactly what int8 cannot change.

**So: optimizable, and possibly worth more than int4** -- but which row of that
table applies is not known, because the int8 kernel has never been phase-profiled.
Its 48% is a fact; whether the missing 52% is a fence/drain that spare LDS can
absorb, or metadata arithmetic that only a narrower scale format can shrink,
decides between +4% and +40%.

### Answered: the int8 overhead is instruction issue, not LDS

> **Superseded twice over, and left in place as a warning.** Every instruction
> count in this section is for `WKQuantA8BlockedWmmaGEMMKernel<128,128,2,4,2,32>`,
> which is **never dispatched** by this model -- as is the wave64 variant a later
> revision substituted for it. The live instantiation is
> `<128,64,4,8,1,Type,32>`, reached only for chunks below 1024 tokens. The two
> conclusions drawn here (issue-bound, route closed) were reached by measuring a
> real kernel correctly and then attributing the result to the wrong one. See
> `int8: the tuning target was dead code, and the A/B harness had no noise floor`
> at the end of this document.

The in-kernel clock instrumentation that worked for fp16 **does not work here**.
Repeated runs of the same binary produced K-loop shares of 57%, 76%, 354% and
726% -- impossible values -- so no phase number from that attempt is reportable.
The fp16 phase table stands; this kernel's body defeats that technique, and the
attempt was reverted rather than left in the tree as a trap.

Static ISA analysis is deterministic and answers the question instead. The K-loop
body of `WKQuantA8BlockedWmmaGEMMKernel<128,128,2,4,2,Q4_K>`, against the fp16
kernel's:

| | fp16 256x256 bk4 | int8 128x128 bk2 |
| --- | ---: | ---: |
| instructions in the loop body | 193 | **861** |
| WMMAs in the loop body | 32 | 32 |
| **instructions per MMA** | **6.03** | **26.91** |

**4.5x more instructions per MMA.** The body is 32 WMMA, 40 `ds_load_b128`, 14
`global_load`, and **572 VALU + 221 SALU**. The fp32 arithmetic inside that is
`v_cvt_f32_i32` 128, `v_fma_f32` 128, `v_mul_f32` 87, `v_fmac_f32` 83 and
`v_dual_fmac_f32` 45 -- **471 fp32 operations applying the per-block weight and
activation scales and the affine offset correction.**

That is the whole story, and it refutes the LDS hypothesis with data:

* the barrier share is the same as fp16 (7.0% against 6.5%) and the commit is
  *cheaper* (3.7% against 7.3%), so there are no stalls for spare capacity to
  absorb -- whatever the spare LDS is, nothing is waiting on it;
* at 26.91 instructions per MMA with about six waves per SIMD, the issue unit is
  fed roughly **83%** of what it can retire in the ~32.5 cycles an `iu8` MMA
  occupies per SIMD. The kernel is **issue-bound**, not latency-bound;
* fp16 at 6.03 instructions per MMA uses about 19% of issue -- that slack is
  exactly what lets it hold the matrix pipe at 100% while int8 cannot.

So the int8 kernel is not waiting on memory, LDS bandwidth or barriers. It is
arithmetic-bound, and halving the operand width does not shrink fp32 scale
application -- which is also why the stage did not halve. Making int8 faster means
removing per-element operations, not freeing memory; and since the int8 rate
ceiling is only 4% above fp16's, the return is capped regardless. **The int8
route is closed on measurement, not on inference.**


## Deferred: Q2_K native prefill (chunked-path eligibility)

Q2_K is the only type in the 3.84 bpw IQ4_XS shard outside
`IsNativeWmmaQuant`, and `prefill_chunk.cpp`'s eligibility predicate tests every
projection of every layer, so one Q2_K tensor (`blk.22.ssm_beta`, 48x5120)
removes the whole model from the chunked prefill path. That matches the
measured shape exactly: pp512 is equal for both shards (both fail the
`batch < 1024` term) while pp1024/pp2048 diverge only for the shard that
carries Q2_K.

Q2_K cannot simply be added to `IsNativeWmmaQuant`: its min is per 16 elements,
while `WKQuantA8BlockedWmmaGEMMKernel` carries one offset per 32-element kb
(`HasOffset` -> `s_off[BK][kOffRowTiles][16]`, corrected once per kb with
`sx = s_sx[kb][ts][sub_lane]`). Feeding it the per-32 activation sum would be
numerically wrong.

Scope:

1. `StoreQ8ActLane<StoreActivationSum>` (prefill_quant_gemm.hpp, and the second
   copy in prefill_quant_gemm.hip): widen the `__shfl_xor` reduction to stop at
   16 lanes so both half sums are available, and write low and high sums.
   `Q8ActSumBytes` doubles; every allocation and the capacity check at
   prefill_quant_gemm.hip:1634 already route through it, so sizing stays
   consistent.
2. Kernel: add `constexpr bool PerHalfOffset = (WType == kQ2_K)`, store both
   offsets per row, and correct with `off1 * sx_full + (off0 - off1) * sx_lo`
   (equivalently two independent half sums). `PerHalfScale` must also cover
   Q2_K, which it already does for the scale half.
3. Dispatch: add Q2_K to the wave32 and wave64 WMMA dispatch and to
   `IsNativeWmmaQuant`.
4. Validation gate: add Q2_K to `kFormats` in the q4kxl oracle test, which
   exercises `TestPrefillGemm` (production vs a CPU model of the same Q8
   activation arithmetic) and `TestSmallBatchExactness` (bit-exact against the
   decode GEMV). Both must be green before the type is promoted.

Risk: the sidecar is shared by every q8-activating format, so a mistake here
corrupts all quantised models rather than failing loudly. Land it with the
oracle suite green at each step, and expect the payoff to be model-level (the
whole shard regaining the chunked path) rather than Q2_K's own 0.0% byte share.


## int4 eligibility, and the one tensor that disqualifies the 3.84 bpw shard

Eligible for int4 packing: the linear Q family at four bits or below.

| format | code | work needed |
| --- | --- | --- |
| Q4_0 | already -8..7 | none |
| Q3_K | already -4..3 | none, PerHalfScale already covers it |
| Q4_K | 0..15 | q-8 remap folded into the existing offset |
| Q4_1 | 0..15 plus per-block min | constant per-32 offset |
| Q2_K | 0..3 plus per-16 min | needs the per-half activation sum |
| Q2_0 | 0..3 | constant offset |
| Q1_0 | 0/1 | constant offset |
| TQ1_0 / TQ2_0 | -1/0/+1 | none |

The weight repack is lossless by capacity: a linear format of four bits or fewer
has at most sixteen distinct codes and signed int4 holds exactly sixteen values,
so the map is a bijection and nothing is requantised. The integer codes are exact;
the fp32 evaluation is merely rearranged, so the result moves at the ULP level and
any bit-exact A/B spanning the change must move together.

Excluded despite being four bits or fewer: the whole IQ family (grid codebooks -
no integer WMMA can consume a grid code) and MXFP4/NVFP4 (4-bit float; the fp4
WMMA builtins are gfx12). Excluded by width: Q5_*, Q6_K, Q8_*. Only Q4_K and Q3_K
are present in the current shards - 25.7% of UD-Q4_K_S and 15.6% of the 3.84 bpw
one - and the remainder is structurally excluded, so the payoff on these artifacts
is a low-double-digit percent of prefill and only a change of quantisation mix
enlarges it.

Verified by enumeration: of the 455 tensors the chunked-prefill predicate inspects
in the 3.84 bpw shard, exactly one is not Q8_0 or a native WMMA quant -
`blk.22.ssm_beta.weight`, 5120x48, Q2_K. One 245k-parameter tensor (0.0009% of
the model) removes the entire 27B model from the fast path. That is the ByteShape
prefill deficit, and closing it is the highest-value fix available on that shard:
make Q2_K native (per-half offset in the WMMA kernel) or re-encode that one tensor
to a native format.

### Q2_K native: implementation details derived so far

Sidecar (`prefill_quant_gemm.hpp` and the second copy in
`prefill_quant_gemm.hip`): reduce over 8,4,2,1 only, then fetch the other half
with a single `__shfl_xor(qsum, 16)`. After that reduction lanes 0..15 hold the
low 16-element sum and lanes 16..31 the high one, so lane 0 can write both; the
per-(token, 32-block) slot becomes two floats and `kQ8ActSumTileBytes` doubles.
Every allocation and the capacity check already route through `Q8ActSumBytes`.

Kernel: the inner loop already keeps the two 16-element halves separate -
`a0/b0` feed `c0` and `a1/b1` feed `c1` - so the correction is simply
`acc -= off0*sx0 + off1*sx1`, or `off1*sx_full + (off0-off1)*sx_lo` if only the
full and low sums are stored. `s_sx` and `s_off` each need a second half; the
least invasive shape is a trailing dimension of `PerHalfOffset ? 32 : 16`, which
leaves existing types bit-identical.

Read the current `s_off` / `s_sx` indexing before choosing the layout: the
offset path already partitions its 16 slots by `half_id * kAccumulatorElements`,
and that decides whether the second half is a new leading dimension or the upper
half of the trailing one. `PerHalfScale` must also gain `kQ2_K`; its list is
currently Q6_K, Q3_K, IQ2_XS, IQ2_S only.

Order: add Q2_K to `kFormats` in the q4kxl oracle test first and watch it fail
red, then the sidecar, then the kernel, then the dispatch and
`IsNativeWmmaQuant`. The sidecar is shared by every quantised format, so that
test is the only thing that catches a mistake; it corrupts silently otherwise.

### Result: Q2_K native, and the 3.84 bpw shard rejoins the chunked path

Implemented and green in `qwen_q4kxl_quant_ops_test`:

* `prefill_quant_gemm.hpp`: `PerHalfScale` gains Q2_K; new `PerHalfOffset` and
  `UsesOffset`; the activation-sum sidecar doubles to two floats per
  (token, 32-block) -- slot 0 the whole-block sum, slot 1 the low 16-element half.
  `StoreQ8ActLane` reduces over 8,4,2,1 and then fetches the other half with one
  `__shfl_xor(qsum, 16)`; `Q8ActSumTileBytes` doubles, and every allocation and
  the capacity check already scale through `Q8ActSumBytes`.
* Inner loop: `acc -= off_high * full + (off_low - off_high) * low` (the member
  `lo.offset` feeds the low half and `hi.offset` the high one).
* `IsNativeWmmaQuant`, the WMMA and small-batch dispatches, and the FP16
  prefill `DirectGemm` switch all gain Q2_K.
  `IsFusedSwiGluGemmEpilogueSupported` stays Q8_0-only.

Both sidecar writers were converted -- the shared-pass `StoreQ8ActLane` and the
fused SwiGLU epilogue in `BlockedSwiGluQuantEpilogue`, which the Q8_0 fused
kernel instantiates with `StoreActivationSum=true` -- and `ZeroQ8ActTailKernel`
now zeroes both halves. Every other format still reads the identical whole-block
sum from slot 0, so their results cannot move: integer addition is exact and
associative, so stopping the butterfly at sixteen lanes and adding the two
halves changes nothing.

Oracle coverage for Q2_K, all four routes: decode GEMV relative error 9.6e-08,
native prefill GEMM 6.9e-08 (small-batch 2.8e-08), FP16 prefill RMSE ratio 0.108,
and batch 1..8 bit-exact against the decode GEMV.

Measured with `gufo bench -p 2048 -n 128 -r 3`, 30 s between runs, two runs each:

| artifact | pp1024 | pp2048 | tg128 |
| --- | ---: | ---: | ---: |
| 3.84 bpw before | -- | 409.55 +/- 1.81 | 12.99 |
| 3.84 bpw after | 539.02 +/- 0.70 | 522.32 +/- 7.73 / 513.54 +/- 25.51 | 12.94 |
| `UD-Q4_K_S` reference, same session | -- | 562.38 +/- 9.85 / 547.91 +/- 45.63 | 13.17 |

The 3.84 bpw shard was the slow one only because that single Q2_K tensor removed
it from the chunked FP16 prefill path. Re-enumerating the predicate over the real
files now finds 503 inspected tensors and zero outside
`{Q8_0} union IsNativeWmmaQuant` on both shards, so both take it. That is +27%
pp2048 on the 3.84 bpw shard, and the route table in `gemm_route.hpp` already
declared Q2_K `hip_prefill_direct`; only the kernel dispatch was missing.

End-to-end: `gufo bench --validate-prefill 2048` on the 3.84 bpw shard returns
`top1_match=yes finite=yes`, cosine 0.99999923, max abs diff 0.0111 against the
token-at-a-time reference. The FP16 route is also the more accurate one by
construction -- the FP16-prefill oracle requires the FP16 error to
come in under a quarter of the A8 error, and Q2_K measures a ratio of 0.108.

Residual: the 3.84 bpw shard trailed the reference by about 7% at pp2048. The
shapes are identical and the shard is smaller (12.18 vs 14.30 GiB), so the
remainder has to be per-format cost in the FP16 prefill kernel -- its FFN
matrices are predominantly IQ3_S/IQ3_XXS grid codebooks where the reference is
IQ4_XS / Q4_K / Q5_K affine nibbles. Measured, attributed, and partly closed
below.

### Closing the residual: the fetch schedule, not the decode

Built `tools/qwen27b/prefill_fp16_bench.hip`, a per-format bench of the FP16
prefill kernel `HalfPrefillGemmKernel` (the one `LaunchBatchedQuantGEMMFp16`
dispatches, so the one the chunked path actually runs).
`tools/bench/build.sh` now links the host quant TU for it. At the production
FFN shapes, batch 2048, ms per GEMM:

| format | gate/up m=17408 k=5120 | down m=5120 k=17408 |
| --- | ---: | ---: |
| Q4_K | 11.53 | 11.37 |
| Q5_K | 11.60 | 11.18 |
| IQ4_NL | 11.74 | 11.57 |
| IQ4_XS | 11.85 | 10.77 |
| Q8_0 | 12.03 | 11.58 |
| Q6_K | 12.49 | 11.81 |
| Q3_K | 12.54 | 12.14 |
| Q2_K | 12.74 | 12.14 |
| IQ3_S | 13.14 | 12.39 |
| IQ3_XXS | 13.31 | 12.78 |
| IQ2_XXS | 13.38 | 12.97 |
| IQ2_XS | 13.49 | 12.99 |
| IQ2_S | 13.62 | 13.02 |

Weighting those by each shard's real per-layer mix gives 38.34 ms for the
3.84 bpw layer against 35.60 ms for the reference, +7.7% -- the measured
model-level gap, so the residual is entirely inside this kernel.

Two candidates were tested and refuted:

* Divergent grid-table loads. Filling the weight payload with one constant byte
  so every `kDeviceIq3sGrid` / `kDeviceIq3XxsGrid` / sign index collapses to a
  single entry changes nothing (IQ3_S 13.09 -> 13.10 ms, IQ3_XXS 13.31 -> 13.47)
  while Q4_K moves 2%. The 1-2 KB tables are not the cost.
* FP32 op count in the generic fetch. `DecodeQuantSub16` leaves `offset` at zero
  for every sub16 format except Q4_K/Q5_K/Q2_K, so the subtract can be compiled
  out exactly. Building two bench binaries and interleaving them showed no
  change at all -- every delta inside the 2% noise band, including Q4_K, which
  the elision cannot affect. Reverted; the change was not kept.

What did move it is the instruction-scheduling hint the IQ4 decoders already
carry. `__builtin_amdgcn_iglp_opt(0)` after the K loop interleaves the LDS
reads with the WMMA, which the immediate-decoding formats need and the deferred
`LoadRaw` formats (Q4_K/Q5_K/Q6_K/Q8_0) do not. The condition was extended from
IQ4 only to the whole immediate-decode set (Q3_K, Q2_K, IQ3_S, IQ3_XXS, IQ2_XS,
IQ2_S, IQ2_XXS) at the 256x256 tile. Paired against the previous binary,
back-to-back so only the deltas read:

| format | before ms | after ms | delta |
| --- | ---: | ---: | ---: |
| Q2_K | 13.15 | 12.35 | -6.1% |
| IQ3_S | 14.47 | 13.72 | -5.2% |
| IQ2_S | 14.38 | 13.74 | -4.4% |
| IQ2_XS | 13.71 | 13.14 | -4.1% |
| IQ2_XXS | 14.37 | 13.86 | -3.6% |
| Q3_K | 12.77 | 12.60 | -1.3% |
| Q4_K (hint not applied either side) | 11.97 | 11.97 | -0.0% |

`__builtin_amdgcn_iglp_opt` only reorders instructions, so results are exact;
the oracle suite is green at 269 checks with it in place. Projected per layer:
3.84 bpw 38.34 -> 36.81 ms (-4.0%), reference 35.60 -> 35.35 ms (-0.7%).

Measured on the box, interleaved with 30 s between runs, two runs each:

| artifact | pp1024 | pp2048 |
| --- | ---: | ---: |
| 3.84 bpw, before Q2_K native | -- | 409.55 +/- 1.81 |
| 3.84 bpw, Q2_K native | 539.02 +/- 0.70 | 522.32 +/- 7.73 |
| 3.84 bpw, + fetch-schedule hint | 548.29 / 556.65 | 540.33 / 531.07 |
| reference `UD-Q4_K_S`, same session | 587.65 / 586.04 | 542.45 / 561.84 |

That is +31% pp2048 on the 3.84 bpw shard (409.6 -> ~536) and a gap of about 3%
to the reference, from 26%. The reference barely moves because its FFN is
already IQ4_XS/Q4_K/Q5_K. The remaining spread is inside the kernel again, but
it is now small enough that the next lever worth pulling is the kernel's own
58%-of-peak efficiency, not the per-format decoders.

### Kernel efficiency: the ceiling is 48 TFLOPS, and the tile space is exhausted

The fp16 ceiling had been assumed from the int8 figure. Measured properly with
a wall-clock harness (`wmma_f32_16x16x16_f16_w32`, eight live accumulator chains
so the compiler cannot delete six of them -- a first attempt read only two and
reported 4x too high):

| instruction | TMAC/s | TFLOPS |
| --- | ---: | ---: |
| WMMA fp16 | 24.17 | 48.35 |
| WMMA bf16 | 23.73 | 47.47 |
| WMMA iu8 | 25.15 | 50.31 |

So 48.3 TFLOPS is the fp16 ceiling and the prefill kernel at ~31.7 TFLOPS sits
at 66% of it, not the 58% of a 55 TFLOPS guess. The int4 rate really is 2x --
52.6 TMAC/s at the same 16x16x16 shape -- but that path needs four-bit
activations and is therefore a quality decision, not a kernel one.

The tile space was then swept by launching `HalfPrefillGemmKernel` directly on
candidate (BM, BN, WM, WN, Complete), each verified bit-wise against the
production launcher (mismatch 0 everywhere), gate/up m=17408 k=5120 batch 2048,
ms per GEMM:

| variant | Q4_K | Q6_K | IQ4_XS | IQ3_XXS |
| --- | ---: | ---: | ---: | ---: |
| 256x256 w8n4 (production) | 12.68 | 11.92 | 11.28 | 12.12 |
| 256x256 w4n8 | 12.89 | 11.93 | 11.78 | 11.96 |
| 256x256 w4n4 | 12.00 | 11.80 | 11.85 | 12.90 |
| 256x256 w16n2 | 14.33 | 13.73 | 12.28 | 12.77 |
| 256x192 w8n4 | 14.54 | 13.65 | 13.61 | 13.37 |
| 128x256 w4n8 | 14.82 | 14.46 | 14.37 | 13.85 |
| 256x128 w8n4 | 15.85 | 15.89 | 15.80 | 15.14 |
| 128x128 w4n4 | 16.05 | 17.91 | 15.11 | 14.74 |
| 256x256 w8n4 Complete | 12.13 | 11.96 | 13.02 | 12.24 |

Only the 256x256 family is competitive; every narrower tile is 15-40% slower.
BK is fixed at 4 and the two stages already fill the 64 KiB LDS budget, so a
smaller tile stages proportionally less work per byte and the occupancy it buys
back does not repay it. `Complete` (drop the tail predicates) is worth 4% on
Q4_K, neutral on Q6_K/IQ3_XXS, and costs 15% on IQ4_XS -- which is exactly why
`DirectGemm` gates it on `LargeKvTile` and leaves IQ4_XS on the general kernel.
Production is already at the optimum of this space.

The remaining 34% is also not occupancy that can be bought: LDS is exactly full
at 64 KiB per block, so there is one block per CU, and 1024 threads is the
hardware workgroup limit. The block supplies 8 waves per SIMD and that is what
the pipeline gets. Closing the rest needs a different inner loop -- wider LDS
reads, or a wave64 variant -- which is a new kernel rather than a parameter.

### wave64 is not a lever for fp16

The obvious next structure was a wave64 fp16 kernel, mirroring
`prefill_quant_wave64.hip` on the int8 side. Measured the ceiling first, since
that decides it. `wmma_f32_16x16x16_f16_w64` takes the same operands as the w32
form (16 + 16 halves) with a v4f accumulator instead of v8f:

| form | TMAC/s | TFLOPS |
| --- | ---: | ---: |
| wmma f16 w32 | 24.17 / 24.41 | 48.35 / 48.82 |
| wmma f16 w64 | 24.25 / 24.32 | 48.50 / 48.64 |

Identical. The w64 instruction computes the same 4096 MACs and the SIMD issues
it over two cycles, so it buys half the accumulator registers and no throughput.
Together with the tile sweep (occupancy is not what is binding) and the int8
evidence -- the int8 path, wave64 where it qualifies, still measures 14.8-17.5
ms per FFN GEMM against 11.5-13.0 for the fp16 w32 path -- a wave64 fp16 prefill
kernel is not worth building: it can only reach the ceiling the current kernel is
already closest to. That closes the fp16 efficiency question at 66% of a 48.3
TFLOPS ceiling.

What is left is the int4 rate: 52.6 TMAC/s at the same 16x16x16 shape, 2.1x
int8 and 2.2x fp16, and the only measured lever above 1.5x anywhere in this
kernel. It is a quality decision -- both operands must be four-bit -- and it is
eligibility-bound on the current shards (Q4_K and Q3_K only: 25.7% of
UD-Q4_K_S, 15.6% of the 3.84 bpw one), so a low-double-digit percent of prefill
on these artifacts. One caveat before building on it: all-ones operands give
c[0] = 16 for both iu4 and iu8, which is consistent with both consuming K = 16
per output element, but the remaining fragments of the iu4 accumulator did not
read back as clean sixteens, so its fragment mapping deserves its own check.

### Kernel efficiency: ablating the real kernel

Ablations on `HalfPrefillGemmKernel<256,256,8,4>` at gate/up m=17408 k=5120
batch 2048, back-to-back in one session, ms per GEMM, base repeated so the
drift is visible:

| ablation | Q4_K | IQ3_XXS |
| --- | ---: | ---: |
| base | 11.06 / 12.12 | 12.90 |
| staging removed, valid LDS | 10.45 / 10.45 | 10.04 |
| store epilogue removed | 11.74 / 11.97 | 11.42 / 12.70 |

So the weight + activation staging costs about 10% on Q4_K and 22% on IQ3_XXS,
and the epilogue costs nothing measurable. Ruled out by measurement along the
way: grid-table divergence, decode FP32 op count, operand-register sharing
(48.2 TFLOPS with fully distinct operands against 48.4 with shared ones), and
the whole tile space. The MMA ceiling is 48.3 TFLOPS on wave32 and 48.5 on
wave64, so the flat kernel is at 64-68% of it.

That leaves the K loop itself. With staging removed the kernel runs at 34.9
TFLOPS, 72% of the ceiling, and what is left is the six swizzled LDS reads plus
eight WMMA per K step, the loop overhead across forty stages, and the barriers.
A reduced microbenchmark of exactly that sequence was built and measured 18.6
TFLOPS -- *slower* than the real kernel -- so it was not a faithful model and
its number cannot be used as a reference. The remaining 28% is localised to the
K loop but not yet explained.

**Retraction.** An earlier pass at the staging ablation removed the fetch
without first putting valid data in LDS. It reported 7.35 ms, 49.7 TFLOPS,
above the measured ceiling, and was briefly read as "the staging is the entire
deficit". Invalid operands are not free: that number is an artefact, the
valid-data rerun above supersedes it, and the strong claim is withdrawn.

Two harnesses in this investigation also produced confident wrong numbers and
are worth recording. A peak harness that read only two of eight accumulator
chains let the compiler delete six MMAs and reported 4x the real rate; a
reduced inner-loop probe whose LDS was never written had the arrays optimised
away entirely (`LDS Size: 2 bytes`) and measured an empty loop. Both looked
plausible. Ceiling and ablation numbers here are trusted only when the resource
report and the operand values both check out.

### Hardware counters: the load path is the loaded unit

`rocprofv3` counter collection works on this platform even though its kernel
timestamps do not. That is worth knowing: the earlier "no profiling possible"
note applies to timing/tracing, not to PMCs. For the Q4_K 256x256 kernel at
gate/up m=17408 k=5120 batch 2048, per dispatch (19 launches aggregated):

| counter | value | reading |
| --- | ---: | --- |
| `SQC_LDS_BANK_CONFLICT` | 26 total (max 495) | zero -- LDS conflicts are ruled out for good |
| `SQ_INSTS_LDS` | 7.24e7 | 12 LDS per K step per wave, exactly the design |
| `SQ_INSTS_VALU` | 1.77e8 | 44.6e6 are MMA; 132e6 are address/loop arithmetic |
| `SQ_INSTS_FLAT` | 1.09e7 | the global load instructions |
| `TA_TA_BUSY` | 1.87e7 of 2.35e7 cycles | L1/texture path busy 80% of the runtime |
| `GL2C_HIT` / `GL2C_MISS` | 2.08e7 / 5.42e6 | 79% L2 hit rate |
| L2 miss traffic | 347 MB per dispatch | against a 60 MB irreducible minimum |

The traffic is the finding. The weight matrix is 50.1 MB, so reading it once per
token block is 401 MB of requests. Each token block's activation tile is 1.31 MB
and is re-read once per row block, 68 times, so activations request
8 x 68 x 1.31 = 713 MB. Against that, unique data is only
50.1 + 8 x 1.31 = 60.6 MB. The measured 347 MB of misses sits just under the
weight stream, so it is the weights that mostly miss while the activations mostly
hit -- which is backwards from the intent: the grid is x-fastest, so the eight
token blocks of one row tile run consecutively and that row tile's 720 KB of
weights should stay resident in the 32 MB MALL. It evidently does not; the eight
blocks stream the same lines in lockstep and the tile is re-fetched each time.

What the counters do not give: a stall reason. This ASIC exposes no
`SQ_WAIT_*` counters, so the division of the remaining gap between memory
latency and the matrix pipe cannot be read off directly. What can be said is
that L1/TA is the busiest unit (80%), the matrix pipe is not saturated (the
kernel is at 64-68% of a ceiling that a pure-MMA harness reaches), and the LDS
and VALU instruction counts match the design exactly.

The next candidate is therefore cross-block reuse rather than the inner loop:
either an order that keeps a row tile's weights resident across its token
blocks, or a two-level blocking that keeps both streams resident. Cutting
347 MB toward 60 MB is worth roughly 5.8x less DRAM traffic; whether it
converts to time depends on how much of the gap is latency, which the missing
stall counters leave open. Worth a prototype before believing it.

### The L2 residency lever does not exist either

The kernel already swizzles the block grid so that a group of row tiles walks
all of its token blocks back to back, and `DirectGemm` already tunes that group
width per format (IQ4_XS uses 32 rows, Q5_K 32 for small K, everything else 8).
Since the counters reported 347 MB of L2 misses against a 60 MB unique working
set, widening that group was the obvious lever. Made it a runtime parameter and
swept 2..64 rows on the gate/up shape, ms per GEMM:

| group_shift | rows | Q4_K | IQ3_XXS | IQ4_XS |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2 | 10.19 | 11.45 | 10.08 |
| 2 | 4 | 10.30 | 11.03 | 10.30 |
| 3 | 8 (current) | 10.27 | 10.87 | 10.04 |
| 4 | 16 | 10.26 | 10.93 | 10.02 |
| 5 | 32 | 10.28 | 11.11 | 10.00 |
| 6 | 64 | 10.27 | 10.99 | 9.93 |

Flat: 1% on Q4_K and 5% on IQ3_XXS with no monotone trend. The reuse schedule
is not what limits the kernel, which also fits the bandwidth arithmetic --
347 MB over 23.5M GPU cycles is 43 GB/s, roughly 17% of the box DRAM peak. The
deficit is latency, not traffic.

That was the last structural hypothesis. The fp16 prefill kernel stands at 66%
of a 48.3 TFLOPS ceiling with every lever tried and measured: tile geometry,
LDS bank conflicts, decode work, operand-register reuse, wave64, the store
epilogue, L2 residency, DRAM traffic, and the row-group schedule. The only unit
reading high is the L1/TA path at 80% busy, and the staging ablation puts
10-22% of the runtime there, but nothing that keeps the current structure
recovers it.

Tally for this stretch of work: the Q2_K fix took the 3.84 bpw shard from 409.6
to ~522 t/s (+27%), and the fetch scheduling hint took it to ~536 (+2.6%). The
kernel-efficiency investigation added nothing further, and is recorded here so
that the next attempt starts from what has already been ruled out rather than
from the tile geometry.

### Isolating the 34%: the load path starves the matrix pipe

Four more experiments, all on the real kernel or on the pure-MMA harness it is
measured against:

1. **Occupancy is not it.** The MMA rate is flat across the whole occupancy
   curve: 23.7 TMAC/s with no LDS and high occupancy, 24.8 at 64 KiB per block
   with 256 threads (one block per CU, eight waves per SIMD), 25.7 at 64 KiB
   with 128 threads (four waves per SIMD). Even four resident waves sustain the
   ceiling, so the kernel.s eight are not the constraint.
2. **The LDS operand reads are not it.** Collapsing the K loop.s six swizzled
   loads to one shared set -- a quarter of the LDS traffic, same MMA count --
   buys 5% on Q4_K and nothing on IQ3_XXS.
3. **The barriers are inconclusive.** Removing them, with or without the
   staging, changes the operand data (the output goes non-finite), and operand
   data measurably changes the rate -- see the caveat below. Those runs cannot
   be read.
4. **The load path is the differentiator.** Profiling the pure-MMA harness with
   the same counters:

| kernel | TA_TA_BUSY / GRBM | matrix pipe |
| --- | ---: | ---: |
| pure-MMA harness (256 thr, 64 KiB LDS) | 0.3% | ~100% of ceiling |
| pure-MMA harness (no LDS, high occ) | 3.7% | ~100% of ceiling |
| `HalfPrefillGemmKernel` Q4_K | 80% | 66% |

That is the cleanest statement available: the only unit anywhere near
saturation is the L1/texture load path at 80%, and the matrix pipe sits at 66%
behind it. The kernel is starved by its own weight and activation loads, which
also fits the staging ablation (10-22%) and the earlier finding that the two
halves cost roughly independently (weights 1.6 ms, activations 2.7 ms on
Q4_K). What is *not* established is why the TA is at 80% when the traffic is
43 GB/s of DRAM and about 1.7 GB across L2 -- both far from any bandwidth
limit. Answering that needs L1/TA sub-unit counters (miss handling, MSHR
occupancy, per-wave stalls on memory) that this ASIC does not expose, and
`SQ_WAIT_*` is absent from the counter list entirely.

**Caveat discovered here.** Operand *values* change the measured MMA rate: an
ablation that drops the staging without first staging valid data measures
7.35 ms, while the same ablation with valid LDS data measures 10.45 ms. The
harness with garbage operands runs about 40% faster than with real ones. Every
ablation that changes correctness therefore has a data-dependent bias, which is
what makes experiment 3 unreadable. Treat any correctness-breaking ablation here
as usable only for direction, never for magnitude.

### Instrumentation inventory, and the end of the attribution

What this box actually exposes, verified by trying each one:

| capability | status |
| --- | --- |
| PMC counters | ~37 in the gfx1151 flat list: instruction counts, `SQ_WAVE_CYCLES`, occupancy, `SQC_LDS_BANK_CONFLICT`, L2 hit/miss, `TA_BUSY_avr`, `MemUnitBusy`, `GPUBusy` |
| PC sampling | **no agents support it** (`rocprofv3-avail list --pc-sampling` is empty) |
| `SQ_WAIT_*` stall counters | not defined for gfx1151; the SDK lists them for other archs, gfx1151 reports `Missing` |
| `SQ_VMEM_TA_CMD_FIFO_FULL`, `SQ_VMEM_TA_ADDR_FIFO_FULL`, `SQ_INSTS_VMEM*` | not available |
| `TA_BUFFER_COALESCEABLE_WAVEFRONTS`, `TA_ADDR_STALLED_BY_TC_CYCLES`, `GL2C_EA_RDREQ_*` | not available |
| ATT perfcounters | gfx9 only |
| kernel timestamps | broken (start == end), so no per-kernel duration either |

So the top-down method stops here: it can establish that no unit is saturated,
but not which latency is binding, because every stall-reason counter is absent.

**Correction.** `TA_TA_BUSY` at 80% of `GRBM_COUNT` was read above as the load
path being saturated. It is an aggregate; the per-unit average is
`TA_BUSY_avr` = 4.61e6 of 23.46e6 cycles = **20%**, and `MemUnitBusy` = 21.5.
So the load path is not saturated either. The honest reading of the counters is
that *nothing* is saturated -- matrix pipe 66%, TA ~20%, memory unit ~22%,
issue ~14%, DRAM ~17% of peak, LDS conflicts zero -- while a pure-MMA loop with
the same 8 accumulator chains, the same ISA and the same occupancy reaches 100%.

The load-path conclusion is therefore withdrawn too: it was based on the
aggregate misread.

### Last lever tried: software pipelining the LDS operand loads

The remaining structural reading was that each K step loads its six operands and
immediately consumes them, so the LDS latency sits in front of the MMA group.
Restructured the K loop to issue step i+1's loads before step i's MMAs, same
instructions, same arithmetic:

| | Q4_K | IQ3_XXS |
| --- | ---: | ---: |
| base | 11.47 / 11.45 | 12.34 / 12.72 |
| pipelined | 11.92 / 11.82 | 12.31 / 12.36 |

Neutral on IQ3_XXS and 3% worse on Q4_K, with VGPRs up from 136 to 160. Reverted.
Either the compiler already schedules this, or the register cost cancels the
benefit.

The state after all of it: a reproducible 34% deficit that no unit attribution
explains and no restructuring recovers. Everything tried is listed above -- tile
geometry, LDS conflicts, decode cost, operand reuse, occupancy, wave64, the
store epilogue, L2 residency and row-group width, LDS read volume, manual
pipelining, and the staging ablation. The staging is worth 10-22% but no change
that keeps the kernel correct has collected it.

What is left, in order of expected value: static ISA cycle-accounting of one
stage body read straight out of the disassembly (no profiler needed, and the
only route that can show an unexpected serialisation the counters cannot see);
or accept 66% and change the instruction instead (int4, 2.2x, a quality
decision).

### ISA cycle-accounting: attempted, inconclusive

Dumped device-only ISA for the exact production instantiations (Q4_K 256x256,
kStore, Complete false and true) and tried to account one stage body by hand.
Two problems stopped it:

1. The compiler fully restructures the stage. Searching for barrier-separated
   regions containing exactly 32 WMMAs -- the stage body MMA count -- finds
   spans of 6500+ instructions, over 130 instructions per MMA, against the
   counter-derived 4 VALU per MMA. Those spans are not stage bodies, so no
   accounting from them is meaningful. Discarded rather than reported.
2. `GRBM_COUNT` is not the dispatch duration either. It reads 23.46M cycles
   (8.09 ms at 2.9 GHz) while the kernel measures 11.5-12 ms, about 34M cycles.
   Every "% of GRBM" figure above is therefore a lower bound by about 1.45x.
   Corrected: matrix pipe ~64%, TA ~14%, issue ~10%. Nothing saturated -- the
   conclusion is unchanged, only the numbers move.

What the ISA did show, and is checkable: `s_delay_alu` is **15.6% of the static
instruction stream** (35819 of 230132), and 21% inside the stage-like span.
That is RDNA3.s dependency-padding instruction. If those occupy issue slots they
are a fifth of the issue budget; if the hardware treats them as free, they mean
the scheduler could not otherwise separate dependent ALU operations. Either
reading points at dependency latency in the address and loop arithmetic rather
than at the memory system -- which would also explain why every memory-side
hypothesis failed. Not enough to act on, and not confirmed.

**Honest close: the cause of the 34% is not established.** The instrumentation
surface is exhausted (no PC sampling, no stall counters, no TA sub-counters, no
usable timestamps), and every structural lever tried is neutral or negative.
The strongest facts remain that a pure-MMA loop with the same accumulator count,
ISA and occupancy reaches 100% of the 48.3 TFLOPS ceiling, while the real kernel
reaches 64-66% with no unit saturated. Acting on that needs stall data this ASIC
does not expose.

### In-kernel clock instrumentation: the phase breakdown

Since the ASIC exposes no stall counters and rocprofv3 cannot sample PCs, the
kernel is instrumented directly. `prefill_fp16.hip` under
`-DGUFO_FP16_PROTO_CLOCK` wraps each stage phase in `clock64()` and accumulates
four per-block totals into a `__device__` array, which the host reads with
`hipMemcpyFromSymbol`. Zero cost when the macro is off (verified: default build
unaffected). The scratch driver is ~150 lines and lives outside the repo.

Block (0,0) of Q4_K at gate/up m=17408 k=5120 batch 2048, cycles and share of
the estimated block lifetime (wall time / 28 sequential blocks per CU):

| phase | cycles | share |
| --- | ---: | ---: |
| `commit()` (registers to LDS) | 65,846 | 6.4% |
| `__syncthreads()` x2 | 75,590 | 7.4% |
| `fetch()` (global loads + decode) | 24,339 | 2.4% |
| K loop (LDS reads + 8 WMMA per step) | ~474k-639k | 46-62% |
| **unaccounted** | **~380k** | **~37%** |

The unaccounted third is the interesting part, and it is not the epilogue: the
store ablation already showed that costs nothing measurable. The candidates are
block launch and drain, the partial last round of the grid (544 blocks over 20
CUs leaves 4 blocks in round 28 with 16 CUs idle), LDS init at block start, and
the initial `fetch(0)` plus its `commit()`. Twenty-eight rounds x per-block
fixed overhead would land exactly here.

That is a different lead from everything tried so far: it says a large part of
the deficit may not be in the kernel body at all, but in how the grid executes.
It is directly testable with the same instrument -- record `clock64()` at the
top of the kernel and at the end of the epilogue, and compare one block.s true
duration against wall_time / rounds. If the true block duration is well under
that quotient, the gap is inter-block overhead and the fix is grid shaping
(fewer, longer-lived blocks) rather than anything inside the K loop.

Instrument caveat: one of the four accumulators comes back wrapped in some runs,
which is `clock64()` reads being reordered by the compiler rather than a real
negative duration. The reads need a compiler barrier between them before the
per-phase numbers are trustworthy to better than a factor check. The shares
above reproduce across runs for the phases that do not wrap.

And the K loop itself, measured directly here, is the same story as the counters:
it is 46-62% of the block and it runs at roughly 63% of the MMA rate its own
instruction stream should sustain. So the deficit is still not explained -- but
it is now located to a specific phase with a working instrument, and the
unaccounted third is a concrete, testable alternative to the K loop.

### The cause: the K loop is optimal, the block overheads are not overlapped

The clock instrument settles it. Q4_K, gate/up, m=17408 k=5120 batch 2048,
544 blocks over 28 rounds of 20 CUs:

| | cycles | share of block |
| --- | ---: | ---: |
| `commit()` | — | 7.3% |
| `__syncthreads()` x2 | — | 6.5% |
| `fetch()` | — | 2.9% |
| **K loop** | 648,295 | **70.4%** |
| initial fetch + epilogue | — | 12.9% |
| true block | 920,230 | 100% |
| inter-block + tail gap | — | 14.0% of wall |

The decisive arithmetic, all on wall-clock-comparable quantities:

* MACs per block = 256 x 256 x 5120 = 335.5M; x 544 blocks = 1.825e11 MACs.
* K loop share of wall = 0.704 x 10.33 ms = 7.27 ms.
* K loop rate = 1.825e11 / 7.27e-3 = **25.1 TMAC/s**.
* Pure-MMA harness, wall clock, same 8 accumulator chains: 24.2-26.2 TMAC/s.

**The K loop runs at the machine.s MMA peak.** It is not starved, not
latency-bound, not LDS-bound. Every counter and ablation that pointed at the
inner loop was pointing at something already at 100%.

The deficit is the other 30%:

* 16.7% is per-stage block work -- `commit()` 7.3%, two barriers 6.5%, `fetch()`
  2.9%. Forty stages of it, none of it overlapped with anything, because there
  is exactly one block per CU.
* 12.9% is the initial fetch plus the store epilogue.
* 14.0% is wall time that is not block execution at all: block launch and drain
  between 28 rounds, plus the last round where 544 = 27 x 20 + 4 leaves 16 CUs
  idle while 4 blocks finish.

That is the whole 34%, and it is consistent with every earlier measurement: the
tile sweep (smaller tiles lose more per-block reuse than they gain in overlap),
the flat occupancy curve (the MMA rate does not need more waves), the store
ablation (the epilogue instructions are not the cost -- they are the *unoverlapped*
time), and the LDS and decode ablations (the K loop has slack it does not need
because it is not the limit).

The fix is therefore overlap, not a faster inner loop:

1. **Persistent blocks.** One block per CU that loops over work items removes
   the launch/drain between rounds and lets one tile.s epilogue overlap the next
   tile.s fetch. That targets the 14% gap and part of the 12.9%.
2. **Fewer stages.** The 14% of commit+barrier is per stage; BK=8 would halve
   it, but BK=4 already fills the 64 KiB LDS budget at BM=BN=256, so this needs a
   tile that trades reuse for stages -- which the tile sweep says loses.
3. **The epilogue** is 12.9% and is the largest single non-K-loop item. It is not
   slow in itself; it is serial with everything else.

So: theoretically fixable, and the target is the ~30% of block time that is not
the K loop plus the 14% between blocks. The realistic route is a persistent-CTA
schedule, which is a real kernel change rather than a parameter, and is the first
direction since the Q2_K fix that has a measured mechanism behind it.

### Instrument reliability, honestly

The clock instrument is good enough to have found the cause and not good enough
to leave running unattended. `clock64()` is a pure function as far as the
compiler is concerned, so in some instantiations it is commoned up or moved
across the loop and an accumulator comes back wrapped -- true_block smaller than
a phase inside it. The reliable readings reproduced across runs (Q4_K true_block
920,230 with K loop 70.4%, IQ4_XS 913,366 with K loop 69.1%), which is what the
conclusion rests on. The broken ones are obvious when they happen (any share
over 100%).

Hardening it properly needs reads the compiler cannot fold. Neither
`s_memtime` nor `s_memrealtime` assembles for gfx1151 with this toolchain
("instruction not supported on this GPU"), so the options are a `volatile`
accumulator, a per-read data dependency, or reading through an
`s_waitcnt`-separated sequence. Until then, treat any single reading as a
sanity check and require a repeat before believing a number.

### Correction: there is no inter-block gap

The 14% "inter-block and tail gap" reported above is an arithmetic error of
mine. I divided by the nominal 2.9 GHz, but the clock the instrument reads is the
one the part actually runs at under load, about 2.5 GHz. Check it: 28 rounds x
920,230 ticks = 25.77M cycles; the wall time was 10.33 ms; 25.77e6 / 10.33e-3 =
2.49 GHz. At that frequency the block time times the round count accounts for the
wall time exactly, so every cycle is inside a block and there is no gap to
recover.

That makes the whole picture self-consistent instead of leaving a residual:

* block = 100% accounted -- K loop 70.4% and at the MMA peak, `commit` 7.3%,
  barriers 6.5%, `fetch` 2.9%, epilogue + init 12.9%;
* persistent CTAs lost 4% because there was no launch or tail overhead to remove,
  only loop overhead to add;
* 128x128 with two blocks per CU lost 27-50% because the overhead is throughput,
  not latency, so a second resident block cannot fill it.

The fp16 prefill deficit is now fully explained and none of it is a mystery:
70.4% is the K loop running at the machine MMA rate, and the other 29.6% is block
overhead that is required in kind -- LDS stores to stage the operands (7.3%), the
barriers that cooperative staging needs (6.5%), the global operand loads (2.9%),
and the output write plus the pre-loop setup (12.9%). The output alone is 143 MB
per GEMM, which at the measured epilogue rate is about 7% of the runtime.

The only piece with visible headroom is that output rate: 143 MB in ~1.33 ms is
108 GB/s where the box can do roughly double. A coalescing change could plausibly
recover a few percent. Everything else is pinned.

### The 128x128 comparison was unfair; the fair one agrees

I flagged that the tile sweep compared a fully tuned 256x256 against 128x128
variants denied every optimisation gated on the big tile: the iglp scheduling
hint (worth 4-6% on the grid types by direct measurement), the cached Q5/IQ block
headers, the row-group width override, and `Complete`. Lifted those gates for any
tile at least 128 wide and re-ran both tiles from the *same* weight and activation
buffers, interleaved, six repetitions each. The standard deviations were
0.04-0.06 ms, about 0.5%, so these are real differences and not box noise:

| type | 256x256 ms | 128x128 ms | 128 vs 256 |
| --- | ---: | ---: | ---: |
| Q4_K | 9.876 +/- 0.061 | 13.387 +/- 0.049 | +35.5% |
| Q6_K | 10.219 +/- 0.061 | 16.052 +/- 0.054 | +57.1% |
| IQ4_XS | 10.151 +/- 0.054 | 11.551 +/- 0.061 | +13.8% |
| IQ3_XXS | 11.045 +/- 0.041 | 12.150 +/- 0.045 | +10.0% |

So occupancy really is a dead end, and now the error bars are an order of
magnitude smaller than the effect. The reason is in the fragment arithmetic,
which had not been articulated before: LDS loads per MMA is
(WRS+WTS)/(WRS*WTS). At 256x256 with 1024 threads, WRS=2 and WTS=4, so 8 MMAs
cost 6 loads, or 0.75 loads per MMA. At 128x128 with 512 threads, WRS=WTS=2, so
4 MMAs cost 4 loads, or 1.0. The small tile needs a third more LDS traffic per
MMA *and* has four times as many blocks each running the same 80 stages, so its
per-stage overhead is quadrupled per unit of work. Two blocks per CU cannot pay
for either.

Side finding: the 128x128 Q6_K variant produces the wrong answer -- the entire
output mismatches -- while 256x256 agrees with the validated production path.
`DirectGemm` never selects 128x128, so it has never mattered, but it is a latent
bug in the small-tile Q6_K path. Recorded, not fixed.

### Corroborating the 70.4%: the peak is the arbiter

The clock instrument is one flaky instrument, so the K-loop share needed an
independent check. The direct ablation does not work: making the compute loop run
1..4 of its 4 K steps changes register pressure with the loop bound, so the
variant at span 1 is a different kernel, not the same kernel minus work. Measured
in separate binaries (one instantiation each; four in one binary thrashed the
instruction cache and doubled every number), four interleaved passes:

| K steps per stage | ms |
| ---: | ---: |
| 1 | 6.425 |
| 2 | 6.539 |
| 3 | 8.026 |
| 4 (production) | 9.968 |

Those marginal costs are wildly non-uniform -- 0.11 ms for the second step, 1.49
for the third, 1.94 for the fourth -- which is the confound, not the K loop. A
naive linear read would put the K loop at about 47% of the runtime.

That reading is impossible, and saying why is the corroboration. The K loop
performs every MAC by construction: 1.825e11 MACs. A 47% share of 9.97 ms is
4.69 ms, which is **38.9 TMAC/s** -- above the machine.s measured fp16 WMMA peak
of 24.2-26.2 TMAC/s, which was itself checked with eight live accumulator chains
and found flat across the whole occupancy curve. The K loop therefore cannot
occupy less than 1.825e11 / 26.2e12 = **6.97 ms, or 69.9%** of the runtime. The
clock instrument says 70.4%. They agree to within half a point, and the two
methods share no machinery.

So the K loop really is at the MMA peak: the constraint that it cannot exceed the
measured peak fixes its share at 70%, and at that share its rate is 26.0 TMAC/s,
right at the ceiling. The remaining 30% is the staging structure and the output
write, and the ablation.s shape adds one useful detail -- nearly half the K loop.s
cost lands on the *first* K step of each stage, which is the one that stalls on
the LDS operands issued after the barrier. The later steps pipeline behind it,
which is also why manual software pipelining of the operand loads changed
nothing.




### Persistent CTAs: bit-exact, and 4% slower

Implemented the persistent schedule -- 20 blocks, one per CU, each striding
over the 544 logical tiles with the existing group swizzle re-derived from the
item index. The output fingerprint is identical to the launched-grid baseline,
so the restructuring is correct. Clean interleaved A/B, no instrumentation, four
runs each:

| | ms per GEMM | TFLOPS |
| --- | --- | --- |
| baseline, 544 blocks | 9.90 / 9.99 / 9.93 / 10.01 | 36.5-36.9 |
| persistent, 20 blocks | 10.36 / 10.38 / 10.34 / 10.37 | 35.2 |

**4% slower, consistently.** So the ~14% residual is not launch and drain that a
persistent schedule can remove. The workgroup scheduler evidently overlaps block
transitions already, and dynamic issue beats static assignment: persistence pins
the ragged tail (8 CTAs take 28 items while 12 take 27) into the critical path,
where the scheduler would otherwise fill it. Reverted; the macro-guarded
scaffolding is gone from the tree.

That was the last idea with a measured mechanism behind it, and it failed. What
remains is the decomposition: K loop 70.4% and at the MMA peak, `commit` 7.3%,
barriers 6.5%, `fetch` 2.9%, epilogue and init 12.9%, and a residual that is
neither launch nor tail. Overlapping the block-level 30% needs two blocks per
CU, which needs 32 KiB of LDS each; the only routes there are BK=2, which doubles
the stage count and gives back exactly what it saves, or a 128x128 tile, measured
15-40% worse. So the fp16 prefill kernel is at its practical limit for this
structure -- 66% of the pure-MMA ceiling, and the ceiling is what its K loop
already achieves.

### The last structural idea is blocked by the ISA, verified

Producer/consumer warp specialisation was the one remaining idea that attacks the
actual constraint rather than a resource that is not full. The matrix pipe
saturates at about four waves per SIMD (measured flat across the occupancy
curve), so four of the eight could be repurposed to staging while the other four
do nothing but MMAs against a double-buffered stage and never touch a full-block
barrier. Issue is 86% idle, so the staging instructions would fit.

It needs a way to barrier a *subset* of the workgroup. Assembled every candidate
for gfx1151:

| instruction | result |
| --- | --- |
| `s_barrier` | assembles |
| `__builtin_amdgcn_wave_barrier()` | assembles (wave-level only) |
| `s_barrier_signal <n>` | **instruction not supported on this GPU** |
| `s_barrier_wait <n>` | not supported |
| `s_barrier_signal -1`, `s_barrier_wait -1` | not supported |
| `s_barrier_init` | not supported |
| `s_barrier_signal_isfirst` | not supported |

So gfx1151 exposes only the workgroup-wide barrier and the wave barrier. Named
or partial barriers are CDNA (and gfx12), not this part. The idea is dead at the
ISA level.

That closes the structure question, and the chain is worth stating in one place
because no single link is the whole answer:

1. staging must be shared -- private per-wave staging needs about 384 KB of LDS
   against the 64 KB available, because the reuse that makes the GEMM fast *is*
   the sharing;
2. sharing needs cross-wave synchronisation;
3. the only cross-wave primitive is a full-workgroup barrier;
4. so all 80 stage boundaries are full-block barriers and the matrix pipe drains
   at each one;
5. the drain can only be hidden by a second resident block, which needs 32 KiB of
   LDS and half the accumulator registers;
6. and the tile that fits that, 128x128, needs a third more LDS loads per MMA, so
   it loses more than it hides -- measured at 10-57% worse with the gates lifted.

**The binding constraint is the interaction of an ISA synchronisation primitive
with LDS capacity -- not FLOPs.** That resolves the apparent paradox: the matrix
pipe is at 70% utilisation, not 100%, and nothing is rate-limited. The machine
has plenty of multiply throughput; what it lacks is a way to overlap the staging
that feeds it. The fp16 instruction.s 48-52 TFLOPS is its own ceiling, and the
same silicon does 105 TFLOPS on int4, which is the only door left.

### LDS efficiency, taken to the end: the last axis is BK

The open question was whether LDS *use* could be made more efficient, as opposed
to merely being under-utilised. Every axis except one was already pinned:

| axis | state | evidence |
| --- | --- | --- |
| footprint | exact, no padding | `s_a`/`s_b` are precisely `BM*BK*16`/`BN*BK*16` halves |
| bandwidth | about 6% of the port | 5.2 MB written per block; 0.75 reads per MMA |
| transaction width | maximal | `ds_read_b128`/`ds_write_b128` throughout |
| bank conflicts | zero | `SQC_LDS_BANK_CONFLICT` ~ 0; constant-payload test flat |
| fragment ratio | 0.75 loads per MMA | `(WRS+WTS)/(WRS*WTS)` at `(2,4)` |

The one axis never swept was **BK**, which had been hardcoded to 4. It is an LDS
axis: `BK*(BM+BN)*32 = 64 KiB` exactly at 256x256, so `BK` is what the 64 KiB
buys. Making it a template parameter (`BKDepth`, defaulted to 4, so production is
bit-identical) and sweeping it, Q4_K, M=17408 K=5120 batch=2048, three passes:

| variant | tile | BK | VGPRs | spill | LDS | blocks/CU | ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 256x256 w8n4 | 4 | 192 | 0 | 65536 | 1 | **12.22** |
| 1 | 128x256 w4n4 | 2 | 191 | 0 | 24576 | 2 | 14.14 |
| 2 | 128x256 w4n4 | 4 | 256 | 4 | 49152 | 1 | 13.90 |
| 3 | 128x256 w4n8 | 2 | 135 | 0 | 24576 | 2 | 18.77 |
| 4 | 256x256 w8n4 | 2 | 191 | 0 | 32768 | 1 | 14.30 |

All five are numerically exact (`mismatch=0`), so halving the stage depth is
correct; it is simply slower.

Three readings, and the first is the one that matters:

1. **Variant 4 vs 0 isolates the stage count.** Same tile, same 1024 threads,
   same VGPR count, same fragment ratio, same one block per CU; the only
   difference is 160 stages instead of 80. **Halving BK costs 17%.** Fitting
   `T = T_k + N*c` to the two points gives `c ~ 26 us` per 80 stages, so the
   per-stage overhead is real and large. Extrapolating the same line, `BK=8`
   (40 stages) would be about 8.5% faster -- which is the price the 64 KiB LDS
   cap is charging, and it is an estimate, not a measurement, because LDS cannot
   hold a `BK=8` stage at this tile.
2. **The second resident block buys nothing.** Variant 1 is the configuration
   the earlier chain said would be needed to hide the drain: 24 KiB of LDS, two
   blocks per CU, 8 waves per SIMD, *and the same 0.75 loads per MMA as
   production*. It still loses 16%. Compared against variant 2, which is the same
   tile and thread count at `BK=4` with one block per CU, the extra block is
   worth less than the doubled stage count costs: 14.14 against 13.90. Step 5 of
   the chain above is now measured, not argued -- the overlap does not pay for
   the extra stage boundaries, even without the fragment-ratio penalty that
   disqualified 128x128.
3. **The fragment ratio is a strong lever and 0.75 is its floor here.** Variant 3
   is exact and has the second block, and is 54% slower because it reads 1.0 LDS
   loads per MMA instead of 0.75. LDS reads are on the critical path after all.
   But at 1024 threads and a 256x256 tile there are exactly 8 MMAs per wave, so
   the feasible splits are `(1,8),(2,4),(4,2),(8,1)` and `0.75` is the minimum
   of that set. The `0.5` split needs 16 accumulators, 256 VGPRs, and spills --
   variant 2 is what that looks like.

So LDS use is not improvable on any axis, and the reason is not that LDS is
fast: it is that **BK is the parameter LDS capacity buys, and the kernel is
already spending all 64 KiB on the largest BK that fits.** The residual overhead
is rent on that ceiling.

### The second resident block, tested properly: it is worth about one percent

Step 5 above -- "the drain can only be hidden by a second resident block" --
implicitly assumed there *is* a drain to hide. That had never been tested with a
fair pair, because every two-block configuration found so far also paid either
the stage-count penalty (`BK=2`) or the fragment-ratio penalty (128x128 at 512
threads). Two configurations pay neither, and neither had been tried:

| | tile | threads | BK | loads/MMA | blocks/CU |
| --- | --- | --- | --- | --- | --- |
| baseline | 256x256 w8n4 | 1024 | 4 | 0.75 | 1 |
| A | 128x128 w4n2 | 256 | 4 | **0.75** | **2** |
| B | 128x128 w4n4 | 512 | 4 | 1.00 | **2** |

Both keep `BK=4` (80 stages), so nothing else changes. To get a control that
differs *only* in block residency, each was also run with 32 KiB of unused
dynamic shared memory passed to the launch, which pushes `32768 + 32768` to the
64 KiB cap and leaves room for exactly one block. The runtime's own occupancy
model confirms the control does what it claims:

```
blocks/CU 256x256 w8n4 bk4   thr=1024  dyn=0      -> 1
blocks/CU 128x128 w4n2 bk4   thr=256   dyn=0      -> 2
blocks/CU 128x128 w4n2 bk4   thr=256   dyn=32768  -> 1
blocks/CU 128x128 w4n4 bk4   thr=512   dyn=0      -> 2
blocks/CU 128x128 w4n4 bk4   thr=512   dyn=32768  -> 1
blocks/CU 128x256 w4n4 bk4   thr=512   dyn=0      -> 1
```

Q4_K, M=17408 K=5120 batch=2048, median of three passes:

| tile | 2 blocks/CU | 1 block/CU | value of the second block |
| --- | --- | --- | --- |
| 128x128 w4n2 (0.75 loads/MMA) | 12.88 ms | 13.03 ms | **1.2%** |
| 128x128 w4n4 (1.00 loads/MMA) | 15.25 ms | 15.17 ms | **-0.5%** |
| 256x256 w8n4 (baseline) | -- | 12.14 ms | -- |

The noise floor between two instantiations of the identical baseline is about
1.3% (measured separately), so a genuinely coscheduled second block buys
**nothing beyond the noise floor**. It is resident -- the occupancy model and the
LDS arithmetic both say so, and the timing moves when it is taken away -- but it
does not recover the 30%.

That is a real result, because it rules out the whole class of explanation the
structure argument rested on. If a fifth of the block's cycles were matrix-pipe
idle while it staged, decoded, waited at a barrier or wrote its epilogue, a
second block would be filling them and the speedup would be large. It is 1.2%.
**There is no idle pipe to fill**, so the 30% is not latency, not a drain, and
not something a second block, a persistent CTA or a partial barrier could have
recovered. The phases are disjoint in *program order within a block*, but the
matrix pipe is still retiring the MMAs issued before the barrier while the waves
are in the next staging region -- which is why the phase percentages do not sum
to a description of pipe occupancy.

The instruction mix says the same thing from the other side. The Q4_K 256x256
kStore kernel's K-loop body is 193 instructions carrying 32 WMMAs, and the
weight decode appears in the same loop *body* -- 50 of those instructions are the
`v_bfe`/`v_cvt`/`v_fma_mix` nibble unpacking. The whole function contains 48
`ds_load_b128` and 4 `ds_store_b128` in total. But being in the same body is not
the same as being overlapped: that body also contains two `s_barrier`
instructions, and the decode sits entirely on the `commit()` side of the first
one. See the buffering note below -- an earlier draft of this section claimed the
scheduler had interleaved the decode with the MMAs, which the barriers make
impossible. Per stage a SIMD issues 8 waves x 193 = 1544 instructions against
about 8000 cycles of matrix-pipe time, so roughly 80% of issue slots are spare:
the non-MMA instructions in the loop are free, and they are already overlapped.

**What that leaves is work removal, not re-overlap.** The only two sizeable
non-MMA blocks are the fp32 output store (about 7% of the block in the kStore
variant this was profiled on) and the quant decode (7.3%). The first is likely
already absent from the production path, which runs the fused SwiGLU/GateUp
epilogue and writes quantized activations rather than fp32 -- so the "66% of the
instruction ceiling" figure is pessimistic for production and should be
re-measured on the fused epilogue rather than estimated from kStore. The second
is the int4 decision: nothing else in the block is removable without changing
the numerics.

### The fused epilogue, and why there are two barriers per stage

The 12.9% epilogue figure above was measured on the `kStore` epilogue, which
writes fp32. Production's FFN runs the *dual* `kGateUp` kernel and writes fp16,
so the figure was suspect. Measured directly, same tile and shape, median of
three passes:

| variant | ms | vs kStore |
| --- | --- | --- |
| 256x256 w8n4 `kStore` (fp32 out) | 12.38 | -- |
| 256x256 w8n4 `kSwiGLU` (fp16 out, fp32 gate read) | 13.13 | **+6%** |
| 128x128 w4n2 `kStore` | 13.05 | +5% |
| 128x128 w4n2 `kSwiGLU` | 13.85 | +12% |
| **production dual `kGateUp` fused** (2 GEMMs per launch) | **22.84** | **--** |

The standalone fused epilogue is *slower*, not faster: `kSwiGLU` reads a fp32
gate (4 B) and writes fp16 (2 B), so 6 B/element against `kStore`'s 4 B of write.
The dual kernel is the one that wins, because it never reads a gate at all --
both operands are already in registers -- and writes 2 B. Per GEMM it does
22.84/2 = 11.42 ms against the single kStore GEMM's 12.38 ms, so about **8%
better per GEMM**, which is the activation staging being shared plus the halved
output.

That gives the number that matters. The dual kernel moves 2 x 1.825e11 MACs in
22.84 ms, so **31.96 TFLOPS, or 66.1% of the measured 48.35 fp16 ceiling**. The
"66% of ceiling" figure was therefore *not* pessimistic: it holds for the kernel
production actually runs, and the fused epilogue does not remove the overhead.
That hypothesis is refuted.

What the epilogue costing 6% does confirm is where the two barriers come from.
`commit()` writes `s_a[ks]` and `s_b[ks]` with **no parity dimension** -- the LDS
is single-buffered -- and `sizeof(s_a) + sizeof(s_b)` is exactly the 64 KiB
budget, so there is no room for a second buffer. The loop is therefore

```
commit()    decode + StoreSwizzled into s_a/s_b     7.3%   <- serial
barrier A   publish the single buffer                      \
fetch()     global loads to registers               2.9%   | 6.5% barriers
K loop      48 ds_load_b128 + 32 WMMA              70.4%   |
barrier B   nobody may overwrite until readers done        /
```

Barrier B exists *only* because the buffer is single, and barrier A is the
publish that a double buffer would let you overlap with compute. Both are paid
because 64 KiB cannot hold two stages. This is the same capacity wall as the BK
result, seen from the other side: at 256x256 the stage is exactly the LDS budget,
so single-buffering is forced, so both barriers are forced.

**The one structural lever that survives all of this** is that the decode half of
`commit()` is pure VALU reading registers, and issue is ~80% spare, so it does
not have to be inside the fenced serial window -- it could be pre-decoded into
registers during the K loop of the previous stage, leaving only `StoreSwizzled`
after barrier B. The register cost is bounded and small: one stage's worth is
`BM*BK/kThreads` = one 16-half sub-block for s_a and one for s_b, 16 VGPRs,
against the 64 VGPRs of headroom between the 192 in use and the 256 the
occupancy allows. The compiler cannot do this itself because `s_barrier` is a
scheduling fence. The ceiling on the gain is the decode's share of the 7.3%
commit -- a few percent of the block, not a third of it.

### The decode hoist: tried, bit-exact, and refuted

That lever was implemented and measured behind a `HoistDecode` template flag in
one binary, so both arms come from the same build. Q4_K 256x256 w8n4 bk4, M=17408
K=5120 batch=2048, median of three passes. Both arms are numerically exact
(`mismatch=0`), which is the point of choosing `DecodeRaw` rather than the
fetch-stage decoder: the arithmetic is untouched, only the placement moves.

| placement of the hoisted decode | VGPRs | spills | ms | vs commit-window decode |
| --- | --- | --- | --- | --- |
| in `commit()` (current) | 192 | 0 | **12.21** | -- |
| head of the region, right after `fetch()` | 191 | 0 | 12.83 | **+5.1%** |
| tail of the region, after the K loop | 192 | **209** | 60.87 | **+398%** |

Both placements fail, for two different reasons, and together they close the
idea rather than just this implementation of it.

**The head placement loses the latency hiding.** In the current kernel the fetch
at iteration *i* loads the next stage into `raw[]`, and the decode does not touch
those registers until `commit()` at iteration *i+1* -- a full K loop plus a
barrier later. That deferral is what hides the global-load latency, and it is not
incidental: it is why the gap exists. Decoding immediately after the fetch
removes the slack, so the decode stalls on loads that have not landed. It is
serial either way, just serial in a worse place.

**The tail placement is the version that would actually work, and the register
file forbids it.** The whole point was to put the decode inside the K loop's
instruction stream so the MMAs would cover it. At that point in the schedule the
live set is the accumulators (64 VGPRs), the MMA operand fragments (about 48),
the raw weights from `fetch` (8) and the decoded weights (8), on top of
addressing and loop state. The compiler responds by spilling 209 registers, and
at 128x128 w4n2 (256 threads, more per-thread budget pressure) 253 -- five times
slower. There is no placement that both preserves the one-stage deferral and
avoids the live-range conflict, because preserving the deferral is what forces
`raw[]` to stay live across the K loop in the first place.

One curiosity worth recording rather than acting on: at 128x128 w4n2, where two
blocks are resident, the *head* placement measured 12.67 ms against 12.92 ms for
the unhoisted form -- a 1.9% gain. That is the second block absorbing the stall,
and it is consistent with everything above; it is also far too small to chase.

The 7.3% commit is therefore not recoverable by reordering. That also bounds
what was ever available: the window holds one decode and four `ds_store_b128`
per thread, and at 1024 threads that store is 64 KiB per stage, roughly 512
cycles at 128 B/cycle. If the window is store-bandwidth-bound, the decode's share
of it is small enough that perfect hoisting would have been worth a few percent
at best -- which is what the head placement measured, as a loss.


## int8: the tuning target was dead code, and the A/B harness had no noise floor

This section retracts the int8 findings above it, including the section it
replaces. Three separate things turned out to be false.

### 1. The model does not run the wave64 kernel

`UseQwen27bFp16Prefill` (prefill_chunk.cpp:102) returns false -- electing the
q8/int8 route -- only when `batch < 1024`. A bench chunk of 2048 tokens therefore
takes the **fp16** path. Tracing the chunk sizes for `gufo bench -p 2048 -n 1 -r 1`:

| chunk_size | ForwardPromptChunk calls | GEMM path |
| ---: | ---: | --- |
| 2048 | 2 | `HalfPrefillGemmKernel<256,256,8,4,...>` (fp16) |
| 32 | 1 | `WKQuantA8BlockedWmmaGEMMKernel<128,64,4,8,1,...>` |
| 16 | 1 | same |

Tracing `LaunchBatchedQuantGEMMPreQuantized` across the whole run gives 496 calls
at `batch=32` and 496 at `batch=16`, and **not one call at batch >= 96**.
rocprofv3 agrees: 916 int8 GEMM dispatches, **all** `<128,64,4,8,1,Type,32>`.

| instantiation | dispatches | reachable |
| --- | ---: | --- |
| `<128,64,4,8,1,Type,32>` (regime 3, 9 <= batch < 96) | **916** | yes, always |
| `<128,128,2,4,2,Type,32>` (regime 2, batch >= 96) | 0 | no |
| `<128,128,2,4,1,Type,64>` (TryLaunchQuantPrefillWave64) | 0 | no |

The whole wave64 investigation -- the ISA tables, the eight-geometry tile sweep,
the occupancy comparison and the "production tile" conclusion -- describes a
kernel this model never dispatches. `pp2048` is an **fp16** measurement. The int8
GEMM serves only chunks below 1024 tokens, and in this bench only 32 and 16.

Occupancy of the instantiations that exist (Q4_K, measured via
hipOccupancyMaxActiveBlocksPerMultiprocessor):

| instantiation | registers | LDS | blocks/CU | wave |
| --- | ---: | ---: | ---: | ---: |
| `<128,64,4,8,1>` (live) | 186 | 30720 | 2 | 32 |
| `<128,128,2,4,2>` (dead) | 205 | 20480 | 3 | 32 |
| `<128,32,4,4,2>` | 185 | 25600 | 2 | 32 |
| `<128,16,4,8,1>` | 160 | 23040 | 2 | 32 |

### 2. The A/B harness had no measured noise floor

Two byte-identical copies of one binary, same harness, same alternating order,
same prompt:

| round | first arm | second arm |
| ---: | ---: | ---: |
| 1 | 503.51 | 524.51 |
| 2 | 528.99 | 511.18 |
| 3 | 523.79 | 480.34 |
| **mean** | **518.8** | **505.3 (-2.6%)** |

The null effect is -2.6% on the mean with per-round swings reaching -8%. Two of
the three int8 A/Bs were **no-ops** -- the unforced `bk4` and `bk1` runs never
dispatch the edited kernel, so their true effect was exactly zero and the
measured -5.9% and -11.2% are that noise. The `w2n2` run was made under
`GUFO_FORCE_Q8_PREFILL=1`, which does force the int8 route at batch 2048 and so
does dispatch wave64, but -5.8% sits at the edge of, not outside, this envelope.
**All three are retracted.**

Standing rule from here: a model-level A/B carries a null control (or an
already-established envelope) *and* a confirmation that the edited kernel is
actually dispatched on the measured path. The large effects recorded elsewhere in
this document -- fp16 double buffering at +62%/+296%, the 209-spill decode hoist
at +398% -- are far outside this floor and stand.

### 3. What this actually means

* **The q8/int8 prefill route is not on the pp2048 critical path.** Optimising it
  cannot move `pp2048`; the fp16 kernel owns that number for this model.
* The int8 route serves `batch < 1024`: verification, speculative drafting and
  short prompts. That is where its value is, and where it must be measured.
* `<128,64,4,8,1,Type,32>` (2 blocks/CU, 30 KiB LDS) has never been tuned. With
  `kBN = 64`, a 32-token batch fills half the token tile before anything else is
  considered.


## fp16 prefill, measured properly: a verified pp2048 budget

The int8 retraction above ends with a rule: establish that the target is on the
measured path before optimising it. This is that step for fp16, and it replaces
the earlier phase estimates in this document, whose in-kernel clock numbers were
mutually inconsistent with the derived claims.

`rocprofv3 --pmc` over `gufo bench -m Qwen3.8-27B-IQ4_XS-3.84bpw.gguf -p 2048
-n 1 -r 1`. Kernel timestamps are broken on this box (start == end), but the
counters are sound, and **`GRBM_COUNT` sums to 9.33e9 cycles = 3.73 s at
2.49 GHz against a measured 3.86 s wall** -- so the GPU is essentially never idle
and these shares are the real budget:

| stage | share of GPU-busy cycles |
| --- | ---: |
| **fp16 GEMM, all `HalfPrefillGemmKernel<256,256,8,4,...>`** | **78.1%** |
| int8 GEMM (`<128,64,4,8,1,...>`, the sub-1024-token chunks) | 8.6% |
| all non-GEMM, fragmented (DeltaNet 2.6%, GEMV 2.8%, attention 1.2%, ...) | ~13% |

For a 2048-token chunk specifically the fp16 share is higher still, because the
int8 8.6% comes only from the 32- and 16-token chunks. **The fp16 GEMM kernel is
the prefill.** Nothing else is worth touching until it moves.

### Efficiency by instantiation, and its mechanism

Joining the traced FLOPs per instantiation against the per-instantiation counters
(`eff = FLOP share / time share`; >1 is faster than the family average):

| type | epilogue | FLOP% | time% | eff | VALU/MFLOP | vgpr | LDS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| IQ3_S | kStore | 15.44 | 15.57 | 0.99 | 697 | 144 | 65536 |
| **IQ3_S** | **kResidual** | 11.22 | 12.53 | **0.89** | 681 | 144 | 65536 |
| IQ3_XXS | kResidual | 11.56 | 12.06 | 0.96 | 637 | 144 | 65536 |
| IQ3_S | kSwiGLU | 7.32 | 7.91 | 0.93 | 789 | 144 | 65536 |
| Q4_K | kStore | 8.61 | 7.57 | 1.14 | 482 | 136 | 65536 |
| IQ3_XXS | kSwiGLU | 6.95 | 7.49 | 0.93 | 744 | 144 | 65536 |
| IQ3_XXS | kStore | 6.55 | 6.54 | 1.00 | 652 | 144 | 65536 |
| IQ4_XS | kGateUp | 6.59 | 5.76 | 1.14 | 603 | 192 | 65536 |
| IQ4_XS | kResidual | 5.38 | 5.04 | 1.07 | 555 | 192 | 65536 |
| IQ4_XS | kSwiGLU | 5.12 | 5.01 | 1.02 | 658 | 192 | 65536 |
| IQ4_XS | kStore | 5.15 | 4.60 | 1.12 | 561 | 192 | 65536 |
| Q3_K | kStore | 3.66 | 3.66 | 1.00 | 704 | 192 | 65536 |
| Q4_K | kResidual | 3.14 | 2.96 | 1.06 | 478 | 136 | 65536 |

Three things follow, and they replace the earlier account of this kernel:

1. **Efficiency is set by VALU instructions per FLOP**, which is the weight
   decoder plus the epilogue. Q4_K needs ~480 per MFLOP; the IQ3 family needs
   637-789, **1.4-1.65x more**, and that ordering reproduces the efficiency
   ranking. IQ3_S and IQ3_XXS together are **59% of the fp16 FLOPs**.
2. **Occupancy is identical for every instantiation** -- exactly 65536 B of LDS,
   one CTA per CU, 32 warps -- so the spread is not occupancy, and there is no
   spare block anywhere to absorb a stall.
3. It is **not** LDS-bandwidth or memory bound: `SQC_LDS_BANK_CONFLICT` is
   0.04-0.11 per LDS op, and `MemUnitBusy` never exceeds 0.3%.

So the kernel is limited by decoder work sitting on the critical path, with the
64 KiB LDS cap pinning occupancy at one block and the doc's own BK fit charging
**~26 us per stage boundary, ~17% over the 80 stages**.

### Structural options, with their measured or fitted prizes

| # | change | prize | evidence |
| --- | --- | ---: | --- |
| A | cut IQ3 decoder VALU per FLOP | ~5-7% of pp2048 | IQ3 family at 0.89-1.00x against Q4_K's 1.14x on 59% of FLOPs |
| B | break the 64 KiB cap so BK can reach 8 | ~8.5% of the GEMM | BK=2 vs BK=4 is 17%; the fitted line gives BK=8 |
| C | remove the `kResidual` read-modify-write | ~1-2% | 31.7% of FLOPs; worst instantiation is IQ3_S kResidual at 0.89x |
| D | extend gate/up fusion to IQ3_S/IQ3_XXS | ~2% | only 6.6% of FLOPs take the dual path today |

A and B are the structural ones; C and D are narrower. All prizes except the
0.89x/1.14x spread are extrapolations from the fits above, not measurements.


### Three fp16 levers, screened and refuted

The verified budget above identifies the fp16 `<256,256,8,4,...>` kernel as 78.1% of
pp2048 and its efficiency as tracking VALU per FLOP. Three candidate attacks were
screened at the production shape (M=17408, K=5120, batch=2048, 256x256 w8n4
kStore) before any of them was built. All three fail.

**Per-format throughput, production dispatch (`LaunchBatchedQuantGEMMFp16`),
best of three:**

| type | bytes/weight | TFLOPS | decode cost (VALU/MFLOP) |
| --- | ---: | ---: | ---: |
| IQ4_XS | 0.531 | **32.61** | 561 (grid codebook) |
| Q4_K | 0.5625 | 32.34 | 482 (4-bit unpack) |
| Q3_K | 0.430 | 31.42 | 704 |
| Q5_K | 0.6875 | 31.04 | 491 |
| IQ3_XXS | 0.344 | 30.86 | 652 |
| IQ3_S | 0.4375 | 30.63 | 697 |
| Q8_0 | 1.0625 | 29.54 | **231 (about one multiply)** |
| Q6_K | 0.820 | 27.97 | 589 |

**1. Pre-expanding the weights to FP16 is refuted.** Q8_0 has the cheapest possible
decoder -- it needs one multiply per element against Q4_K's 4-bit unpack and
IQ3_S's grid lookup -- and it is still **9% slower** than Q4_K, because it moves
1.06 bytes per weight against 0.5625. So the entire decode is worth **less than
9%**, and resident FP16 weights would move 2.0 bytes, 3.6x Q4_K. The bandwidth
cost exceeds the decode saving, and the plan is a net loss. This is the cheapest
refutation in this document: one bench run, no kernel written.

**2. Cutting decoder VALU is refuted as the primary limiter.** IQ4_XS has a *more*
expensive decoder than Q4_K (561 against 482 VALU/MFLOP) and is *faster* than it
(32.61 against 32.34). Q8_0 has by far the fewest instructions of any format and
is slower than all but one. Whatever sets throughput, it is not decoder
arithmetic -- the earlier VALU correlation was real but not causal.

**3. Widening `Complete` (tail-check elimination) to all types is refuted.** The
production condition admits only Q4_K/Q5_K; a comment claims the other formats
retain a better register schedule. That claim is correct and the measurement is:

| type | Complete=false | Complete=true | effect |
| --- | ---: | ---: | ---: |
| Q4_K (control, already true) | 32.34 | 32.4 | -- |
| IQ4_XS | 32.61 | 32.63 | neutral |
| Q3_K | 31.42 | 31.81 | +1.3% |
| IQ3_S | 30.63 | **29.10** | **-5.0%** |
| IQ3_XXS | 30.86 | **29.80** | **-3.5%** |
| Q6_K | 27.97 | 31.72 | +13.5% (0.18% of FLOPs) |

Only Q6_K gains, and it carries 0.18% of the fp16 FLOPs. Widening the condition
would cost 5% on IQ3_S, which carries 34%. Reverted.

### What the plateau actually is

Across eight weight formats whose decoders differ by **4x in instruction count**
and **3x in bytes per weight**, throughput spans only **28.0-32.6 TFLOPS** -- and
with the best `Complete` setting per type, 29.1-32.7. Every format lands within
about 12% of every other. **The kernel is limited by something the weight format
does not touch.**

The structure says what: the two candidates that could break the plateau are both
already closed. LDS is exactly 65536 B at `(BM+BN)*BK*32`, so `BK` is pinned at 4
and there are 80 stage boundaries, each costing a fitted **~26 us** (BK=2 against
BK=4 is 17%). A `BK=8` stage would need 128 KiB and cannot exist at this tile.
And LDS is not the only cap: variant 4 of the BK sweep halved LDS to 32 KiB and
still reported **one block per CU**, because 191 VGPRs at 1024 threads already
consume the register file. Both resources forbid a second resident block, so
there is nothing to hide the decode -> commit -> barrier -> MMA serialisation
behind.

The structural root is that **LDS is being used as a layout transformer, twice**:
once to convert the decoder's register output into WMMA fragment order (the A
stage, 32 KiB), and again for IQ3_S to transpose the accumulator into
token-major order at store time. Removing the first is the only remaining door --
a decoder that emits directly in fragment layout would free 32 KiB for `BK=8` --
and it is a deep rewrite of every per-format decoder. Nothing cheaper is left:
the epilogue's scalar 2-byte stores are forced by the accumulator fragment layout
(each lane holds 8 values strided by `m`), so the direct-store path cannot be
widened without the very transpose IQ3_S already pays for.



### gfx1151 capability survey: two corrections, and what is left unused

**I got the int4 mechanism wrong in an earlier revision of this section, and the
document already had it right.** The 2x int4 ceiling does *not* come from a 32-wide
MMA. It comes from `wmma_i32_16x16x16_iu4` -- the **same 16x16x16 shape** -- which
retires about twice the instructions per second of iu8 and carries half-size
operand fragments (2 VGPRs against 4). `Deferred: INT4 WMMA doubles matrix
throughput` says exactly this, and the 32-wide `wmma_i32_16x16x32_iu4` is gfx12.
The claim that `iu4` is "the only wide MMA" was wrong twice over.

Verified inventory of what gfx1151 exposes, by compilation and measurement rather
than by inference:

| capability | status | evidence |
| --- | --- | --- |
| `16x16x16 iu4` MMA (**2x**, the int4 lever) | **available, unused** | compiles: `v_wmma_i32_16x16x16_iu4 v[1:8], v[0:1], v[0:1], v[1:8]` |
| `16x16x16 f16`, f32 accumulate | available (in use) | `v_wmma_f32_16x16x16_f16` |
| `16x16x16 f16`, **f16 accumulate** | available, **no faster** | measured **1.02x** f32-acc in an 8-chain loop |
| `16x16x32 f16` | gfx12.5 only | intrinsic declared in `AMDGPUWMMAIntrinsicsGFX1250` |
| `16x16x32 iu4`, `16x16x64 iu8`, fp8 / f8f6f4 | gfx12 only | prior section |
| fp8 conversions (`cvt_pk_fp8_f32`) | gfx12 only | `needs target feature` |
| `global_load_lds` (VMem -> LDS, no registers) | **not available** | `needs target feature vmem-to-lds` |
| `s_barrier_signal` / `s_barrier_wait` | **not available** | `needs target feature` |
| `ds_swizzle_b32` | **available, unused** | compiles |
| `v_permlane16_b32` | **available, unused** | compiles |
| VALU int4 dots (`sdot8`/`udot8`) | gfx12 only | prior section |

Device limits, measured with `hipDeviceGetAttribute` on gfx1151: **LDS 65536 B per
CU and 65536 B per block**, 196608 VGPRs per CU, 2048 threads per CU. A `BK=8`
stage at 256x256 needs `(256+256)*8*32 = 131072` B, so the "BK=8 cannot exist"
claim is now **measured rather than assumed** -- it is correct.

Three conclusions, and only the third is new work:

1. **Nothing in the unused set is a second route to 2x.** F16-accumulate, the one
   candidate that would have doubled the fp16 path at a precision cost, is 1.02x.
   **This is a statement about the ceiling, not about the kernel.** 48.35 TFLOPS is
   the widest the f16 MMA gets on this part; the production kernel reaches about
   **32 TFLOPS, or 66% of it**, so roughly **34% -- about +50% on the GEMM -- is
   structural and still on the table.** An earlier revision of this section called
   the kernel "at its hardware ceiling", which conflated the two and was wrong.
2. **The int4 lever is confirmed available, and confirmed narrow.** It is the same
   16x16x16 shape, reachable today, and gated entirely by `linearity` of the
   weight codes -- the eligibility problem, not a kernel problem.
3. **The one unused primitive with a plausible target is the lane-permute pair.**
   `ds_swizzle_b32` and `v_permlane16_b32` are available and this engine uses
   neither. The IQ3_S epilogue -- 34% of fp16 FLOPs -- currently transposes its
   accumulator through LDS with a `__syncthreads` plus two wave barriers per
   fragment, precisely because the raw fragment layout cannot be stored
   contiguously. A shuffle-based transpose is the alternative the hardware still
   offers there. It is untested, and it is the only thing this survey turned up
   that is both unused and aimed at a measured cost.


### The fp16 staging gap, decomposed by ablation

Compile-time ablation flags on the real kernel (`int Ablate = 0`, defaulted, so
production is bit-identical -- the q4kxl equivalence test passes unchanged). All
arms at m=17408, k=5120, batch=2048, Q4_K, 256x256 w8n4, Complete=true, in one
process:

| ablated | ms | TFLOPS |
| --- | ---: | ---: |
| baseline | 9.188 | 39.7 |
| fetch only (commit kept) | 8.299 | 44.0 |
| barriers only | 8.772 | 41.6 |
| commit + fetch (all staging) | 7.251 | **50.4** |

**Production dispatch and a direct instantiation of the same template agree to
0.5% in the same process** (9.245 against 9.297 ms). That also retires the
per-format numbers above: those were taken back to back on a thermally loaded
GPU, and the true rate for Q4_K at this shape is **39.5 TFLOPS, not 32.3**. Any
part of the "format-insensitive plateau" that was read as a steady ~20% format
effect was substantially heat, and that conclusion needs re-measuring before it
is relied on again.

Decomposition of the **2.04 ms (22%)** that staging and barriers cost:

| component | ms | share | mechanism |
| --- | ---: | ---: | --- |
| commit (LDS stores) | **1.15** | 12.5% | LDS-write bound |
| fetch (global loads + decode) | 0.89 | 9.7% | issue and latency |
| barriers | 0.42 | 4.6% | serialisation |

The commit is **LDS write bandwidth**, and the arithmetic confirms it: 64 KB per
CTA per stage, times 80 stages, times 1088 CTAs is **5.6 GB** of LDS writes, which
at 128 B/cycle across 20 CUs and 2.49 GHz floors at **0.87 ms**. Measured 1.15 ms.

The K loop reads only about 3 KB of LDS per stage against those 16 KB of writes,
so the port is roughly 30% used during the K loop -- **there is spare LDS
bandwidth to hide the commit in**, which is the opposite of the conclusion the
earlier barrier/LDS-capacity chain reached.

The commit is also cleanly partitionable: `ks = idx % BK` assigns every thread to
exactly one K sub-stage, so the commit can be cut into `BK` pieces and each piece
issued during the matching sub-stage of the K loop, with a second LDS buffer so
the write does not collide with the read. That is the next structural change.


### The first fp16 double-buffer attempt: register-bound, and it hung the GPU

Implementation: split the four LDS sub-stages into two buffers of two
(`kStep = BK/2`), commit buffer 0 in the prologue, then each iteration reads one
buffer while committing the next into the other -- **one barrier per iteration
instead of two**, with the fetch still overlapping the K loop.

**It does not work, and the reason is measurable before running anything.** A
compile-only resource probe (`hipFuncGetAttributes` + occupancy, no launch):

| variant | static LDS | private/scratch | VGPRs |
| --- | ---: | ---: | ---: |
| `Stages=1` (production) | 65536 | **0** | 136 |
| `Stages=2` (double buffer) | 65536 | **136 B/thread** | **192** |

192 VGPRs is **exactly the ceiling for 1024 threads** on this part
(196608 VGPRs per CU / 1024). The runtime `buf` index forces the compiler to keep
both buffers' addressing live at once, so it hits the ceiling and **spills 136
bytes per thread to scratch**. This is the same VGPR wall the previous
double-buffering attempts hit ("35 spills"); the design reproduced it rather than
avoiding it.

**The full-size run of that variant hung the GPU hard enough to require a host
restart, and I could not confirm the mechanism.** `dmesg` is not permitted on
this box (`read kernel buffer failed: Operation not permitted`) and
`journalctl -k` contains no amdgpu reset for the period -- its most recent
entries are from 13:01, so kernel GPU messages are simply not being captured here.
What is established is the correlation: a 192-VGPR, 136-bytes-of-scratch kernel
was built, launched at production size, and wedged the device. The mechanism is
not proven, and the change was reverted.

**Two conclusions that change the method:**

1. **Probe the resource footprint before launching anything new.** The probe above
   is compile-only, takes seconds, and would have flagged this variant as
   dangerous before it touched the GPU. Every future kernel variant gets this
   check first, and a tiny-grid run under a hard timeout before a full-size one.
2. **A runtime buffer index is the wrong mechanism here.** It is what pushes the
   register pressure over the edge. The next attempt must keep the LDS addressing
   *immediate* by making the buffer alternation compile-time -- unrolling the loop
   by two so the compiler sees two constant-indexed bodies rather than one
   variable-indexed one -- which is also how the existing kernel gets its
   registers down to 136.


### The hang, resolved from the journal

Follow-up with `doas` (which is available; plain `dmesg` is not permitted here).
`journalctl --list-boots` shows the long session as boot `-2`, ending
2026-09-26 12:57:57, and its final entries identify the culprit exactly:

```
12:57:55 systemd[2927]: Stopping [systemd-run] .../runner.js -- bash -c
  "cd /home/q/gufo && bash tools/bench/build.sh .../fp16_ablate.hip ...; /tmp/fp16_ablate"
```

That is the **`Stages=2` double-buffer bench**, still running when the machine was
powered off. The same boot contains **no amdgpu error of any kind** -- no ring
timeout, no GPU reset, no page fault, no VM fault, no OOM -- and the kernel
command line carries `amdgpu.gttsize=24576` (24 GB), so memory was never the
constraint.

**So there was no hardware fault: the kernel never completed and never reported.**
That is the signature of an infinite loop or a barrier deadlock inside my own
variant, not a device problem -- which also means the GPU did not need the
restart, the hung process did. The variant is reverted and the register/scratch
profile above explains why it was never safe to launch in the first place.

**Method rule adopted from this:** before any new kernel variant touches the GPU,
(1) probe `sharedSizeBytes`, `localSizeBytes` and `numRegs` compile-only -- 192
VGPRs at 1024 threads with 136 B/thread of scratch is a rejected profile -- and
(2) run a tiny grid under a hard `timeout` with `doas dmesg` watched alongside
before ever running at production size.


### The double buffer, done properly: bit-exact, spill-free, and 10% slower

Second attempt, fixing the register problem two ways: the buffer index is now
**compile-time** (two literal-indexed K bodies via `k_loop.template operator()<0|1>()`
so the LDS slot is a constant), and the merged mapping puts A on threads
`[0, BM*kStep)` and B on the rest so both keep full thread occupancy instead of
half idling.

Compile-only resource gate, run before the kernel went near the GPU:

| variant | static LDS | scratch | VGPRs |
| --- | ---: | ---: | ---: |
| `Stages=1` | 65536 | **0** | 136 |
| `Stages=2` | 65536 | **0** | **155** |

Zero scratch and 155 VGPRs against a 192 ceiling, so it is launchable -- and it
launches: **bit-identical output at two shapes** (0 of 4,194,304 and 0 of
35,651,584 elements differ) with no new dmesg event.

And it is slower:

| shape | `Stages=1` | `Stages=2` | delta |
| --- | ---: | ---: | ---: |
| m=4096 k=2560 batch=1024 | 0.6047 ms / 35.51 TF | 0.6314 ms / 34.01 TF | -4.4% |
| m=17408 k=5120 batch=2048 | 8.8800 ms / 41.11 TF | 9.7789 ms / 37.33 TF | **-10.1%** |

**The mechanism is the LDS capacity, from the other direction.** A double buffer
needs two *full-size* stages; at 256x256 that is 128 KiB against the 64 KiB the part
has. So each buffer can only be **half** a stage: the stage count doubles from 80 to
160, and because the single buffer needed two barriers per stage while each half
buffer needs one, the **barrier count is unchanged at 160**. The barrier saving is
exactly zero, and the per-call fixed cost of fetch and commit doubles. The overlap
does not pay for the extra boundaries.

This is the same verdict the earlier attempt reached, but now with a bit-exact,
spill-free implementation and the reason identified. **The commit's 1.15 ms cannot
be overlapped at this tile size**: hiding it needs a second buffer of the same stage
size, the LDS budget is exactly consumed by the first, and shrinking the stage to fit
two costs more in boundaries than the overlap returns.

One measurement note: `Stages=1` reads **41.11 TFLOPS** here against 39.5 in the
ablation session, so there is roughly 4% of session-to-session variance in this
harness. Effects below that are not resolvable without a null control.


### The last structural door, measured shut: freeing the A stage is worth at most 5.5%

The remaining idea was a weight decoder that emits **directly in WMMA fragment order**,
so LDS stops being used as a layout transformer and the A stage (32 KiB) could be
freed for a deeper K stage. Before rewriting every per-format decoder, measure its
ceiling. An `AFree` ablation removes A entirely -- no A staging, and the A fetch is
dead-coded because nothing consumes it, so **no decode either** -- and reads the A
operand out of `s_b`'s storage (valid memory, garbage values, timing unaffected).

Compile-only resource gate, all safe:

| variant | LDS | scratch | VGPRs |
| --- | ---: | ---: | ---: |
| BK=4, A staged (baseline) | 65536 | 0 | 136 |
| BK=4, AFree | 32768 | 0 | 154 |
| BK=5, AFree | 40960 | 0 | 166 |

**BK=8 is impossible even with A free**: B alone is `8*256*16*2 = 65536`, exactly the
budget, and the CTA must still declare something for `s_a`. BK=5 (64 stages against
80) is the deepest that fits.

Measured at m=17408, k=5120, batch=2048:

| variant | ms | TFLOPS | delta |
| --- | ---: | ---: | ---: |
| BK=4, A staged | 9.087 | 40.17 | -- |
| BK=4, AFree | 8.764 | 41.66 | +3.6% |
| BK=5, AFree | 8.589 | 42.51 | **+5.5%** |

**+5.5% is the absolute upper bound on the entire decoder rewrite**, and +3.6% of it
lies inside the ~4% session variance measured above. A real implementation would
deliver less: it would not remove A's *decode*, only its staging, and the fragment
layout replicates each row across two lanes, so the decode would either double or
need extra permute-shuffle work to avoid that.

**Verdict: not worth attempting.** Rewriting every decoder -- with the hang and
correctness risk that carries -- for a measured ceiling of +5.5% on the GEMM (about
+4% on pp2048) that a realistic implementation would not reach.

### The fp16 structural search, closed

The 22% overhead decomposes as **commit 1.15 ms + fetch 0.89 ms + barriers 0.42 ms**,
and each component is now pinned by a measured constraint rather than an argument:

| constraint | evidence |
| --- | --- |
| the commit cannot be overlapped | LDS is exactly full; the double buffer measured **-10%** |
| the K stage cannot grow (BK <= 4) | `(BM+BN)*BK*32 = 64 KiB` exactly at 256x256 |
| the tile cannot grow | 8 accumulators/thread at 1024 threads; 256x512 needs 16 and blows the regfile |
| freeing the A stage does not pay | measured ceiling **+5.5%**, and it would double the decode |

The fp16 kernel is at a genuine structural optimum for this design on this part.
Further gain requires a different *design*, not a different configuration -- and the
one design that would break these constraints was measured to be worth at most 5.5%,
so it is not worth its risk.


### The harness was biased; corrected decomposition and re-measurements

Every ablation and A/B above ran its arms **sequentially in a fixed order**, which loads
the GPU progressively. A null check had already exposed the size of that: two
byte-identical binaries measured **-2.6%** on the mean with per-round swings to -8% when
one was always run second, and later sessions read the *same* production configuration at
**8.34 ms and 9.09 ms -- ~9% apart -- with the first-run arm always fastest.**

Rebuilt as **interleaved rounds with the arm order alternated every round**, reporting the
median plus the min/max. Re-measured decomposition, m=17408, k=5120, batch=2048, 7 rounds:

| ablated | median ms | min | max | TFLOPS | gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 9.181 | 8.864 | 9.256 | 39.77 | -- |
| no barriers | 8.558 | 8.466 | 8.810 | 42.66 | **+6.8%** |
| no commit (all staging) | 7.206 | 7.082 | 7.287 | 50.67 | +21.5% |
| no fetch | 8.237 | 8.143 | 8.287 | 44.32 | +10.3% |
| no epilogue | 9.129 | 8.982 | 9.284 | 39.99 | +0.6% |

Commit **1.03 ms**, fetch **0.94 ms**, barriers **0.62 ms**, epilogue **~0**. The earlier
sequential figures are superseded, and the barrier number was the one most wrong
(4.6% -> **6.8%**).

Re-measured on the same interleaved harness, the items that had been decided on the biased
one:

| item | sequential | interleaved | verdict |
| --- | ---: | ---: | --- |
| double buffer (`Stages=2`) | -10.1% | **-11.0%** (9.640 vs 8.688 ms, ranges disjoint) | **stands** |
| `iglp_opt(0)` for Q4_K | -2.8% | **+0.5%** (ranges overlap) | neutral |
| epilogue ablation | +0.8% *slower* | **+0.6% faster** | epilogue is negligible |

And the grid swizzle, which had never been swept (`GShift`, production default 3):

| group_shift | median ms | TFLOPS | delta |
| ---: | ---: | ---: | ---: |
| -1 (default 3) | 9.157 | 39.87 | -- |
| 0 | 9.371 | 38.96 | **-2.3%** |
| 1 | 9.199 | 39.69 | -0.5% |
| 2 | 9.125 | 40.01 | +0.3% |
| 4 | 9.196 | 39.70 | -0.4% |
| 5 | 9.163 | 39.84 | -0.1% |

The shipped heuristic is already at the optimum; shift 0 is clearly worse and the rest is
inside noise. Not a lever.

**The residual that remains:** even interleaved, the session-to-session spread is ~5% --
the same production configuration read 42.02 and 39.87 TFLOPS in consecutive runs.
Interleaving cancels ordering bias *within* a run but not across runs, so only within-run
comparisons are trustworthy at the few-percent level. Every ablation in this kernel
(`Ablate` bits 1/2/4/8/16) and the grid swizzle (`GShift`) are kept as defaulted
compile-time diagnostics so they can be re-run on the interleaved harness.


### The harneess validated, and the K stage repriced

The interleaved harness now runs the **same configuration twice**, as the first and last
arm, so the residual bias is measurable rather than assumed. m=17408, k=5120, batch=2048,
5 alternating rounds, median of the per-round means:

| arm | median ms | min | max | TFLOPS | delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| bk4 control A | 8.775 | 8.671 | 8.801 | 41.60 | -- |
| bk1 | 16.456 | 16.332 | 16.553 | 22.19 | **-87.5%** |
| bk2 | 11.393 | 11.305 | 11.448 | 32.04 | **-29.8%** |
| bk4 control B | 8.781 | 8.659 | 8.878 | 41.58 | **-0.1%** |

**The two controls agree to 0.1% with overlapping ranges**, which is the null control this
harness never had. Everything measured interleaved on it is trustworthy at the ~1% level;
everything measured sequentially was not.

Fitting `T = T_k + (320/BK)*c` to bk4 and bk2:

* per-stage cost **c = 32.7 us** (the sequential fit said 26 us),
* K-loop floor **T_k = 6.16 ms**.

So a `BK=8` stage -- 40 stages instead of 80 -- would be **7.47 ms against 8.78 ms, about
+15%**, more than twice the +8.5% the old fit predicted. It remains impossible, and now the
reason is a tight three-way argument rather than one measurement:

| budget | limit | gives |
| --- | --- | --- |
| accumulator registers | 8 accumulators/thread at 1024 threads | `BM*BN <= 65536`, minimised at 256x256 |
| LDS | `(BM+BN)*BK*32 <= 65536` at BM+BN = 512 | `BK <= 4` **exactly** |
| consequence | -- | BK=8 needs 128 KiB, and B alone at BK=8 already fills the 64 KiB, leaving no room even for a 2-byte A placeholder |

The tile re-test seals the first row independently: every non-square aspect at the same
area is worse, and the mirror warp split (256x256 w4n8) is **-4.7%**. So the accumulator
budget really does pin the tile at 256x256, LDS really does pin BK at 4, and the +15% that
BK=8 would buy is behind a wall with no gap in it.

## The fetch cost is not the fetch: three ablations that measured the compiler

The decomposition above (commit 1.03 ms, fetch 0.94 ms, barriers 0.62 ms) was built
from ablation bits that **skipped** the fetch loop. That is not a neutral edit. With
`raw`/`rb` left undefined, `DecodeRaw` and the `s_a`/`s_b` stores become undefined
behaviour, and LLVM is entitled to fold them away. It did: at `Ablate=4` the loop body
dropped from 218 to 113 instructions, and at `Ablate=32` the shared-memory footprint
itself fell from 65536 to 32768 bytes because `s_b` went dead. **Those arms were not
measuring the fetch; they were deleting the decoder and half the LDS tile.**

The bits now route into the existing zero-fill `else` branch instead of skipping the
loop, so `raw`/`rb` are defined and every dependent store survives. Each arm is then
checked in the ISA before it is believed:

| ablation | VGPRs | LDS | loop body | global | lds ops | wmma | waitcnt |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 control | 136 | 65536 | 218 | 9 | 52 | 32 | 21 |
| 32 no B (activation) fetch | 130 | 65536 | 211 | 7 | 52 | 32 | 19 |
| 64 no A (weight) fetch | 150 | 65536 | 116 | 4 | 52 | 32 | 16 |
| 96 neither | 116 | 65536 | 108 | 2 | 52 | 32 | 17 |

Interleaved run, m=17408, k=5120, batch=2048, 5 alternating rounds, median:

| arm | median ms | TFLOPS | delta |
| --- | ---: | ---: | ---: |
| control A | 9.215 | 39.62 | -- |
| no B fetch (32) | 9.226 | 39.57 | **-0.1%** |
| no A fetch (64) | 8.705 | 41.94 | **+5.5%** |
| no fetch (96) | 8.119 | 44.96 | **+11.9%** |
| control B | 9.266 | 39.40 | -0.5% |

Three results, each with its own falsifier:

1. **The activation fetch costs nothing.** Bit 32 is clean -- LDS, lds ops and wmma are
   all unchanged, only 7 of 218 instructions leave -- and it buys -0.1%. An independent
   probe agrees: `Ablate=128` keeps the identical instruction count and per-lane address
   pattern but pins the k window so the loads re-read 128 bytes, and it buys +0.6%. So
   the 8.2-8.6% previously attributed to the activation loads was the deleted
   `s_b`/decoder work, not the loads. **Falsifier: an arm that removes exactly the
   activation loads while preserving every dependent instruction, showing a cost.**
2. **The weight path is the cost, and it is the decode as much as the load.** Bit 64
   removes 102 instructions, most of them the Q4_K decoder the constant `raw` lets the
   compiler fold, so the honest claim is 5.5% for load *plus* decode, not for the load.
   **Falsifier: a weight-fetch arm that zero-fills the loaded bytes but keeps the decode
   input opaque to the compiler.**
3. **Interleaving the fetch into the K loop is refuted at the compiler, not the
   hardware.** Moving the `fetch` call between the K sub-stages produces
   instruction-for-instruction identical code (218/218, same VGPRs, same mix), so it
   measured +0.1%. The scheduler already places those loads; there was nothing to
   interleave. **Falsifier: an interleaved arm whose ISA differs from the control's.**

### Why the remaining cost is stall structure, not work

The per-stage loop body is 218 warp instructions: 48 `ds_load_b128` feeding 32
`v_wmma_f32_16x16x16_f16`, 90 VALU of Q4_K decode, 4 `ds_store_b128`, 9 global, 21
`s_waitcnt`, 2 `s_barrier`. With 32 warps per CTA, one CTA per CU and four schedulers,
that is `8 x 218 = 1744` issue cycles per scheduler per stage against a ~10075-cycle
stage -- **issue is only ~17% occupied**. Independently, 256 WMMA per SIMD per stage at
~32 matrix-pipe cycles each is ~8192 cycles, or **~81% of the matrix pipe**. The kernel
is matrix-pipe-bound and the staging is not paying for itself in instructions: it costs
because the per-stage `s_barrier` plus 21 `s_waitcnt` re-synchronise every warp every
stage, so when one warp stalls on staging no other warp has matrix work to cover it.

That closes the two cheap structural escapes as well, and for the same reason as BK=8:
two CTAs per CU would let one CTA's MMAs fill the other's staging, but it needs LDS
<= 32 KiB per CTA, and the only tiles that fit (256x256 at BK=2, or 128x128 at BK=4)
were already measured at -29.8% and -76.8%. Named/partial barriers would remove the
second barrier per stage, but `s_barrier_signal`/`s_barrier_wait` do not exist on
gfx1151.

What is left is a producer/consumer split -- warp specialisation -- so that the warps
that fetch and decode are never the warps holding live matrix work. That is a rewrite of
the kernel's warp mapping, not a parameter, and it is the next thing to try.
