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

### The decode cannot be moved off the critical path either

The same ISA pass locates the staging precisely. Inside one stage body, the two
`s_barrier`s split it into regions:

| region | instructions | wmma | decode VALU |
| --- | ---: | ---: | ---: |
| before the first barrier | 100 | 30 | **0** |
| between the barriers | 95 | 2 | **75** |
| after the second barrier | 23 | 0 | 1 |

So the weight decode is a straight-line block sitting between the publish barrier and
the next stage's MMAs, with the matrix pipe idle. It is pure register work on `raw`,
so it is legal to run it inside the *previous* K loop, where the matrix pipe is busy and
issue is only ~17% occupied. Two ways to ask for that were tried:

| arm | median ms | delta |
| --- | ---: | ---: |
| control A | 9.283 | -- |
| decode moved into the K loop (`Ablate=512`) | 9.421 | **-1.5%** |
| `__builtin_amdgcn_iglp_opt(0)` (`Ablate=16`) | 9.303 | -0.2% |
| control B | 9.300 | -0.2% |

**Neither works, and the ISA says why: the decode does not move.** At `Ablate=512` the
loop body is 216 instructions instead of 218 and the region split is unchanged -- 75
decode instructions still sit between the barriers and none before the first. The
AMDGPU machine scheduler treats `s_barrier` as a boundary and re-sinks the register-only
decode into the post-barrier block even when the source places the call inside the K
loop. `iglp_opt(0)` is the scheduler hint the other instantiations already use, and it
is neutral here too. This also re-confirms, on the interleaved harness, the old
sequential verdict that `iglp_opt` was neutral for Q4_K.

The consequence is that the ~19% staging gap is not reachable from the source. The
schedule is the compiler's, the barriers are the synchronization the tile needs, and
every parameter that would amortise them is behind the LDS wall. The one move left is to
change which warps hold matrix work: a producer/consumer split, so the warps that fetch
and decode are never the warps the matrix pipe is waiting on. That is a warp-mapping
rewrite, not a parameter, and it is the next experiment.

### Staggering the decode by warp is also neutralised by the scheduler

If every warp enters the decode together -- they all just passed the same barrier -- the
matrix pipe starves for the whole block. RDNA3 gives each warp its own PC, so the decode
can be moved to a warp-dependent sub-stage: `if (ks == (wave & (BK-1))) decode();`. No
new synchronisation, no extra registers (within one iteration the commit reads `ra` and
the K loop then overwrites it, so a single buffer suffices).

| arm | median ms | TFLOPS | delta |
| --- | ---: | ---: | ---: |
| control A | 8.661 | 42.15 | -- |
| decode in K loop (512) | 8.791 | 41.53 | -1.5% |
| staggered decode (1024) | 9.132 | 39.98 | **-5.4%** |
| control B | 8.754 | 41.70 | -1.1% |

The ISA explains it. `Ablate=1024` has a 229-instruction body against 218, VGPRs 136->153,
and decode still split `[1, 73, 1]` across the barrier regions versus `[0, 75, 1]` for the
control. The compiler recognised that the four warp-dependent branches are mutually
exclusive, merged them back into a single decode block, and sank it below the barrier
again. The stagger never reached the hardware, and the 11 extra instructions plus 17
extra VGPRs are the branch overhead that causes the regression.

That is three independent attempts to move the decode off the critical path -- source
placement inside the K loop, the `iglp_opt` scheduler hint, and per-warp staggering --
all ISA-verified as no-ops at the machine level. **Per-warp instruction scheduling cannot
express this overlap**, because the work has to move to *different warps*, not to a
different point in the same warp's stream. That is exactly what a producer/consumer split
does, and it is now the only route left rather than a preference.

### The producer/consumer design is forced into one shape

The LDS budget fixes everything else. Overlap requires two LDS slots (producers fill slot
B while consumers read slot A); at BK=4 one slot is already 64 KiB, so two slots force
**BK=2** (16 KiB A + 16 KiB B per slot, 2 slots = 64 KiB). The MMA mapping then forces
the warp split: 256x256 tile / 16x16 fragments = 256 warp-tiles, so 16 consumer warps
give 16 accumulators each (128 VGPRs, near but under the 192 ceiling) and 16 warps are
free to produce. No other split works -- 24 consumer warps cannot map a 16-row fragment
grid. Synchronisation must be LDS arrive/wait counters, because gfx1151 has no named or
split barriers and a full `s_barrier` is exactly the coupling being removed.

The prize is the whole staging gap: the BK=2 K loop shares the BK=4 floor of 6.16 ms, and
the 2.6 ms that BK=2 loses today is precisely per-stage staging that overlap would hide.

### Warp specialisation: correct, and still slower

The kernel is built. 16 producer warps (8 for A, 8 for B) fill one of three LDS slots
while 16 consumer warps run WMMA on another; the handoff is LDS arrive/wait counters
with `atomicAdd` and a monotonic target, and there is one `__syncthreads()` at kernel
entry to zero them. BK=1 with three slots is what fits: at BK=2 a double-buffered A+B
tile is exactly 65536 bytes and there is no room left for the counters.

It is **numerically exact** -- `max|diff| = 0` and 0 mismatches over 35,651,584 outputs
against the production kernel -- so the protocol, the counter arithmetic and the fragment
layout are all right. It is also much slower:

| arm | median ms | TFLOPS | vs production |
| --- | ---: | ---: | ---: |
| production `256x256 w8n4 bk4` | 10.638 | 34.32 | -- |
| warp-specialised (correct) | 16.529 | 22.09 | **1.554x** |
| warp-specialised, handoff removed (UB) | 11.095 | 32.90 | 1.043x |

The third arm is the important one. Removing the waits and the atomics makes the result
garbage, but it bounds what *any* handoff could achieve: **with a free handoff the design
is still 4.3% behind the kernel it is meant to replace.** The handoff as written costs
5.43 ms, 49% of the runtime, but even spending nothing on it does not win.

The reason is the shape the LDS budget forced. BK=1 means 320 handoffs and a K-block of
16 MMAs, so the consumer's LDS-to-MMA pipeline restarts every stage, and only 16 consumer
warps - 4 per SIMD - are left to feed the matrix pipe. The overlap gain never materialises
because the per-stage critical path grew by more than the decode it was hiding. The escape
is BK=2 with 2 slots and 160 stages, which needs more than 64 KiB once counters are
included; shrinking the tile to make room doubles either the activation traffic or the
weight traffic, both of which are already at their limits.

**Falsifier: a warp-specialised arm at BK=2 with two slots and a free handoff that beats
10.6 ms.** Until that exists, warp specialisation on this part is refuted, not merely
unfinished.

### The K loop is not at its ceiling: the missing measurement

A dedicated probe (`tools/qwen27b/fp16_kloop.hip`) runs the production K loop at the
production shape -- 256x256, w8n4, BK=4, identical `s_a`/`s_b` layout and swizzle -- with
the decode, the commit and the global fetch stripped out. A store the compiler cannot prove
dead keeps the invariant LDS reads from being hoisted out of the stage loop. Interleaved,
5 arms with a duplicated production control:

| arm | median ms | TFLOPS | vs control |
| --- | ---: | ---: | ---: |
| production control A | 9.026 | 40.45 | -- |
| pure K loop, LDS reads + barriers | 8.779 | 41.58 | +2.7% |
| pure K loop, barriers removed | 16.844 | 21.67 | -86.6% |
| **MMA only (fragments in registers)** | **7.330** | **49.80** | **+18.8%** |
| production control B | 8.868 | 41.17 | +1.7% |

Three things follow.

1. **The old 50.67 TF "K-loop-only" number was an artifact.** That arm removed the commit,
   so nothing wrote `s_a`/`s_b`, so the stage loop's LDS reads were loop-invariant and got
   hoisted -- it was an MMA-only measurement wearing a K-loop label. The proper MMA-only
   figure measured here, 49.80 TF, lands on it almost exactly. The 48.35 TF "peak tool"
   number was never the thing being compared against.
2. **Staging is nearly free.** The real kernel is 8.868-9.026 ms and the pure K loop is
   8.779 ms: the decode, the commit and the global fetch together cost only **1-3%**, not
   the 11.9% the fetch ablation suggested. They fit in issue slots the matrix pipe leaves
   idle. This is the strongest evidence yet that fp16 staging was never the problem.
3. **The headroom is in the LDS fragment feed and the two barriers per stage.** Dropping
   from 41.58 to 49.80 TF -- **16.5%** -- is what removing the LDS reads and barriers buys.
   That is the largest single identified fp16 gap and it is inside the K loop, not around
   it.

Caveat, stated because it limits the conclusion: the two hypotheses inside point 3 cannot
be separated from these arms. The no-barrier control came out 2x *slower* than the barrier
version, which is the opposite of what a barrier costs and means the scheduler produced a
different and much worse loop for that instantiation. Until a no-barrier arm behaves, the
16.5% is "LDS feed + barriers", not either one alone.

**Falsifier: a K-loop variant that keeps the LDS feed and the barriers but beats 41.58 TF**
-- that would mean the 16.5% is recoverable and the ceiling is higher still. Conversely,
any claim that fp16 prefill is at its ceiling is now dead: the measured MMA-only rate at
this exact shape is 49.80 TF while the shipped kernel runs at 40.45.

### The 16.5% is the LDS fragment feed, not the barriers

The previous probe could not separate the two because its no-barrier arm came out 2x
slower than the barrier version, which is impossible for a barrier. That arm is discarded
and replaced by one holding the barriers constant while the fragment feed is removed, so
the two terms become a clean difference:

| arm | median ms | TFLOPS | vs control |
| --- | ---: | ---: | ---: |
| production control A | 9.055 | 40.32 | -- |
| pure K loop: LDS feed + barriers | 8.927 | 40.90 | +1.4% |
| pure K loop: no LDS feed, barriers kept | 7.536 | 48.44 | +16.8% |
| MMA only, no barriers | 7.424 | 49.17 | +18.0% |
| production control B | 8.969 | 40.70 | +0.9% |

Because arms 2 and 3 differ only in the fragment feed, and arms 3 and 4 only in the
barriers, the gap decomposes without ambiguity:

| term | cost | share of the 16.5% |
| --- | ---: | ---: |
| LDS fragment feed | 1.391 ms | **15.2%** |
| the two `s_barrier`s per stage | 0.112 ms | **1.2%** |
| decode + commit + global fetch | ~0.13 ms | ~1.4% |

**The barriers are essentially free and the staging is essentially free. The entire
remaining fp16 gap is the LDS fragment feed.** This also retires the \"barriers cost 6.8%\"
figure from the old sequential harness; measured interleaved, at the production shape, the
two barriers per stage cost 1.2%.

That is consistent with the arithmetic: the feed moves ~24 KiB per stage per CU against an
LDS that can move megabytes in the same window, so it is not bandwidth. It is the
latency/issue pattern of the fragment loads -- 12 `ds_load_b128` per sub-stage feeding 8
MMAs, with 21 `s_waitcnt` per stage -- and the two swizzles that shape it have never been
touched: the `^ (ks & 3)` row-index xor, and the `(row ^ (row >> 2)) & 1` uint4 ordering
inside `LoadSwizzled`. Those, plus prefetching the next sub-stage's fragments into registers
across sub-stages, are now the only fp16 levers left, and they are all inside this 15.2%.

**Falsifier: a K-loop variant that keeps the LDS feed and beats 40.90 TF** -- i.e. that
closes part of the 15.2% -- or an arm showing the feed can be made free, which would put the
fp16 ceiling back above 48 TF.

### The LDS feed is fragment traffic, and the warp split sets it

Three levers were tried against the 15.2% LDS feed, in the pure K-loop probe.

**Prefetching is neutral.** Holding the next sub-stage's six fragments in registers so
each load has a full sub-stage of MMAs to land behind: 8.928 ms against 8.944 ms for the
shipped order. Load-to-use distance is not the cost, so the compiler was already scheduling
far enough ahead.

**Padding the fragment row stride is impossible.** Rows sit 32 B apart, so rows 0, 4, 8 and
12 all start on bank 0. Widening the stride to 40 B or 48 B does not just fit badly, it
does not link: `local memory (81920) exceeds limit (65536)`. The tile already occupies all
64 KiB, so the bank pattern cannot be changed by padding.

**The warp split is the lever, and the traffic model predicts it exactly.** A fragment
costs its warp 16 rows x 32 B = 512 B, so the LDS bytes per MMA are
`512 * (1/WRS + 1/WTS)`. At 32 warps the tile forces WRS=2, WTS=4 -> 384 B/MMA. At 16 warps
it allows WRS=4, WTS=4 -> 256 B/MMA, a 33% cut:

| arm | median ms | TFLOPS | vs control |
| --- | ---: | ---: | ---: |
| production control A | 9.381 | 38.91 | -- |
| 32 warps, 8x4, LDS feed | 9.415 | 38.77 | -0.4% |
| 16 warps, 4x4, LDS feed | 9.113 | 40.06 | +2.9% |
| 16 warps, 4x4, no LDS feed | 7.875 | 46.36 | +16.1% |
| 32 warps, 8x4, no LDS feed | 7.662 | 47.65 | +18.3% |
| production control B | 9.184 | 39.75 | +2.1% |

The LDS feed costs 1.753 ms at 32 warps and 1.238 ms at 16 -- **a 29% reduction against
the 33% the model predicts**, which confirms the feed is fragment traffic and not latency.
But the no-feed baselines move the other way (7.662 against 7.875): 16 warps are 2.8% worse
at hiding MMA latency, because the CU has four warps per SIMD instead of eight. The net is
+2.9-3.2%, and the control spread in the same run is 2.1%, so the net is marginal while the
feed reduction is not.

Trading the other way is blocked: with 32 warps the tile fixes WRS=2, WTS=4, and getting 16
MMAs per warp would need a 512-row or 512-column tile, which does not fit in 64 KiB. So
the 64 KiB LDS budget sets the fragment traffic, the barrier count, the BK depth and the
warp split all at once -- it is the single constraint behind every blocked lever this
document has recorded.

**Falsifier: applying the 4x4 warp split to the production kernel and beating it by more
than the ~2% control spread.**

### Retraction: the \"LDS feed costs 15%\" attribution is a schedule artifact

The LDS-vs-no-LDS difference was read as the cost of the fragment loads. It is not. Cutting
the fragment requests by four, ISA-verified, makes the kernel *slower*:

| arm | VGPRs | loop body | `ds_load` | `v_wmma` | `s_waitcnt` | median ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LDS feed, shipped order | 116 | 103 | 48 | 32 | 16 | 8.543 |
| LDS feed, requests cut 4x | 120 | 52 | 12 | 32 | 1 | **9.182** |
| no LDS feed (fragments in registers) | 114 | 39 | 0 | 32 | 0 | 7.331 |
| production control A / B | -- | -- | -- | -- | -- | 8.995 / 8.799 |

Removing 75% of the loads costs 7%. Removing all of them buys 18%. The response is not
monotonic in the number of loads, so the 15-18% previously attributed to the LDS feed is
the distance between two *schedules*, not the price of the loads. That attribution is
withdrawn, along with the traffic model built on it.

This also explains the shape of the whole fp16 search. Every lever tried this session --
fetch interleaving, staggered decode, decode inside the K loop, `iglp_opt`, fragment
prefetch, request reduction, and the 4x4 warp split -- came out neutral or negative, and
the ISA showed in three of those cases that the scheduler had reproduced or re-sunk the
original schedule. The inner loop is 103 instructions for 32 MMAs; at that size the
compiler's scheduling choices dominate any change to what the loop computes.

**Falsifier: an ISA-verified arm whose instruction count and time move together** -- for
example a probe where halving the loads approximately halves their cost. Until one exists,
any attribution of the fp16 gap to a specific loop resource is unsupported, including the
ones this document records above it.

### The isolated probe is not predictive: batched loads lose 4.7% in the real kernel

Batching every B fragment load ahead of the MMA block -- so none is consumed back to
back with its own load -- measured **+3.8%** in the pure probe (8.858 against 9.207 ms,
control spread 2.6%). The same change, gated as ablation 4096 and A/B'd in the production
kernel, measures **-4.7%**:

| arm | median ms | TFLOPS | vs control |
| --- | ---: | ---: | ---: |
| production control A | 8.781 | 41.57 | -- |
| batched B loads (4096) | 9.195 | 39.70 | **-4.7%** |
| batched + decode in K loop (5120) | 9.213 | 39.63 | -4.9% |
| production control B | 8.771 | 41.62 | +0.1% |

With a 0.1% control residual this is not marginal, and the sign has flipped. That is the
methodological result of this section: **the isolated K-loop probe can falsify a hypothesis
but cannot predict a production gain.** The probe reproduces the inner loop's instructions
but not the schedule the compiler produces for the whole kernel, and at 103 instructions for
32 MMAs that schedule is what sets the time. Every probe-based gain in this document -- the
fragment-traffic model, the request-rate model, prefetch, batching -- either failed to
transport or was withdrawn when checked in the kernel.

The practical consequence is that fp16 candidates must be screened by A/B in the production
kernel behind an ablation bit, using the interleaved harness with its duplicated control,
and not in the probe. That harness has now shown a 0.1% residual on a 6-round run, which is
good enough to resolve a 1% effect.

Production is unchanged: the batched path is default-off, and `q4kxl quant equivalence`
passes.

**Falsifier: a probe arm and a production arm of the same change disagreeing in sign.**
One such pair now exists above, so the probe is retired as a screen and kept only as a
source of ISA-level facts.

The remaining scheduler-control lever is also neutral. `iglp_opt` had only ever been tried
at value 0; values 1 and 2 were exposed as ablations 2048/8192 and A/B'd in the production
kernel with the interleaved harness:

| arm | median ms | TFLOPS | vs control |
| --- | ---: | ---: | ---: |
| control A | 8.819 | 41.40 | -- |
| `iglp_opt(1)` | 8.853 | 41.24 | -0.4% |
| `iglp_opt(2)` | 8.842 | 41.29 | -0.3% |
| control B | 8.857 | 41.22 | -0.4% |

All three are inside the control spread, so the hint does nothing for this instantiation
at any value. With that, **every fp16 lever tried since the K-loop ceiling was measured has
come out neutral or negative**: fragment prefetch, stride padding (blocked by the 64 KiB
budget), fragment-request reduction, the 4x4 warp split, batched loads, and `iglp_opt`
1 and 2. The shipped inner loop is what the compiler produces for this data flow, and
perturbing it loses.

### The 4x4 warp split does not transport to the real kernel

The probe's +2.9% was measured on the pure loop. Applied to the production kernel -- same
tile, same BK, same LDS budget, only WM/WN changed -- it is a clear loss:

| arm | median ms | TFLOPS | vs control |
| --- | ---: | ---: | ---: |
| 8x4 control A | 10.320 | 35.38 | -- |
| 4x4 warp split | 11.337 | 32.20 | **-9.9%** |
| 8x4 control B | 10.397 | 35.11 | -0.7% |

The two configurations are bit-identical (`max|diff| = 0` over 35,651,584 outputs), so this
is purely a schedule and occupancy effect, and at 0.7% control spread the -9.9% is solid.

The probe warned, but understated it. Its no-LDS baseline showed 16 warps 2.8% worse at
hiding MMA latency. The probe has no staging, and the real kernel does: at 512 threads a
CTA has half as many warps, so the fetch, the decode and the commit have half as many warps
to hide behind as well. That is the same reason \"staging is nearly free\" was measured at
1024 threads and only holds there. Losing a third of the fragment traffic is worth about 3%;
losing half the warps costs about 10%.

**So 8x4 stands, and this closes the warp-split lever.** The falsifier for any future claim
is the same arm: a 4x4 (or other non-8x4) split beating the production 8x4 kernel on the
full kernel, not just the inner loop.

## The IQ3 formats: where their 15% penalty comes from, and why it is structural

All of the fp16 investigation above was aimed at `kQ4_K`. That was the wrong target: Q4_K is
the *fastest* format and only 11.8% of the 3.84 bpw shard's fp16 FLOP share, while IQ3_S
(34.0%) and IQ3_XXS (25.1%) together are 59% and are the two slowest. ISA-compared at the
production shape (kStore, 256x256, w8n4, BK=4):

| format | K-loop body | vs Q4_K | global | wait | decode VALU | other VALU | measured ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Q4_K | 218 | -- | 9 | 21 | 76 | 14 | 11.97 |
| IQ4_XS | 222 | +2% | 6 | 19 | 83 | 16 | ~11.85 |
| IQ3_XXS | 275 | **+26%** | 18 | 29 | 71 | **48** | ~13.9 |
| IQ3_S | 286 | **+31%** | 17 | 32 | 77 | **52** | 13.72 |

Unlike the intra-Q4_K perturbations earlier in this document, **loop-body size tracks
measured time across formats** -- Q4_K and IQ4_XS are within 2% of each other in both, and
IQ3_S/IQ3_XXS are 26-31% larger and 15-16% slower. The excess is +38 other-VALU, +8 global
loads and +11 `s_waitcnt` per stage. Signedness is not the cause: Q3_K and Q6_K also produce
signed bytes through the same biased-subtract path and are only ~5% slower than Q4_K. The
IQ3-specific cost is the codebook.

`DecodeQuantSub16`'s kIQ3_S path does eight data-dependent table lookups per thread per
stage -- four `kDeviceIq3sGrid[512]` codebook reads and four `kDeviceIq3sSignMask[16]`
reads -- and the dependent chain index -> grid load -> mask load -> xor/add -> store is what
the extra `s_waitcnt`s are waiting on. Both obvious fixes are dead:

* **LDS-resident tables.** 2 KiB for the IQ3_S grid plus the mask table. The block already
  occupies all 64 KiB, so this does not fit -- the same wall that blocked BK=8, the stride
  padding, the fragment prefetch and warp specialisation.
* **Computing the sign mask instead of loading it.** The table is exactly reproducible:
  `m = (s&1) | ((s&2)<<7) | ((s&4)<<14) | ((s&8)<<21); mask = m * 0xFF`, which matches all
  sixteen entries. But it costs about eleven ALU ops against one L1-resident load, and there
  are eight such lookups per stage, so it replaces ~8 instructions with ~88. Worked out
  before implementing; not attempted. The grid table is a 512-entry codebook and has no
  arithmetic form at all.

So the IQ3 penalty is the price of a lookup-based format, and the decoder is already the
compact implementation of it. The remaining IQ3-specific structural item is that IQ3_S is
excluded from the paired gate/up path (`static_assert(WRS % 2 == 0 && Type != kIQ3_S)`), so
its gate/up tensors cost two passes instead of one -- but that role is only 6.6% of the
shard's FLOPs.

**Falsifier: an IQ3_S or IQ3_XXS arm that removes its codebook lookups without adding more
than the ~8 instructions they cost, and beats 13.72 ms.**

## The held-out metric, run: the int4 activation cost is real, reproducible and small

The int4 recommendation above was to build the held-out metric *before* the kernel, because
a self-consistency cosine cannot settle \"is 2.0e-2 acceptable\". That metric already
exists: `qwen27b_target_test <model> <reference>` reports mean KL, total variation, a
teacher-forced NLL delta and top-1 agreement over 27 logit rows on three fixed prefixes,
comparing two GGUFs' full 248,320-token distributions. No BF16 target is on disk, so the
reference is `UD-Q4_K_S` and the candidate is the 3.84 bpw shard; the *differential*
between the last two rows is the measurement, and one model is resident at a time so it
fits the 30 GiB box.

| configuration | mean KL | mean TV | mean NLL delta | top-1 |
| --- | ---: | ---: | ---: | ---: |
| baseline (production paths) | 0.0147376 | 0.050077 | 0.0061615 | 25/27 |
| `GUFO_FORCE_Q8_PREFILL=1` (int8 activations) | **0.0147376** | 0.050077 | 0.0061615 | 25/27 |
| forced q8 + `GUFO_INT4_MIXED=1` | **0.0179338** | 0.055400 | 0.0196409 | **24/27** |

Two results, both exact:

1. **int8 activations cost nothing this instrument can see.** Forcing the q8 path changes the
   logits to six significant figures -- identical, not merely close.
2. **Four-bit activations cost a real and reproducible amount.** mean KL +0.0032 (+21.7%),
   mean NLL delta 3.2x, and one top-1 flip, 25/27 -> 24/27.

**The figure is deterministic.** Run three times, mixed int4 gives 0.0179338 / 24 top-1
every time and the baseline gives 0.0147376 / 25 every time. That resolves the caveat
recorded above: the earlier 0.9794 cosine was never shown to be stable, but this KL
instrument reproduces bit-exactly, so the number may be used.

Reading it: the int4 step is +22% on top of a shard-to-shard difference that is itself
0.0147 with 2/27 top-1 flips. It is not free -- but it is a fraction of the difference
between two legitimate quantizations of the same model, and the sample is only 27 rows, so
the single top-1 flip is one event. Whether that is acceptable is a policy call rather than
a measurement, but it can now be made on a stable number instead of a cosine.

**The scoping caveat matters more than the number.** `GUFO_FORCE_Q8_PREFILL` changes the
result by exactly nothing, which means these fixtures never reach the fp16 prefill path at
all -- they are short teacher-forced rows, so every GEMM runs the q8 route. This therefore
bounds the cost of four-bit activations on the **short-batch path where int4 is currently
implemented**, not on the pp2048 fp16 path that holds 78% of the GEMM time. Nothing here
says what four-bit activations would do to the path where the 2x would actually be taken.
A fixture of at least 1024 tokens is the prerequisite for that, and it does not exist yet.

## The int4 2x is real, and it is coupled to the shard's quantisation mix

The 9% figure above is a property of the shipped shard, not of the int4 path. Weight elements
are proportional to GEMM FLOPs, so the eligible share *is* the FLOP share, and it is set
entirely by how much of the model is stored on a linear grid:

| shard | int4-eligible elements | FFN mix |
| --- | ---: | --- |
| IQ4_XS-3.84bpw | 15.5% | IQ3_S 36.9%, IQ3_XXS 33.3%, IQ4_XS 21.0% -- **all codebooks** |
| UD-Q4_K_S | 25.7% | IQ4_XS 50.3% |
| **UD-Q4_K_S-requant294-Q4K** | **91.2%** | **Q4_K 97.9%**, Q6_K 2.1% |

On the third shard -- which is already on disk -- the ceiling is `1/(0.09 + 0.91/2.2)` =
**1.99x**, not 1.09x. That is the \"quantisation mix\" the section above called the
larger lever, and it exists.

The catch is that the 4-bit *activation* cost scales with the same eligible share, so the
2x and the quality are inversely coupled. Measured on the held-out instrument, two repeats
each, all figures bit-exact:

| shard | eligible | configuration | mean KL | top-1 | mean NLL delta |
| --- | ---: | --- | ---: | ---: | ---: |
| IQ4_XS-3.84bpw | 15.5% | production | 0.0147376 | 25/27 | 0.00616 |
| IQ4_XS-3.84bpw | 15.5% | + mixed int4 | 0.0179338 | 24/27 | 0.01964 |
| requant294-Q4K | 91.2% | production | 0.0066611 | **27/27** | 0.02358 |
| requant294-Q4K | 91.2% | + mixed int4 | 0.0124258 | 26/27 | **0.06523** |

Reading it:

* **The requantised shard beats the shipped one on this instrument** -- 0.0067 mean KL and
  27/27 top-1 against 0.0147 and 25/27, both measured against `UD-Q4_K_S`. Its weight
  requantisation is closer to its source than the 3.84 bpw shard is.
* **Four-bit activations cost far more there, as they must.** +0.0032 KL at 15.5% eligible
  becomes +0.0058 at 91.2%, and the NLL delta nearly triples, 0.0236 -> 0.0652.
* **The metrics disagree, and that is the finding.** On KL, TV and top-1, requant294 with
  4-bit activations (0.0124, 26/27) is still *closer to the reference* than the shipped
  3.84 bpw shard without them (0.0147, 25/27). On teacher-forced NLL -- which is what
  perplexity is -- it is ten times worse, 0.0652 against 0.0062. Twenty-seven rows over
  three fixtures cannot adjudicate that, and the sample is the binding limitation, not the
  instrument.

So the honest state of the int4 direction: **the 2x is reachable, the shard that unlocks it
exists and is better than the shipped one on distribution metrics, and the price is a 4-bit
activation grid whose true-token cost is large enough that the current evaluation cannot
clear it.** The next step is a larger held-out evaluation -- the same instrument over many
more tokens -- before any kernel work, because the two metrics currently point opposite
ways and 27 rows is not enough to break the tie.

### The larger evaluation: the 27-row NLL result does not replicate

The instrument was extended rather than replaced. `qwen27b_target_test` now grows its context
window with the fixture, takes a corpus file (`GUFO_QWEN27B_QUALITY_CORPUS`, with
`..._PROMPT` and `..._ROWS`), and can dump and compare captures
(`--dump <path>` / `--compare <a> <b>`) so two runs of the *same* model under different
environment can be differenced -- necessary because the config knobs are process-global.
The run below uses a 1024-token prompt, which is past the fp16/int8 prefill switch, and 512
teacher-forced continuation rows: **513 logit rows over the full vocabulary**, against 27
before. Shard: requant294-Q4K, 91.2% int4-eligible.

| comparison | rows | mean KL | top-1 | mean NLL delta |
| --- | ---: | ---: | ---: | ---: |
| determinism (`base` vs `base2`) | 513 | **0** | **513/513** | 0 |
| int8 step (`base` vs `int8`) | 513 | 0.000135 | 509/513 | +0.00134 |
| **int4 step (`int8` vs `int4`)** | 513 | **0.003916** | **498/513** | **-0.00110** |

Three things follow.

1. **The instrument is exactly deterministic**, not merely reproducible: two full runs give
   KL 0 and 513/513 top-1. Any nonzero KL here is the change, not the measurement.
2. **The int4 activation grid costs 29x the int8 step.** int8 activations cost 0.000135 mean
   KL and 4 top-1 flips in 513; four-bit activations cost 0.003916 and 15 flips, 2.9% of
   rows. Both are reproducible, so the ordering is real.
3. **The teacher-forced NLL cost does not replicate.** On the 27 handwritten fixtures the
   int4 step appeared to add ~0.04 nats; over 513 rows of real text it is **-0.0011**, i.e.
   no cost and marginally the other way. The earlier NLL figure was a small-sample artifact
   of the 24 labels behind it. This is the second time in this document that a 27-row
   conclusion failed to survive a larger sample.

So the decision is now posed on a solid number, and the two metrics no longer conflict:
four-bit activations perturb the output distribution measurably -- 0.0039 mean KL, about a
quarter of the gap between the shipped 3.84 bpw shard and UD-Q4_K_S -- while leaving the
true-token likelihood, which is what perplexity measures, unchanged. For a 2x on 91% of
prefill FLOPs, that is a materially better position than the 27-row sample suggested, and
it is the first int4 number in this document that rests on a sample large enough to use.

### The shard that enables the 2x, measured: it costs distribution, not likelihood

The 2x is only reachable on a shard whose weights are linear, and the candidate on disk is
requant294. Its quality relative to the *shipped* shard is therefore part of the price of
the 2x, and it had only been measured on the 27-row fixtures. Same instrument, same 513-row
long-prefix run:

| comparison | rows | mean KL | top-1 | mean NLL delta | implied PPL ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| **requant294 vs shipped 3.84bpw** | 513 | 0.02561 | 441/513 | **-6.7e-05** | 1.0000 |
| requant294 vs UD-Q4_K_S | 513 | 0.01191 | 474/513 | -0.00579 | 0.9942 |
| shipped 3.84bpw vs UD-Q4_K_S | 513 | 0.01896 | 442/513 | -0.00572 | 0.9943 |
| int4 activation step, on requant294 | 513 | 0.00392 | 498/513 | -0.00110 | 0.9989 |

Three readings:

1. **requant294 is closer to the reference than the shipped shard is** -- 0.0119 and 474/513
   against 0.0190 and 442/513, both on 513 rows. The 27-row result held up.
2. **Switching to the eligible shard costs distribution but not likelihood.** Against the
   shipped shard it is 0.0256 mean KL with 72 top-1 flips in 513, and a teacher-forced NLL
   delta of -6.7e-05 -- indistinguishable from zero. The two shards predict different
   tokens without either being less likely on the true one.
3. **The int4 activation grid costs about a sixth of the shard switch** -- 0.0039 against
   0.0256, 15 top-1 flips against 72 -- and, like it, no NLL cost. So on the likelihood
   metric the entire int4 package is free, and the dominant term is a shard choice, not the
   activation grid.

**The caveat that limits this: 513 tokens is a thin sample for an NLL delta.** The implied
PPL ratios are near 1 but their standard errors are not printed, and a per-token NLL spread
of order 0.1 over 513 tokens puts the standard error of the mean near 0.004 -- the same
order as the int4 delta itself. So \"no NLL cost\" here means \"below the resolution of
513 tokens\", not \"measured at zero\". A real perplexity run over tens of thousands of
tokens is the acceptance test for the kernel, and it is the next measurement, not this one.

## Conclusion: shipping IQ4_XS-3.84bpw; the int4 2x is not reachable at quality

**Decision: the 3.84 bpw codebook shard stands, and no int4 kernel will be built.** The
reason is structural, not an implementation gap, and it took a sweep of what actually exists
to establish.

### Eligibility is a property of the stored type, and can be read without downloading

The int4 WMMA needs both operands on a *linear* grid, so what matters is each tensor's
stored type. The GGUF header carries that list at the front of the file, so an HTTP Range
request for the first 24 MB classifies a shard without downloading it;
`tools/qwen27b/quant_eligibility.py` does this. Fourteen published quants:

| quant | GB | int4-eligible | dominant types |
| --- | ---: | ---: | --- |
| ISTA GSQ-RCO IQ3_S | 11.77 | **13.0%** | IQ3_S 31.0%, IQ4_XS 20.8%, IQ3_XXS 19.8% |
| UD-IQ3_S | 12.04 | 10.9% | IQ3_S 31.7%, IQ4_XS 17.9% |
| UD-Q3_K_XL | 13.15 | 12.3% | IQ4_XS 34.3%, IQ3_S 30.1% |
| UD-Q2_K_XL / UD-IQ3_XXS | 9.83 / 10.93 | 14.0% | IQ3_XXS, IQ2_S |
| UD-IQ4_XS | 14.25 | 15.3% | IQ4_XS 49.7% |
| UD-Q4_K_XL | 17.56 | 20.9% | Q5_K 42.2%, IQ4_XS 21.5% |
| UD-Q4_K_S | 15.36 | 27.6% | IQ4_XS 43.4%, Q4_K 19.1% |
| UD-Q4_K_M | 16.46 | 29.6% | IQ4_XS 32.8%, Q4_K 27.4% |
| **Q4_0** | **16.06** | **89.5%** | Q4_0 86.9% |
| **Q4_1** | **17.54** | **89.5%** | Q4_1 89.5% |
| shipped IQ4_XS-3.84bpw | 13.08 | 15.5% | IQ3_S 30.3%, IQ4_XS 24.8%, IQ3_XXS 23.5% |

**No Unsloth UD quant exceeds 29.6% and the GSQ-RCO release is 13.0%.** The two families
that lead on quality per byte are both non-uniform codebooks -- that is *how* they get their
quality -- and the names mislead in the same direction: UD-Q3_K_XL is 12.3% linear, not
Q3_K. Only the legacy uniform schemes reach ~90%, and they cost +23% to +34% in bytes
(4.705 / 5.140 bpw against 3.834). RCO was checked because a *rotation*-based method would
be the one escape; it is not one -- Riemannian Constrained Optimization is a bit-allocation
optimizer that chooses the per-layer precision mix under an exact budget and never touches
the grids, so the ISTA release is GSQ's codebook quantiser plus that mix.

### Why no codebook format can be mapped onto int4

Two independent blockers, and they are the same fact seen twice.

1. **The instruction has no indirection.** `wmma_i32_16x16x16_iu4` computes a dot product of
   4-bit operands. A codebook stores an *index*, so one MMA on indices computes a dot
   product of indices, not of values. Correctness needs index -> value to be linear.
2. **The values need more than four bits.** From our own table, `kDeviceValuesIq4Nl` is
   `-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113` -- sixteen
   levels spanning 240. A 4-bit operand holds sixteen *uniformly spaced* values, about
   -8..7 with a per-block scale.

These are one fact because the codebook spends its four bits on an index precisely so the
levels can be non-uniform *and* span eight bits of magnitude. **The indirection is the
quality.** So the property that makes these formats good is the property that makes them
ineligible -- a structural conflict, not a gap in the kernel.

Two escapes were considered and both are dead:

* **Sum of uniform grids.** Any non-uniform grid is a sum of K uniform grids, so K MMAs
  could synthesise it -- but each stored value must first be split into its K uniform codes,
  which is the decode that was being avoided, and the MMA cost scales by K. K=2 already
  lands at 105/2 = 52 TF against fp16's 48-50. K=1 is the linear case.
* **Use int8.** The codebook values *do* fit int8, so index -> int8 lookup plus `iu8` WMMA is
  legitimate -- but `iu8` measures 50.31 TF against fp16's 48.35, about 4%. The 2x exists
  only at four bits, which is exactly what the codebooks do not have.

### What was actually gained, and what is left

Everything in this section is fp16-side and closes with the shipped kernel unchanged. The
durable results are the negative ones: the fp16 K-loop ceiling is measured (MMA-only 47.65-
51.05 TF against ~41 TF shipped), every inner-loop lever is refuted with an ISA reason, the
isolated probe was shown not to predict production and was retired as a screen, the quality
instrument was made deterministic and scaled to 513 rows with a long prompt, and the int4
direction is now closed by construction rather than by budget.

**The one route left open, and it is the opposite of what both leading quant families do:**
a *rotation*-based uniform-int4 quantisation (QuaRot / SpinQuant style). Rotating weights and
activations reshapes the distribution so a uniform 4-bit grid approaches codebook quality
while staying int4-eligible. It is untested here and no such Qwen3.8 shard is published,
which is why the decision stands.

**Falsifier: a published quality-first quant whose header is more than ~30% linear, or a
rotation-based uniform-int4 shard.** Either would reopen the direction.

## Can an operand be shared without LDS? No, and it is an instruction-set fact

The 64 KiB LDS budget is the single constraint behind every blocked fp16 lever (BK=8, stride
padding, two CTAs per CU, warp specialisation, LDS-resident tables). If an operand could be
distributed between warps in registers, all of them would open at once. The capability
survey had already flagged two RDNA lane-exchange instructions as available and unused, so
this was the one remaining hardware question.

**Every lane-exchange instruction exists on gfx1151.** Compiled and encoded, not assumed:

| instruction | status | scope |
| --- | --- | --- |
| `v_permlane16_b32` | encodes | within a 16-lane group, one wave |
| `v_permlane16_swap_b32` | encodes | swaps the two 16-lane halves |
| `v_permlane32_swap_b32` | encodes | swaps the two 16-lane halves |
| `ds_swizzle_b32` | encodes | cross-lane, **routed through LDS** |
| `v_mov_b32_dpp` | encodes | within a row, one wave |

**They are all intra-warp.** That is the whole answer. The A fragment is shared across 4
warps and the B fragment across 8, so the sharing this kernel needs is inter-warp -- the one
thing none of these can do. There is no cross-warp register path on the hardware, and the
only instruction that crosses lanes at full width, `ds_swizzle_b32`, does it *by way of LDS*.

One apparent opening looked worth chasing and is closed. The ISA shows every
`v_wmma_f32_16x16x16_f16` taking three 8-VGPR operands, so the A operand is 16 halves per
lane = 1024 B for a 512 B matrix -- 2x duplication, exactly what `LoadSwizzled` produces by
having lanes 0-15 and 16-31 read the same rows. If the upper half-warp's registers were
ignored the fragment load could drop from two `ds_load_b128` to one.

`tools/qwen27b/mma_operand_scope.hip` answers that, and because it is a correctness question
the answer is exact -- zero the upper half-warp's operands and bit-compare the accumulator:

| case | differing values of 256 |
| --- | ---: |
| baseline (self-comparison) | 0 |
| A upper half-warp zeroed | **128** |
| B upper half-warp zeroed | **128** |
| both | 128 |

So all 32 lanes' operand registers are consumed. The duplication is the layout the
instruction wants, and the distinct bytes read -- 512 B per fragment, six per sub-stage, 48
`ds_load_b128` per warp per stage -- are already the matrix size, which is the minimum.

**Conclusion: the LDS wall is architectural, not budgetary.** LDS is the only inter-warp
operand path, so every scheme that needs to share an operand must pay LDS, and the fragment
feed is already at its data floor. That closes the question that the whole fp16 search kept
running into, and it closes it the right way -- by instruction-set fact rather than by having
run out of ideas.

One residue is left unbuilt and deliberately so: `v_permlane32_swap_b32` could let the two
half-warps read *different* halves of each row, replacing two `ds_load_b128` with one plus a
permute, halving the LDS instruction count at constant traffic. It is not worth building --
the request-count experiment reduced LDS requests 4x and measured *slower*, so request count
is known not to track time here.

**Falsifier: a cross-warp register exchange on gfx1151 that does not use LDS.** The table
above is exhaustive over the instructions the assembler accepts.

## The non-GEMM share of pp2048, profiled: 8%, and it is not waste

With the fp16 GEMM closed, the last unprofiled area was the part of a prefill pass that is
not GEMM. The earlier PMC-based budget put it at ~13%; a direct rocprofv3 profile says 8%,
and it says what it consists of:

```sh
python3 tools/prof/prof.py run -o /tmp/prof13 --stages qwen -- \
  ./build/release/gufo bench -m IQ4_XS-3.84bpw.gguf -p 2048 -n 0 -r 1
python3 tools/prof/prof.py show /tmp/prof13/prof_results.db --passes 5
```

Two passes; pass 2 is the timed prefill (1882 dispatches, 7978 ms of the 8177 ms total).
Whole-trace stage rollup:

| stage | calls | total ms | % |
| --- | ---: | ---: | ---: |
| gemm: quant x fp16 prefill | 974 | 7316.70 | **89.5%** |
| ssm: deltanet recurrence | 144 | 227.63 | **2.8%** |
| gemm: k-quant prefill | 458 | 176.04 | 2.2% |
| attention | 80 | 102.14 | 1.2% |
| ssm: conv1d | 144 | 90.57 | 1.1% |
| norm+fp16 | 256 | 77.91 | 1.0% |
| ssm: post-norm gate | 144 | 57.16 | 0.7% |
| attention: qk-norm+rope+kv | 48 | 42.35 | 0.5% |
| unpack | 48 | 27.53 | 0.3% |
| ssm: deltanet prologue | 288 | 22.62 | 0.3% |
| everything else | -- | ~36 | ~0.4% |

**GEMM totals 91.9% and non-GEMM 8.1%**, so the earlier 13% figure was high. GPU-busy is
8177.56 ms against an 8231.09 ms span -- **idle 0.7%** -- so the pass is not launch or host
bound and there is no overhead to win back there.

The 8% breaks down by subsystem rather than by kernel:

* **SSM / deltanet: 4.9%** (recurrence 2.8%, conv1d 1.1%, post-norm gate 0.7%, prologue
  0.3%). The recurrence is sequential over the sequence, so it is latency-bound by
  construction, and its launch geometry (`blocks=2x48x1 wg=256`, 96 CTAs on 20 CUs) is poor
  but the split it would need is not obvious.
* **Attention: 1.7%** (1.2% + 0.5% for qk-norm/rope/kv).
* **Memory-bound elementwise: ~3%** across norm+fp16, conv1d, post-norm gate and the
  qk-norm/rope/kv fusion. These are the only ones that look improvable on inspection --
  `HalfNorm5120` runs 256 launches at 304 us each over 2048x5120 fp16, roughly 132 GB/s of
  read plus write -- but together they are small.

**Verdict: the non-GEMM area is not a target.** It is 8%, it is dominated by legitimate
sequential SSM recurrence and attention work, and the plausibly-recoverable part is 1-3% of
prefill. That is smaller than the 15-20% the GEMM itself is leaving, and the GEMM side is
closed by instruction-set fact rather than by budget.

**Falsifier: a non-GEMM stage above 3% whose kernel is not latency-bound by construction.**
The deltanet recurrence and the attention kernel both are, and nothing else exceeds 1.1%.

## The gate/up epilogue is free, and the FFN is IQ3_XXS rather than Q4_K

Two measurements closed the last open explanation for a ~28% gap between the model-level
GEMM rate and the plain-store bench. Both results were negative, which is why they matter.

**The epilogue costs nothing.** One ablation bit (`Ablate & 8` disables the store in every
epilogue branch) is the only difference between the arms, at real FFN geometry --
m=17408, k=5120, batch=2048, paired `kGateUp`:

| arm | median | TFLOPS |
|---|---|---|
| gate/up full | 17.995 ms | 40.57 |
| gate/up no-epilogue | 17.972 ms | 40.63 |
| store full | 9.184 ms | 39.75 |
| store no-epilogue | 9.129 ms | 39.99 |
| gate/up full (control B) | 17.988 ms | 40.59 |

The epilogue is **+0.13%**, 0.023 ms of 18 ms, against a **0.04%** control residual -- the
tightest control this work has had. The hypothesis that the SwiGLU epilogue explained the
gap is dead, and the paired kernel itself is fine at 40.6 TF on production geometry.

**The remaining explanation is the weight format, and the shard says so.** Parsing the GGUF
header of the shipped 3.84 bpw shard shows the FFN is **not** Q4_K, which is what the first
bench had assumed.

**Corrected:** this paragraph originally said every `ffn_gate`, `ffn_up` and `ffn_down` in
all layers is **IQ3_XXS** "with a few IQ2_XXS/IQ2_XS/IQ4_XS outliers". That is wrong. A
byte-weighted census of the FFN tensors alone -- 195 tensors, the shard has 65 blocks
(blk.0..blk.64) and every one carries gate/up/down -- gives **seven** types, 17.380 G weights,
7.05 GiB, 3.484 bpw effective:

| type | FFN weights | FFN size | bpw |
|---|---|---|---|
| IQ3_S | 6.417 G | 2.57 GiB | 3.438 |
| IQ3_XXS | 5.793 G | 2.07 GiB | 3.062 |
| IQ4_XS | 3.654 G | 1.81 GiB | 4.250 |
| Q3_K | 0.713 G | 0.29 GiB | 3.438 |
| Q4_K | 0.446 G | 0.23 GiB | 4.500 |
| IQ2_XXS | 0.267 G | 0.06 GiB | 2.062 |
| IQ2_XS | 0.089 G | 0.02 GiB | 2.312 |
| **total** | **17.380 G** | **7.05 GiB** | **3.484** |

IQ3_XXS is 33.3% of FFN weights; IQ3_S is the largest single type at 36.9%. These percentages
match the whole-shard figures already recorded for this shard earlier in this document (IQ3_S
36.9%, IQ3_XXS 33.3%, IQ4_XS 21.0%) -- the two paragraphs disagreed and this one was wrong. The
conclusion survives, because IQ3_XXS is the most expensive decode in the ISA study (+26% loop
instructions over Q4_K) and is a third of the FFN, which lines up with the 27.7% gap the
profiler recorded. But any repack or kernel-coverage claim has to cover all seven types, and
the FFN is 65 x 3 tensors rather than 48 x 3.

Compounding it, `TryLaunchBatchedDualQuantGEMMSwiGLUFp16` has **no IQ3_XXS case at all**,
so the FFN never reaches the paired kernel; it falls back to two separate `kStore` launches
with `Complete=false`.

A first format A/B at batch 512 (m=17408, k=5120) gave Q4_K 2.495 ms vs IQ3_XXS 2.565 ms,
only **+2.8%** rather than the +26% the instruction counts predict. At low batch the shape is
weight-bandwidth-bound, so decode cost hides behind the 50 MB weight stream.

**Falsifier: the same A/B at batch 2048, which was not completed.** Near +25% means the gap
is the IQ3_XXS decode and the lever is that decoder; near +3% means the gap is elsewhere and
the format hypothesis is dead too.

## The NPU is real: 57 TOPS of MAC issue, and every GEMM captures 15% of it

The question was whether prefill can be split across the XDNA2 NPU and the iGPU. The
short answer is that the NPU's *hardware* is a peer of the iGPU, that the open stack
reaches ~15% of it for every datatype, and that the 15% is **not** the operand feed.

### Getting the NPU to run at all

The device was enumerating but every XRT command died on
`mmap(len=64 MiB, offset=4 GiB) -> EAGAIN`. That is the NPU BAR, and the cause is
`RLIMIT_MEMLOCK`, which is 8 MiB soft *and* hard here against a 64 MiB mapping. The
persistent fix is `/etc/security/limits.d/xrt.conf` with `@render soft/hard memlock
unlimited` plus re-login; running under root with `ulimit -l unlimited` is enough for a
benchmark. It is not a driver or firmware bug, and `force_iova=1` does not help.

Toolchain: mlir-aie **v1.4.3** plus Peano, both prebuilt and both matching this box --
the wheel is `cp314` and the system Python is 3.14.7. `utils/env_install.sh` hard-requires
python3.12 and must be bypassed with a manual venv. The ONNX/VitisAI path is structurally
dead on Linux (the `voe` package does not exist for Linux x86_64; AMD's Ryzen AI release
is Windows-only), so pyxrt or IRON are the only routes.

### The first pass measured a kernel and called it a ceiling

The stock whole-array GEMM and the TileFuse W4A16 kernel, at real FFN geometry:

| workload | shape | time | throughput | check |
|---|---|---|---|---|
| W4A16 int4 wts, in-core dequant | 17408x5120x2048 | 57.97 ms | **6.30 TOPS** | PASS |
| W4A16 calibration | 2048^3 | -- | 5.93 TOPS | PASS |
| i8 whole-array, 64^3 inner tile | 2048^3 | -- | 8.74 TOPS | PASS |
| bf16 via bfp16 | 2048^3 | -- | 5.81 TFLOPS | PASS |

Taken at face value that is 6x slower than the iGPU and yields a ~1.14x split. **That
reading was wrong**: it is a *kernel* number, exactly the error already corrected once on
the GPU side when the fp16 WMMA ceiling had to be measured separately from the kernel.

### The MAC-array roofline

Register-resident operands, no DMA, `mac_8x8_8x8` issued back-to-back with independent
accumulators (`tools/qwen27b/npu_peak_mac.py`):

| datapath | per-tile | 32 tiles | note |
|---|---|---|---|
| int8 `mac_8x8_8x8` | 1.78 TOPS | **56.9 TOPS** | 1.07 MAC/cycle at ~1.62 GHz |
| bfp16 (`bf16` -> bfp16 + `mac_8x8_8x8T_conf`) | 1.21 | **38.9 TOPS** | conversion is inside the loop, so a lower bound |
| native bf16 (`2x mac_4x8_8x8_bf16`) | 0.089 | 2.8 TOPS | 14x slower; avoid this path |

1.07 int8 MACs/cycle/tile is the architectural 1-per-cycle issue rate, so 32 tiles at
~1.62 GHz gives ~57 TOPS -- matching AMD's 50 TOPS figure rather than exceeding it. The
mlir-aie maintainer confirms the model on issue #2735: "the hardware can execute r*s*t
MACs in a single cycle ... since r, s, t are intrinsic tile sizes."

Two practical traps: **native bf16 is 14x slower than routing bf16 through bfp16**, and
the bf16 kernel needs **4224 bytes of stack** against the 1024-byte default, so it will not
build without `Worker(stack_size=8192)`.

### The finding: the capture is datatype-independent

| GEMM | achieved | its datatype's peak | capture |
|---|---|---|---|
| i8 whole-array | 8.74 TOPS | 56.9 | **15.4%** |
| bf16 via bfp16 | 5.81 TFLOPS | 38.9 | **14.9%** |
| W4A16 (bfp16 datapath) | 5.93 TOPS | 38.9 | **15.2%** |

Three datatypes, three kernels, the same ~15%. A datatype-independent plateau is a
dataflow signature, not a silicon limit. Note also that **NPU bfp16 (38.9 TOPS) is at
parity with iGPU fp16 (40.6 TF measured in the FFN kernel)**, so the two engines are
peers and the Amdahl ceiling on prefill is 1.84x rather than 1.14x.

### The 15% is not the operand feed

The obvious suspect was L1 operand bandwidth, since a GEMM must load operands per MAC.
Measuring the int8 MAC rate at 0, 1 and 2 vector L1 loads per MAC
(`tools/qwen27b/npu_peak_feed.py`):

| L1 loads per MAC | 65,536 MACs/call | 262,144 MACs/call |
|---|---|---|
| 0 (register-resident) | 10.84 | 28.43 |
| 1 | 51.75 | **53.92** (95% of peak) |
| 2 | 35.76 | 37.30 (66% of peak) |

**One L1 vector load per MAC is essentially free** -- 95% of peak. The 1- and 2-load arms
are *flat* across a 4x change in work per call, while the 0-load arm rises 2.6x over the
same span: per-call cost is exactly what the streaming arms are insensitive to, and they are
already amortised at the work the GEMM actually does (a 64^3 tile is 262,144 MACs per
kernel call). The 0-load arm is therefore not a valid reference at small work per call --
it measures per-call overhead, not MAC rate.

The whole-array GEMM performs exactly one operand load per MAC, so its operand feed is not
what caps it at 15%. That eliminates the intuitive explanation and moves the target to the
two things this probe deliberately excludes: the **C-accumulator round-trip to L1** (the
stock kernel loads and stores its four accumulators every k-chunk, 8 extra memory ops per 4
MMULs) and the **ObjectFIFO lock acquire/release plus DMA handshake per k-chunk**, which the
stock design pays once per 512-MMUL call.

### The 15% is a tiling limit, and it has been beaten: 30.6 TFLOPS verified

The falsifier below was met by an in-tree example. **Asymmetric tile buffering (ATB)**
(`programming_examples/ml/block_datatypes/gemm_asymmetric_tile_buffering`, after arXiv
2511.16041) decouples the L1 buffering of the A input from the C output. A row of A is dead
once its C row has finished one reduction, while a C row must stay live for the whole
reduction, so forcing both to share one `T_M` pays the peak buffer cost twice. With
`rho = T_MC / T_MA = 4` (`T_MA = 32`, `T_MC = 128`) a 128x64x128 tile fits in the 63 KB
L1 that would need ~91 KB symmetric. The kernel makes register placement explicit with
`__aie_dm_resource_a/b/c/d` DM-bank annotations.

Measured on this chip (Strix Halo, 8 columns, mlir-aie 1.4.3 + Peano):

| config | precision (A / B / C / accum) | L1 tile | NPU time | throughput | check |
|---|---|---|---|---|---|
| config3 | bfp16 / bfp16 / bfp16 / bf16 | 128x64x128 | 2244.6 us | **30.6 TFLOPS** | **PASS** |
| config1 | bf16 / bfp16 / bf16 / bf16 | 128x64x128 | 2895.0 us | 23.7 TFLOPS | FAIL |

The paper reports a 4.54x speedup, from 4.8 to 24.6 TFLOPS. **Config3 reaches 30.6 TFLOPS
here -- 79% of the 38.9 TFLOPS bfp16 MAC peak** -- where the stock whole-array tiling reaches
~15% of that same peak. So the ~15% capture is a *tiling* artifact, not silicon and not the
datatype, which is exactly what the datatype-independent plateau predicted.

Config1 fails its correctness check because it **accumulates in bf16** across K=4096; that is
lossy by construction, not a harness fault. Config3's pure-bfp16 path passes the stochastic
verification, so 30.6 TFLOPS is a verified GEMM result rather than a throughput-only number.

**But it is shape-specific, and our K is not its shape.** Overriding `-M -K -N` produces
incorrect results at every shape tried except the paper's. Holding M=4096 and N=2048 fixed
and sweeping only K:

| K | result |
|---|---|
| 1024 | FAIL (719/1000), 25.8 TFLOPS |
| 4096 | **PASS**, 30.6 TFLOPS |
| 5120 | FAIL (888/1000), 31.5 TFLOPS |
| 8192 | FAIL (511/1000), 31.7 TFLOPS |

At the two real FFN shapes (M=2048 K=5120 N=17408 and M=2048 K=17408 N=5120) it runs at 32.4
and 32.7 TFLOPS but fails verification outright. Isolating the variable: **N=17408 is fine**
-- M=4096 K=4096 **N=17408 PASSES at 32.4 TFLOPS** -- and M=2048 was never implicated.
**Only K=4096 passes.** The test initialises A and B to all ones, so every C[i,j] must equal
exactly K; the failing runs return near zero, which is a structural plumbing failure rather
than accumulated rounding.

So ATB proves the *capability* but is not a general GEMM library: it is a research artifact
validated at one shape. Bringing it to our K=5120 means work on the open-source kernel --
most likely the B-tile DMA loop counts or the shuffle's K stride -- and a K-split does not
dodge it, because K=1024 fails too.

**The K dependence is a hardcoded constant, and the fix is five lines.**
`config3/mm_bfp_mixed.cc` declares `constexpr int K_Problemsize = 4096;` and its own comment
states the limitation plainly: *"the bf16 -> bfp16 flush happens once per output tile, after
exactly (K_Problemsize / k) * DIV matmul calls; this kernel only supports K =
K_Problemsize."* The flush fires at `(4096/64)*4 - 1 = 255` calls, so at K=5120 the design
issues 320 and the kernel converts C a quarter of the way through the reduction. Nothing else
was wrong: diffing the generated MLIR for K=4096 against K=5120 shows only memref sizes, BD
lengths/offsets and loop bounds changing, all scaling correctly.

Making the constant overridable and passing the design's K through fixes it exactly:

```c
#ifndef K_PROBLEMSIZE
#define K_PROBLEMSIZE 4096
#endif
constexpr int K_Problemsize = K_PROBLEMSIZE;
```

```python
kernel_flags = [f"-I{_AIE_KERNELS_INC}", f"-DK_PROBLEMSIZE={K}"]
```

| shape | before | after |
|---|---|---|
| 4096x5120x2048 | FAIL (888/1000) | **PASS, 31.4 TFLOPS** |
| 2048x5120x17408 (gate/up) | FAIL (872/1000) | **PASS, 32.4 TFLOPS** |
| 2048x17408x5120 (down) | FAIL (828/1000) | **PASS, 32.9 TFLOPS** |

So the NPU runs our exact FFN gate/up GEMM correctly at 32.4 TFLOPS -- 83% of the 38.9
TFLOPS bfp16 MAC peak and 80% of the iGPU's 40.6 TFLOPS on the same shape. The "general-K
alternative" below is moot once this lands, but it is kept as the fallback measurement.

Diagnostic note worth keeping: the harness initialises A and B to **all ones**, so C must
equal K exactly and *any permutation of A or B is invisible*. That is why these failures read
as near-zero output rather than misplaced data -- and it means the test cannot detect layout
bugs at all. A non-uniform instrument would be strictly better.

**The general-K alternative is correct-ish but an order of magnitude slower.** The stock
symmetric mixed example (`whole_array_mixed`, the one the ATB README points newcomers to)
does accept K=5120:

| design | shape | throughput | check |
|---|---|---|---|
| whole_array_mixed, 64^3 tiles | 4096x5120x2048 | 3.89 TFLOPS | FAIL (38/1000) |
| whole_array_mixed, 64^3 tiles | 2048x5120x4096 | 3.85 TFLOPS | FAIL (44/1000) |
| ATB config3, 128x64x128 | 4096x5120x2048 | 31.5 TFLOPS | FAIL (888/1000) |
| ATB config3, 128x64x128 | 4096x4096x17408 | 32.4 TFLOPS | PASS |

Neither serves our K=5120 today: the fast design is correct only at K=4096, and the
general-K design sits at 10% of the bfp16 peak. Unlike ATB's near-zero outputs, the
`whole_array_mixed` failures are few (38-44 of 1000) and look like bf16 accumulation
precision rather than broken plumbing.

**Our output width hits a third, independent wall.** N=17408 does not even compile for
`whole_array_mixed`: `'aie.dma_bd' op Stride 3 exceeds the [1:1048576] range`. That is the C
write-back stride `m * n_aie_rows * N` against the 2^20 descriptor limit, so at m=64 the
output width is capped at 4096 and N=17408 needs five launches (four of 4096 plus one of
1024). N-splitting is algebraically free -- output columns are independent -- but it is five
launches per FFN tensor instead of one.

### Concurrency: the iGPU barely notices, the NPU loses 21%

The split only pays if both engines run at once. Measured simultaneously -- gufo prefill
pp2048 on the iGPU while ATB config3 runs on the NPU -- with SoC power sampled throughout:

| phase | GPU (pp2048) | NPU (ATB config3) | SoC power mean / max |
|---|---|---|---|
| GPU solo | 497.80 +- 5.54 tok/s | -- | 73.2 / 95.9 W |
| NPU solo | -- | 31.1 TFLOPS | 30.3 / 61.9 W |
| concurrent | **491.13 +- 10.12 tok/s (-1.3%)** | **24.6 TFLOPS (-20.8%)** | 61.9 / 93.0 W |

The iGPU is essentially unperturbed (-1.3%, inside 2 sigma) and the NPU gives up 21%. The
concurrent ceiling (93.0 W) is no higher than the GPU-solo ceiling (95.9 W), so the SoC is
power-capped and the budget is being split: the iGPU holds its allocation while the NPU is
squeezed. This is a power split, not memory-bandwidth contention. The NPU is also the more
efficient engine here by a wide margin -- 31 TFLOPS for ~30 W against the iGPU's ~40 TF for
~73-96 W.

Recomputing with the *concurrent* NPU rate: balance at t = 24.6/(40.6+24.6) = 0.377, giving
1.61x on the split GEMM work rather than 1.65x. The 21% derate costs about a tenth of the
projected gain, which is worth paying.

**Corrected:** the prefill figure above was given as ~1.53x, and the one at the end of this
section as ~1.68x. Both implicitly assume the split covers essentially all of prefill's GEMM
work. It does not -- the plan splits the **FFN**, which the model-level accounting in this
document puts at **67% of GEMM FLOPs**. Amdahl then gives `1/(0.33 + 0.67/1.61) = **1.34x**`
here and `1/(0.33 + 0.67/1.781) = **1.42x**` for the 31.5 TF case. The 1.61x and 1.78x GEMM
figures are correct; only their translation into prefill was wrong.

Caveat: pp2048 read 497.80 tok/s in this run against the 543.93 recorded baseline, an 8.5%
cross-session drift, so absolute numbers from different sessions are not comparable. Solo
versus concurrent *within* this run is.

### Where this leaves the split

The NPU is a peer of the iGPU for the FFN GEMM, and the two land within 25% of each other:

| engine | FFN-class GEMM, measured |
|---|---|
| iGPU, fp16, production shape | 40.6 TFLOPS |
| NPU, bfp16, ATB config3 | 30.6 TFLOPS |

A row split of the paired gate/up GEMM balances at ~43% of the work on the NPU, worth ~1.75x
on the GEMM and, with GEMM at 91.9% of prefill, **~1.65x on prefill overall -- pp2048 ~898
tok/s against the current 543.93**. That is larger than every remaining GPU-side lever
combined, and it is now a measured capability rather than a projection.

What is *not* done is the integration, and it is real work: weights must be repacked into the
bf16 / AWQ-style tile layout the kernel reads (the NPU cannot read IQ3_XXS codebooks, and the
in-core dequant runs on the AIE tiles), the host must orchestrate two engines with a barrier
per FFN per layer, and bfp16's block exponent needs its own quality measurement on the shard.
None of those is a research question; they are engineering.

**Caveat on the bfp16 path.** bfp16 is block floating point with a shared exponent per 8
elements, which is why the W4A16 PASS criterion is `err / (|A||B|)` rather than plain
relative error. It is float and can consume fp16-class activations directly, so it needs no
int8 activation quantisation -- but the block exponent must be validated on real weights.

## The FFN weight format costs 4%, so the 28% gap is in the non-FFN GEMMs

The last standing explanation for the gap between the model-level effective 29.7 TF and the FFN
kernel's 40.6 TF was the FFN's weight format. The shipped shard runs IQ3_XXS there, and the ISA
study put IQ3_XXS at +26% loop instructions over Q4_K. The earlier batch-512 A/B showed only
+2.8%, but at low batch the shape is weight-bandwidth-bound, so decode cost hides. At
production batch (`tools/qwen27b/fp16_format_ab.hip`, m=17408 k=5120, Q4_K vs IQ3_XXS, both
`Complete=false`):

| arm | batch 512 | batch 2048 |
|---|---|---|
| Q4_K store (mean of 2 controls) | 2.569 ms | 10.137 ms |
| IQ3_XXS store (mean of 2 controls) | 2.624 ms | 10.561 ms |
| **IQ3_XXS cost** | **+2.1%** | **+4.2%** |
| Q4_K paired gate/up | 4.682 ms (38.98 TF) | 18.140 ms (40.25 TF) |

The control spread is ~1%, so +4.2% is real but small. **The format hypothesis for the 28% gap
is dead.**

Two consequences follow.

**1. The gap is in the non-FFN GEMMs.** The FFN is ~47% of the model's parameters (12.83e9 of
27.32e9). If it runs at 40.6 TF and the model-level rate is 29.7 TF, then the attention and SSM
projection GEMMs must be running at roughly **24 TF**. Bringing those to the FFN's rate would
be worth **~1.37x on prefill** -- GPU-only, no new runtime, no quantisation risk, and larger
than anything else outstanding. That is the next measurement.

**2. A smaller cheap win: the FFN gate/up never uses the paired kernel.** The production shard's
gate and up are both IQ3_XXS, and `TryLaunchBatchedDualQuantGEMMSwiGLUFp16` has no IQ3_XXS case,
so it falls back to two separate `kStore` launches: 2 x 10.561 = 21.12 ms where the paired Q4_K
kernel does both in 18.14 ms. Adding IQ3_XXS -- and the IQ3_XXS/IQ2_XXS mixes the shard actually
contains -- to that dispatch list is a few percent on its own.

**Caveat recorded as a warning, not a result.** The `iq3xxs no-epilogue` arm reads 9.619 ms
against 10.561 ms with the epilogue, i.e. -9%, while for Q4_K the same ablation measured the
epilogue at +0.13%. That arm is almost certainly dead code: with the store ablated the compiler
can drop the IQ3_XXS decode entirely, which is the same DCE trap hit earlier in this work. No
conclusion is drawn from it, and per-format ablation arms need their own ISA verification.

## The non-FFN shapes are not slow -- and a 26% harness discrepancy is open

The shape sweep at production batch (prefill_fp16_bench, q4k, batch 2048) is uniform:

| shape | m | k | TFLOPS |
|---|---|---|---|
| FFN gate/up | 17408 | 5120 | 30.01 |
| attn_qkv | 10240 | 5120 | 30.05 |
| attn_q | 12288 | 5120 | 29.69 |
| attn_gate | 6144 | 5120 | 29.42 |
| ssm_out | 5120 | 6144 | 29.49 |
| attn_output | 5120 | 6144 | 29.47 |
| attn_k/v | 1024 | 5120 | 30.88 |

Uniform to within 5%, and ~30 TF agrees with the wall-clock model-level 29.7 TF. So the
inference that the non-FFN GEMMs run at ~24 TF was **wrong**: there is no slow subsystem. The
arithmetic behind it was bad -- the FFN is 67% of GEMM FLOPs (the embedding is a gather, not a
GEMM), not the 47% the parameter count suggested, and at 67% the model-level rate is fully
explained by a uniform ~30 TF.

**But two harnesses disagree by ~26% on the same kernel and shape**, reproducibly and
back-to-back:

| harness | ms | TFLOPS |
|---|---|---|
| prefill_fp16_bench, production launcher | 12.13 - 12.40 | 29.4 - 30.1 |
| prefill_fp16_bench, tile variant 8 (same instantiation) | 12.26 | 29.77 |
| fp16_prod_ab, direct launch | 9.23 - 9.46 | 38.6 - 39.7 |
| fp16_prod_ab, production launcher | 9.19 - 9.47 | 38.6 - 39.7 |

Ruled out as causes: the production launcher (9.468 vs 9.460 ms against a direct launch of the
same instantiation *in one binary*); the `Complete` flag (12.26 vs 12.90 ms *within* the slow
harness, correctly ordered); data content (NaN-filled weights cost 4.5%); sustained load (1
iteration per rep still reads 12.0 ms, and 12 rounds in the fast harness still reads 9.2 ms).
What remains is the allocation set -- the slow harness additionally allocates a 42 MB `fp32_x`
staging buffer and fills weights via `FillWeights` instead of a memset.

**Recorded as unresolved: no absolute TFLOPS from either harness should be trusted until this
is settled.** The relative results in this document survive, because both harnesses order
conditions consistently, but the iGPU side of the split is not sized. Note the direction if the
slow harness is right: at ~30 TF the iGPU would be *slower* than the NPU's 32.4 TF, making the
NPU the faster engine and the split worth more rather than less.

**Update: the discrepancy is real and eleven candidate causes are excluded.** rocprofv3
kernel-trace on both binaries:

| binary | kernel GPU time | wall |
|---|---|---|
| prefill_fp16_bench | **11.663 ms** (n=19) | 12.1 ms |
| fp16_bench_ab | **9.659 ms** (n=40) | 9.45 ms |

The same instantiation genuinely executes 21% slower in one process -- the wall clock is not
lying. Excluded, each by direct measurement:

- the production launcher (9.468 vs 9.460 ms against a direct launch, one binary)
- the `Complete` flag (correctly ordered *within* each harness: 12.26 vs 12.90 ms)
- the include set (replicated exactly, same order -> still 9.45 ms)
- `DeviceBuffer` (same class, same `hipMalloc`)
- data content (zero, valid Q4_K scales, NaN scales, non-zero activations -> 4.7% spread)
- an extra 42 MB allocation
- the slow harness's exact allocation set *and order* (`w, fp32_x(42MB), y, half_x` -> 9.2 ms)
- the 142 MB device-to-host readback before timing
- sustained load (1 iteration per rep still reads 12.0 ms)
- register allocation (VGPR counts do not track the fast/slow split)
- the bench code path (main -> RunFp16 is trivial)

The remaining candidate is that the same template instantiation is compiled to **different
machine code** in the two translation units. Confirming that means dumping and diffing SASS,
which was not done.

**Practical impact.** The iGPU's real FFN GEMM rate is ~30 or ~39 TF depending on which
harness matches the model, and the model-level wall-clock rate of 29.7 TF is independent
evidence for ~30. Note the direction if the slow harness is the truthful one: the NPU at
32.4 TF would be the *faster* engine, and the split worth more than the 1.34x prefill figure
computed from 40.6 TF. Either way the NPU conclusion is unaffected.

**Resolved: it is the GPU boost clock, and the clock is data-dependent.** Reversing the run
order removes ramp-up as an explanation -- each binary keeps its own clock regardless of when
it runs:

| binary | time | steady-state clock | TFLOPS/MHz |
|---|---|---|---|
| fp16_bench_ab (zeroed operands) | 9.645 ms | 2622 MHz | 0.01443 |
| prefill_fp16_bench (real weights) | 11.845 ms | 2141 MHz | 0.01439 |

Efficiency per clock agrees to **0.3%**, so there is no code discrepancy at all: the 26% is
entirely frequency. This part has three SCLK levels (600 / 1408 / 2900 MHz) and never reaches
the top one here. Sweeping the weight pattern inside prefill_fp16_bench moves the settled clock
from 2193 MHz (pattern 0) to 2060 MHz (pattern 2), with the time following, so the clock is
data-dependent: zeroed operands toggle fewer bits, draw less switching power, and let the part
boost ~20% higher.

**Consequence: every hand-written bench in this document that zeroes its operands is inflated
by that boost**, including the 38-40 TF figures for the FFN kernel and the 40.6 TF used to size
the split earlier. The representative number is the one measured with real weight data:
**~31-32 TF at the clock real data induces**, which is also what the model-level wall-clock rate
of 29.7 TF implies. There is no unclaimed 20-30% of iGPU headroom -- the observation that
started this hunt was an artifact of synthetic data, and so was the "non-FFN GEMMs are slower"
inference before it.

The same caveat applies to the NPU's 32.4 TFLOPS, whose ATB harness initialises A and B to all
ones. Whether the NPU's clock is data-dependent was not measured, so the two engines are not
yet compared on equal footing.

Recomputing the split with the corrected iGPU rate (~31.5 TF under production conditions)
against the concurrent NPU rate (24.6 TF): balance at t = 24.6/(31.5+24.6) = 0.438, giving
1.78x on the split GEMM work. On this evidence the NPU is at *parity* with the iGPU rather than
behind it.

**Corrected:** the prefill figure here was given as ~1.68x. With the FFN at 67% of GEMM FLOPs,
Amdahl gives `1/(0.33 + 0.67/1.781) = **1.42x**`. 1.68x would require the split to cover 92.5%
of prefill -- every GEMM the model runs, not just the FFN -- and that is not affordable: the
non-FFN weights are another ~5.13 GiB, which at 9 bits/weight would add ~13 GiB packed on top of
the 18.21 GiB the FFN alone needs. **The projection is ~1.42x on prefill, and it remains a
projection**: no run has ever had the NPU computing part of the model.

### The NPU throughput is data-independent, but the ATB correctness test is not a real test

The same synthetic-data concern that inflated the iGPU numbers could have applied to the NPU's
32.4 TFLOPS, whose ATB harness also uses degenerate data. It does not. Replacing the harness's
deliberately permutation-invariant pattern (B all ones, A in {1,2} -- documented in the source
as "so the host-side layout shuffle is permutation-invariant on the data") with signed
mixed-magnitude operands:

| operands | throughput | result |
|---|---|---|
| stock (all ones / {1,2}) | 30726 GFLOPS | PASS |
| A real, B = 1 | 28216 | FAIL 1000/1000 |
| A = 1, B real | 30934 | FAIL 1000/1000 |
| both real | 30713 | FAIL 909/1000 |

Throughput is **unchanged** -- the FFN gate/up shape reads 11262 us with real operands against
11258.8 us with all ones, and 4096x4096x2048 reads 2230 us against 2211 us -- so the NPU's
number is not a synthetic-data artifact. That question is closed.

**But correctness is now open, and it matters more.** Every non-uniform variant fails,
including the shape the stock test passes. The stock pattern cannot detect layout errors *by
construction*: its own comment says the values are constant within every L1 sub-tile precisely
so the shuffle is invisible. "PASS" therefore never established that the A shuffle, the B
shuffle and the C un-shuffle are correct -- only the accumulation count and the bfp16 writeback
path.

It is not yet determined whether the failure is a layout bug in the ATB pipeline or an
assumption in the host's reference path, because the vendored example was *modified* rather
than diagnosed. Either way the conclusion for the split is the same: **the ATB kernel's
correctness on real weights is unverified** -- SUPERSEDED, see "Resolved: the ATB correctness
failure was the host's tile geometry" below; the cause was a missing `-w/-y/-z` host argument, and that is a prerequisite for the repack rather
than an afterthought. The repack can be written, but it cannot be called validated until a
layout-sensitive test passes.

This is the third time in this work that a degenerate test pattern has hidden the real answer:
the fp16 ablation arms that were optimised away, the all-ones GEMM that hid the K dependence,
and now the all-ones GEMM that hid the layout question.

## The bfp16 quality gate passes on real weights

The one untested risk in the NPU path was bfp16's block exponent. The B operand is stored as
8-element groups sharing one exponent with 8-bit magnitudes (the intrinsic carries the signs
as a separate vector), so a group whose values span a wide dynamic range loses mantissa bits.
The ATB harness initialises A and B to all ones, which exercises none of that.

Measured on real weights -- the first 512 rows of `blk.0.ffn_gate.weight` (IQ3_XXS, 2,621,440
weights) taken straight from the shipped shard, dequantised with the production host decoder
and round-tripped through the bfp16 model (`tools/qwen27b/bfp16_quality.cpp`):

| magnitude bits | RMS abs err | RMS / ||w|| | max rel |
|---|---|---|---|
| 7 | 6.36e-05 | **0.62%** | 11.1% |
| 8 | 3.18e-05 | **0.31%** | 5.9% |

The block exponent adds **0.3-0.6% RMS** to weights whose own quantisation is a ~3-bit
codebook. In quadrature that is a rounding error on the existing error -- a 3% RMS quantiser
plus 0.5% becomes 3.04% -- so the conversion is numerically close to free. The 8-bit reading is
the relevant one, since the sign is carried separately; the 7-bit figure is the conservative
bound if the signs turn out to share the magnitude field.

Note the weight statistics: max|w| = 0.122, RMS|w| = 0.0103. These are small-magnitude
weights, which is exactly the regime where a *shared* exponent is most demanding, so the test
is not an easy one.

All three preconditions for the NPU path now hold on measured evidence: the GEMM is correct at
our geometry (32.4-32.9 TFLOPS, CPU-reference PASS), it runs concurrently with the iGPU at a
1.3% cost to the latter, and the format conversion is numerically free. What remains is
engineering -- the repack, the orchestration, and the row-split on the iGPU side.

## Resolved: the ATB correctness failure was the host's tile geometry

The section above concluded that the ATB layout could not be trusted, because every
data-dependent probe failed. **That conclusion was wrong, and the cause was our own
invocation, not the design.**

`gemm_atb_bfp_test.cpp` takes the L1 tile shape as CLI options -- `-w` (m), `-y` (k), `-z`
(n) -- and defaults all three to **64**. config3 is compiled for `m=128, k=64, n=128`. Every
run so far omitted those flags, so all three layout functions were handed the wrong tile
shape:

| layout function | arguments | value used | correct value |
|---|---|---|---|
| `layout_A_L1_2x1_8x8block` | L1_block_m, L1_block_k | 64, 64 | **128, 64** |
| `layout_transpose_L1_1x2_8x8block` | L1_block_k, L1_block_n | 64, 64 | **64, 128** |
| `layout_inverse_C_L1_2x2_8x8block` | L1_block_m, L1_block_n | 256, 64 | **512, 128** |

The stock all-ones pattern cannot see this, and says so itself: its comment in the test
explains the values are constant within every L1 sub-tile "so the host-side layout shuffle is
permutation-invariant on the data". That is why it passed while every other probe failed, and
why the `ATB_CSET` sweep *appeared* to identify `(4m, n)` as the C tile: it was re-deriving a
tile from the 64-based geometry, which made a bookkeeping error look like a device property.

The correct invocation leaves the `ATB_CSET` / `ATB_ABLK` defaults alone (they are already
`(n_aie_rows*m, n)` and 1) and only supplies the geometry:

```
./atb_bfp_test.exe -x build/<K>.xclbin -i build/<K>.bin -k MLIR_AIE \
  -M <M> -K <K> -N <N> -w 128 -y 64 -z 128 --warmup 1 --iters 2
```

**Result -- `M=4096 K=4096 N=2048`, k4096 xclbin:**

| probe | result |
|---|---|
| stock (all ones) | PASS |
| identity (one nonzero per row, total nz = 2048) | **PASS** |
| ramp (C = j+1 through full K) | PASS |
| tile-a / tile-b | PASS |
| pseudo-random A and B | PASS |

**Result -- the real FFN shapes, each on the xclbin compiled for its own K:**

| shape | xclbin | identity | ramp | tile-a/b | pseudo-random |
|---|---|---|---|---|---|
| 2048 x 5120 x 17408 (gate/up) | gateupfix | PASS | PASS | PASS | FAIL 1000/1000 |
| 2048 x 17408 x 5120 (down) | downfix | PASS | PASS | PASS | FAIL 1000/1000 |

So the A shuffle, the B shuffle and the C un-shuffle are all correct at real geometry. The
only remaining failure is the pseudo-random pattern, and that is **not** a GEMM defect.

## The residual failure is the bfp16 input encoder, proved two ways

Two independent tests isolate it.

**(1) Exactly representable random data.** `ATB_QRAND` uses the *same* mod-29 / mod-31
pseudo-random patterns, divided by 32 and 64 instead of multiplied by 0.05 and 0.03. Both
are dyadic rationals with few enough significant bits to be exact in bfp16, so the device's
inputs equal the host's bit-for-bit. Same structure, same layout, same K:

```
2048 x 5120 x 17408, ATB_QRAND=1 -> PASS, 32.46 TFLOPS (11246 us)
```

**(2) A quantization-aware reference.** `ATB_QUANTREF` replaces the reference's A and B with
the host's *own* `floatToBfp16` -> `bfp16ebs8ToFloat` round trip, regrouped into the
8-element shared-exponent blocks the device actually sees (for A the shuffled and row-major
block memberships coincide; for B they do not, so B is regrouped by `(n-column, k-block)`):

```
2048 x 5120 x 17408, ATB_A_REAL/B_REAL + ATB_QUANTREF -> PASS
2048 x 5120 x 17408, ATB_A_REAL/B_REAL (unquantized control) -> FAIL 1000/1000, max rel 170%
```

The device reproduces the reference *exactly* once the reference is computed from the data
the device was actually given. The GEMM, the accumulation and the C path are correct.

**What the encoder does.** `floatToBfp16` in `block_datatypes/helper.h` keeps `mbits = 7`
magnitude bits per element with one shared exponent per 8 elements, and rounds by
**arithmetic right shift** (line 103), i.e. truncation toward -inf, not round-to-nearest.
Measured on the pseudo-random pattern:

| operand | mean raw | mean quantised | bias | rms err |
|---|---|---|---|---|
| A | 4.8e-09 | -0.00323 | **-0.00323** | 0.0039 |
| B | -0.4091 | -0.41319 | **-0.00404** | 0.0056 |

Both biases are negative, as truncation-toward--inf implies.

**Why it looks catastrophic on this pattern and not on real weights.** The `A_REAL`/`B_REAL`
pattern is periodic in k with periods 29 and 31, so over the 899-element common period the
products sum to exactly zero and the true C is only the residual partial sum: sd **0.88**,
against an individual term scale of ~6.4. The observed device values sit around 6-9. The
apparent "+6.82 constant offset" is therefore not a DC error -- the encoder's measured biases
predict a DC of only `K * 0.00323 * 0.00404 = 0.067`. It is the near-perfect cancellation
being destroyed by a ~0.4% input perturbation, which moves C by an amount comparable to the
*full uncancelled* sum. A badly conditioned probe, not a broken GEMM. The same encoder on
real weights adds 0.31% RMS against the paper's own 3% quantiser (see the quality gate above).

## One further defect in the vendored test

`layout_transpose_L1_1x2_8x8block` documents its input as row-major with `rows = K`,
`cols = N`, but the test's B fill comment says "host B is column-major N x K" and it passes
`b_col_maj = true` to `verify_stochastic`, which then reads `B[k + col*K]`. All three cannot
hold at once. We fill B row-major K x N and pass `b_col_maj = false`, the only self-consistent
combination; it is what passes on random data with the correct geometry. As with the tile
shape, the all-ones pattern is invariant to this too, so the shipped test never exercised it.

## What this does and does not establish

Established: the ATB config3 GEMM computes correctly at our real FFN geometry -- layout,
accumulation and writeback all verified against a CPU reference on data that is not
permutation-invariant, at 32.4-32.9 TFLOPS. The earlier "correctness unverified" caveat is
retracted; the repack is not blocked on the kernel.

Not established, and the falsifier for the above: every probe so far is synthetic. The
strongest test is the one the repack needs anyway -- feed real IQ3_XXS `blk.0.ffn_gate.weight`
and `blk.0.ffn_down.weight` through the same host path and compare against a float reference.
That would exercise the encoder on the actual weight distribution rather than a constructed
one, and it is the next step regardless of this result.

## Real IQ3_XXS FFN weights through the NPU path

The falsifier above is now satisfied, and it moved the quality number.

**Setup.** `blk.0.ffn_gate.weight` and `blk.0.ffn_down.weight` were pulled from the shipped
shard (type 18, IQ3_XXS, 34.1 MB of raw blocks each) and dequantised with the production
decoder (`tools/qwen27b/atb_real_weights.cpp`, linked against `ggml_dequant.cpp`), then
transposed into the ATB B layout (row-major K x N). A is a fixed-seed standard normal
(M x K); the bfp16 error is a per-element relative effect, so the output error is insensitive
to A's scale. Both operands are loaded into the vendored host through new `ATB_AFILE` /
`ATB_BFILE` hooks and run at the real geometry on the xclbin compiled for each K.

**The kernel is exact on real weights.** Against a reference built from the host's own
`floatToBfp16` round trip, both shapes **PASS**, and both also pass against the float
reference:

| shape | quantisation-aware reference | float reference |
|---|---|---|
| 2048 x 5120 x 17408 | PASS | PASS |
| 2048 x 17408 x 5120 | PASS | PASS |

The float-reference PASS is not evidence of accuracy -- it passes because `abs_tol = 0.5` is
loose against an output whose RMS is about 0.73. The number below, not the pass/fail, is the
one that matters.

**What bfp16 costs, measured as relative error on the GEMM output** (2000 sampled cells,
RMS of the difference over RMS of the reference):

| shape | weights only | activations only | both |
|---|---|---|---|
| 2048 x 5120 x 17408 (gate/up) | 1.23% | 1.27% | **1.88%** |
| 2048 x 17408 x 5120 (down) | 1.28% | 1.33% | **2.26%** |

The two contributions are close to independent: `sqrt(1.23^2 + 1.27^2) = 1.77%` against the
measured 1.88%.

**This corrects the earlier bfp16 quality gate.** That gate reported 0.31% RMS for 8-bit
magnitudes and concluded the conversion was "numerically close to free". It modelled
**round-to-nearest**, but `floatToBfp16` in `block_datatypes/helper.h` keeps `mbits = 7` and
rounds by **arithmetic right shift** (line 103) -- truncation toward -inf. The chain is:

| model | weight-side RMS |
|---|---|
| 8-bit, round-to-nearest (what the gate measured) | 0.31% |
| 7-bit, round-to-nearest | 0.62% |
| 7-bit, truncation (what the host actually emits) | 1.24% |

and the measured weight-side output error at the gate/up shape is **1.23%**. The factor of 4
is entirely accounted for: the gate tested a different encoder than the one in use.

**Consequence.** The NPU path is not free numerically. It adds about 2% RMS to the FFN
outputs, split evenly between the weights and the activations. Whether that is acceptable is
an end-to-end question (perplexity with and without the conversion), not something to assume
from a rounding model. The throughput is unaffected -- 32.4-32.9 TFLOPS at 1.3% iGPU cost -- but
the prefill projection is **1.42x, not 1.68x** (see the correction in the concurrency section),
and it is a projection rather than a measurement.

**A free 1.6x is available, and now measured.** The 7-bit/truncation combination is our
choice, not the hardware's: the stored value is just an int8 magnitude times a shared
exponent, so the host may round to nearest when it encodes. Made switchable with one branch
(`BFP16_ROUND_NEAREST` in `helper.h`, adding half an ulp before the shift) and re-measured on
the same operands:

| shape | contribution | truncate (shipped) | round-to-nearest |
|---|---|---|---|
| 2048 x 5120 x 17408 | weights | 1.23% | 0.77% |
| | activations | 1.27% | 0.77% |
| | **both** | 1.88% | **1.07%** |
| 2048 x 17408 x 5120 | weights | 1.28% | 0.79% |
| | activations | 1.33% | 0.80% |
| | **both** | 2.26% | **1.12%** |

The device stays exact under the improved encoding -- both shapes PASS against the
quantisation-aware reference with round-to-nearest on -- and the all-ones stock pattern still
passes, because the change is purely a host-side choice of which int8 to store.

The gain is 1.6x rather than the 2x the uniform-error model predicts, because the truncation
error is not independent of the per-element shared-exponent shift. It costs nothing in
throughput or memory, so **round-to-nearest should be the default in the repack**; the ~2x
figure that would have followed from the earlier 8-bit gate is not reachable.

**Real activations change nothing, and that is the useful result.** `GUFO_DUMP_ACTIVATION`
writes the post-FFN-norm FFN input (fp16, batch x hidden) from a real prefill, so A can be a
real activation rather than a synthetic draw. The hypothesis was that the true FFN input has
outliers a per-8 shared exponent handles worse than a Gaussian. It does not: the dumped
activation has max|a| = 50.3 against the Gaussian's 5.5, but its per-8 `amax/rms` is 1.89
(Gaussian 1.84; p90 2.33 vs 2.22). The outliers are rare and do not co-occur in blocks, so the
shared-exponent cost is set by each block's own dynamic range, not by the tail.

| A operand | encoding | weights only | activations only | both |
|---|---|---|---|---|
| standard normal | truncate | 1.23% | 1.27% | 1.88% |
| real FFN input (layer 0, 2048 x 5120) | truncate | 1.16% | 1.19% | **1.81%** |
| real FFN input | round-to-nearest | 0.71% | 0.76% | **1.04%** |

Real weights throughout (`blk.0.ffn_gate.weight`, IQ3_XXS, via
`tools/qwen27b/atb_real_weights.cpp`), 2000 sampled cells, on the device. The device stays
exact on the real activation with round-to-nearest on (PASS against the quantisation-aware
reference).

So the headline is **1.8% RMS with the shipped encoder and 1.0% with round-to-nearest**, on
real weights and real activations. The synthetic A was not hiding anything, which retroactively
justifies the earlier synthetic probes as well.

## The repack is validated -- and it, not the GEMM, is the expensive part

**The offline packer.** `tools/qwen27b/atb_pack.cpp` turns a raw IQ3_XXS tensor into the
bfp16 B operand in the ATB layout in one pass: dequantise -> transpose to row-major [K, N] ->
(k_tile, n_tile) L1 tiles emitted column-major -> 1x2 super-blocks of 8x8 column-major
sub-blocks -> bfp16, shared exponent per 8, round-to-nearest. It is a standalone port of
`layout_transpose_L1_1x2_8x8block` and `floatToBfp16`, so the engine needs nothing from the
mlir-aie tree at runtime.

Validated two independent ways:

1. **Byte identity.** The vendored host dumps the exact buffer it hands the device; the packer
   reproduces it byte-for-byte for both FFN tensors (100,270,080 bytes each, `cmp` clean).
2. **Device consumption.** The host can load a pre-packed B and skip its own shuffle and encode
   (`ATB_BPACKED`); at both real shapes it then PASSes against the quantisation-aware
   reference.

One trap worth recording, because it is easy to miss: `floatToBfp16` advances its write cursor
**twice** per block -- once per mantissa and once more after writing the exponent -- and that
second increment is what makes the stride 9 bytes instead of 8. Omitting it leaves the stream
one byte short per block, and the *first* block still matches, so the corruption is invisible
until a multi-block comparison. The port reproduced the helper only after that was found.

**What it costs in memory.** bfp16 is 9 bits per weight, against 3.484 bpw for the FFN as
shipped (see the type census above), so a packed FFN is 2.58x larger:

| | weights | size |
|---|---|---|
| FFN gate+up+down, 65 blocks x 3 | 17.380 G | 7.05 GiB |
| the same as bfp16 (9 bits) | | **18.21 GiB** |
| model plus fully packed FFN | | **23.34 GiB** |

Packing *all* FFN weights therefore needs 18.21 GiB and, alongside the 12.18 GiB model, does
not fit in the ~18 GiB available. But the operating point is not "all": at the measured split
the NPU takes 44% of the FFN, so the packed set is ~8.0 GiB gross, and because a channel is
served by either the NPU or the GPU the 3.484-bpw bytes it replaces are freed -- about 3.1 GiB
-- for a net of roughly **+4.9 GiB**, giving ~17.1 GiB total. That fits, with about a gigabyte
of headroom. The ceiling is real and the plan is under it, but the headroom is thin enough that
the split fraction is now a memory decision as well as a throughput one.

**What it costs in time -- and why the first answer was wrong.** Timed on the real gate/up shape
at a 2048-token batch, host-side, single-threaded:

| operand | shuffle | encode | total | vs GEMM |
|---|---|---|---|---|
| A (activation, 2048 x 5120) | 9.3 ms | 42.3 ms | 51.6 ms | 4.5x |
| B (weights, 5120 x 17408) | 77.0 ms | 326.1 ms | 403.1 ms | 35x |
| NPU GEMM | | | 11.45 ms | |

That reads as a dead end for feeding the NPU from the host, and it was written up that way for
one round. It is wrong. The packing is embarrassingly parallel -- every 8-element exponent block
and every L1 tile is independent of every other -- and the measurement was single-threaded on a
32-thread part. `tools/qwen27b/atb_act_bench.cpp` measures the same work at 1 to 32 threads:

| threads | shuffle | encode | total |
|---|---|---|---|
| 1 | 6.38 ms | 56.74 ms | 63.12 ms |
| 4 | 1.88 ms | 14.42 ms | 16.30 ms |
| 8 | 1.41 ms | 7.32 ms | **8.73 ms** |
| 16 | 1.05 ms | 4.49 ms | **5.53 ms** |
| 32 | 0.90 ms | 3.57 ms | 4.46 ms |

At 8 threads the activation pack is 8.73 ms against 11.45 ms of GEMM; at 16 threads it is
5.53 ms, **below the compute it feeds**. The host can prepare A, so fusing the encoder into the
GPU norm and SwiGLU kernels is an optimisation rather than a prerequisite. During GPU-bound
prefill the CPU is otherwise idle, so those threads are close to free.

B stays a one-off: 403 ms single-threaded per tensor, well under a second with threads, paid
once at load.

**The lesson is the same one this document keeps relearning.** The infeasibility came from not
using the other 31 cores, not from the design; the correction is one measurement. The operand
preparation is affordable on both axes, and the NPU path's cost remains the 32 TFLOPS GEMM.

## The repack covers every FFN type, and both operands are exact ports

The packer above was written for IQ3_XXS. The shipped FFN is not IQ3_XXS -- it is seven types
(see the census earlier) -- so the packer was extended to all of them and the block geometry was
checked against the shard before being trusted. For each type, the GGUF header's own span
between consecutive tensors must equal `elements / block * block_bytes`:

| type | block | block bytes | header span matches |
|---|---|---|---|
| Q3_K | 256 | 110 | yes (`blk.0.attn_gate.weight`) |
| Q4_K | 256 | 144 | yes |
| IQ2_XXS | 256 | 66 | yes (`blk.3.ffn_gate.weight`) |
| IQ2_XS | 256 | 74 | yes (`blk.14.ffn_gate.weight`) |
| IQ3_XXS | 256 | 98 | yes (`blk.0.ffn_down.weight`) |
| IQ3_S | 256 | 110 | yes |
| IQ4_XS | 256 | 136 | yes |

Every dequantiser needed was already in `ggml_dequant.hpp`; the work was the dispatch table
(`tools/qwen27b/atb_quant_types.hpp`) and the validation.

**Byte identity, one real tensor per type.** Packing each and comparing with the bytes the
vendored host hands the device:

| type | tensor | K | N | result |
|---|---|---|---|---|
| IQ3_XXS | blk.0.ffn_gate.weight | 5120 | 17408 | identical |
| IQ3_S | blk.20.ffn_gate.weight | 5120 | 17408 | identical |
| IQ4_XS | blk.4.ffn_gate.weight | 5120 | 17408 | identical |
| Q3_K | blk.54.ffn_gate.weight | 5120 | 17408 | identical |
| Q4_K | blk.51.ffn_down.weight | 17408 | 5120 | identical |
| IQ2_XXS | blk.3.ffn_gate.weight | 5120 | 17408 | identical |
| IQ2_XS | blk.14.ffn_gate.weight | 5120 | 17408 | identical |

100270080 bytes each, `cmp` clean. The Q4_K case is an `ffn_down`, so the down shape is covered
as well as gate/up.

**The A operand too.** B is repacked offline, but A is produced per prefill call, so the same
proof is needed for the runtime path. `atb_act_bench.cpp` now also writes its packed output and
the host dumps `AVecBfpShuffled` (`ATB_DUMPA`); on the real FFN activation at the gate/up shape
the two are **byte-identical** (11,796,480 bytes).

So both operand producers are exact ports of the vendored layout and encoder rather than
approximations: A is a port of `layout_A_L1_2x1_8x8block` plus `floatToBfp16`, B a port of
`layout_transpose_L1_1x2_8x8block` plus the same encoder, and equality has been demonstrated
byte-for-byte on real data for every type and both operand roles. What is *not* yet shown is any
of this running inside the engine -- the packer and the A path are tools, and the wiring is the
remaining engineering.

## The one-off repack is a minute, and the per-call packing fits in the GPU's shadow

The batch driver (`tools/qwen27b/atb_pack_gguf.py`) repacks every FFN tensor in the shard,
reading the raw blocks straight out of the memory map and driving `atb_pack` in parallel:

```
packing 195 FFN tensors with 6 jobs (out=discard)
packed 17.380 G weights -> 18.21 GiB at 9 bits/weight in 57.6 s wall (0.30 G weights/s)
```

Under a minute at load, and each written tensor is exactly 100,270,080 bytes -- the same size the
byte-identity checks produced. **Memory, not time, is the binding constraint on the repack.**

The per-call A packing has two shapes, because gate and up share one activation while down
consumes the SwiGLU output:

| A operand | 8 threads | 16 threads | 32 threads |
|---|---|---|---|
| gate/up (2048 x 5120) | 8.73 ms | 5.53 ms | 4.46 ms |
| down (2048 x 17408) | 24.12 ms | 15.75 ms | 12.26 ms |

The down operand here is the real FFN activation tiled along K to 17408 elements; the packing
cost is per-element and tiling preserves the distribution. Per layer that is about **21.3 ms of
CPU at 16 threads**. The GPU's own prefill at the measured 543.93 tok/s does a 2048-token prompt
in 3.77 s, or **59 ms per layer**, and during GPU-bound prefill the CPU is otherwise idle. Since
21.3 < 59, the activation packing can be hidden under the GPU's work instead of serialised
against the NPU's. That is the difference between a ~59% overhead on the NPU's 1.02 s of FFN
compute and none.

So the operand side of the NPU path is settled: exact, a minute to build once, a few GiB net, and
fast enough to overlap. The open items are the two that cannot be settled with tools -- the
engine wiring, and an end-to-end quality measurement of the 1.0% RMS conversion.

## The quality gate, run: the bfp16 operand encoding is an order of magnitude below noise

The gate is whether the bfp16 encoding degrades the model. The per-GEMM number (1.04% RMS)
cannot answer that: the FFN output feeds SwiGLU and the residual stream and repeats 65 times,
so the question is whether 1.04% stays 1.04% or accumulates.

**Instrument.** `qwen27b_target_test`, which already grows its context with a corpus fixture so
the prompt crosses the fp16/int8 prefill switch and can `--dump` / `--compare` two runs of the
same model under different environment. 1024-token prompt, 512 teacher-forced rows = **513 logit
rows over the full 248,320-token vocabulary**, exactly deterministic (two runs give KL 0). Corpus
is `README.md + QUALITY.md + TESTING.md`, 29,405 bytes.

**Perturbation.** A new env-gated diagnostic, `GUFO_BFP16_FFN_BITS=N`, rounds the FFN input
activation in place onto a shared-exponent grid with N magnitude bits -- the same 8-element
grouping (consecutive k within one row) and the same grid the NPU operand encoder uses. It is
applied at the FFN norm output **inside the per-layer loop**, so every layer's FFN sees it and the
effect accumulates the way it would in production. Unset or N outside [1,9] is a no-op.

| configuration | top-1 | mean KL | mean TV | NLL delta |
| --- | ---: | ---: | ---: | ---: |
| no-op (`GUFO_BFP16_FFN_BITS=10`) vs baseline | **513/513** | **0** | 0 | 0 |
| **bfp16 A operand (7 bits)** | 512/513 | **1.91e-05** | 0.000867 | +9.8e-05 |
| 6 bits | 513/513 | 5.91e-05 | 0.001187 | -0.000247 |
| 5 bits | 510/513 | 9.38e-05 | 0.001983 | +0.000732 |

The no-op row is the safety check: the diagnostic is bit-inert when the knob is not on, so the
default path is provably unchanged rather than assumed to be.

**Anchors, measured on the same corpus in the same session** (the doc's published int8/int4
figures use a different corpus, so absolute values are not comparable across them):

| configuration | top-1 | mean KL | mean TV | NLL delta |
| --- | ---: | ---: | ---: | ---: |
| baseline vs int8 activations (forced q8) | 510/513 | 2.007e-04 | 0.002197 | +0.000543 |
| int8 vs **int4** activations | 496/513 | **1.017e-03** | 0.006680 | -0.000900 |
| baseline vs int4 activations | 495/513 | 1.144e-03 | 0.007145 | -0.000356 |

**Reading.** The bfp16 activation encoding costs **1.91e-05 mean KL: 10.5x below the int8
activation step and 53x below the int4 grid**, on a corpus where the int4 grid itself is already
the smaller of the two numbers in the published table. On the likelihood metric the deltas are
+/-1e-4 nats with mixed sign, i.e. nothing. The KL grows sublinearly with the grid: the step
doubles per bit and KL rises 3.1x and then 1.6x, so the response is not near a cliff at 7 bits.

**The one inference, stated as one.** This measures the **activation** half -- 0.76% of the
combined 1.04% per-GEMM error. The weight half (0.71%) is not directly measured; it is a slightly
*smaller* and independent perturbation, and taking the sweep's own slope, a second perturbation
of that size roughly adds, giving a combined estimate of ~4e-05 -- still 5x below the int8 anchor
and ~25x below the int4 grid. Measuring the weight half directly means the same round trip at the
five weight-decode sites in the fp16 prefill kernel instead of the one activation site; on this
evidence that is not worth the hot-path risk, which is why it is an inference and not a
measurement.

**Gate: passed, with margin.** The bfp16 FFN path is below the threshold this project has already
accepted elsewhere, so the engine wiring is not gated on quality. The diagnostic is left in place
behind its env var as the instrument for any future datatype change (an int8 path would want the
same measurement, and on the int8 anchor it would land 10x higher).

## The C handoff is not a risk (item 1 of the split)

The split projection assumes the NPU's FFN output reaches the GPU without a meaningful cost.
Checked rather than assumed:

* **NPU side exports dma-buf.** `amdxdna` carries `amdxdna_dmabuf_ops`,
  `amdxdna_gem_dmabuf_mmap`, `amdxdna_cbuf_dmabuf_ops` and `amdxdna_ubuf_dmabuf_ops`, and
  XRT exposes `xrt::bo::export_buffer()` (`/usr/include/xrt/xrt_bo.h:545`).
* **GPU side imports it.** `hipExternalMemoryHandleTypeOpaqueFd = 1` and
  `hipImportExternalMemory` are present in the ROCm headers we build against.
* **So the intended path is a dma-buf import, not a copy.** The NPU's output buffer is ordinary
  system RAM on this APU, so once the mapping exists a GPU kernel reads it in place.

And the pessimistic case is bounded anyway. The NPU's share of the FFN output is
`0.438 x 81.8 M` elements per layer = 35.8 M, or **40 MB** at bfp16, against the ~19.5 ms of NPU
compute that produces it -- a required 2.05 GB/s. This box does roughly 100-250 GB/s to DRAM
(the document's own figures: 43 GB/s described as 17% of peak, 108 GB/s as half), so even a full
system-RAM **copy** is ~0.4 ms, about **2% of the NPU's per-layer time**, and zero if the import
works. Either way it does not move the 1.42x.

What is *not* done: the two-engine functional test (NPU writes, GPU imports the fd and verifies
the bytes). That is the confirming measurement, and it belongs with the split wiring rather than
before it -- the cost bound above is what the projection needs, and it holds in the worst case.

## Runtime repacking: the packed model never has to exist

The 4.9 GiB a packed NPU share costs is avoidable. `tools/qwen27b/atb_repack.hip` produces the
bfp16 ATB B operand **on the GPU, per layer**, straight from the resident quantised tensor, into a
staging buffer that is reused -- so the shipped 3.484 bpw weights stay the only copy. A
double-buffered staging area is `2 x 43.8% x 3 x 100 MB` = **262 MB**, against 4.9 GiB.

One thread per bfp16 group. A group is 8 consecutive reduction rows at one output column, which
is exactly what the packer's shuffle makes contiguous, so each thread writes 9 contiguous bytes
from 8 weights of one row -- and it gets the dequantisation from the engine's own
`DecodeQuantSub16`, so all seven FFN types are covered without a per-type branch.

**Layout is exact.** The exponent byte of each group encodes which 8 weights share it, so it is
the layout-sensitive byte and only weakly value-sensitive. Across all seven tensors:

| check | result |
|---|---|
| exponent bytes differing | **0 of 77,987,840** |
| mantissa bytes differing | 13.08%, all by exactly one grid step |

A layout error would move whole groups, so zero exponent mismatches is the layout proof.

**The 13% mantissa disagreement is the kernel being more accurate, not less.** The host helper
truncates the mantissa to 7 bits *before* it rounds (its `mantissa >> 17`, then add-half-then-
shift), so its rounding never sees the bits below bit 17. The kernel rounds the exact FP32
`scale*q - offset` onto the same grid. Same grid, so the accuracy is identical; the kernel's
choice is the better one. The one alarming number in the first pass -- Q3_K with a max byte
delta of 255 -- was +127 versus -128, adjacent points in two's complement.

**Cost, after removing a 16x redundancy.** The first version called
`QuantBlockElement` once per weight; that decodes a whole 16-element sub-block each time, so 8
calls decoded 128 elements for the 8 that were wanted. Decoding each sub-block once:

| | per tensor (89.1 M weights) |
|---|---|
| first cut | 2.31 - 3.53 ms |
| sub-block decoded once | **0.91 - 1.20 ms** |
| CPU packer, for scale | 388 ms |

So the repack is ~350x the CPU path, and it is compute-bound on the dequant, not bandwidth-bound
(100 MB written in 1.1 ms is 91 GB/s).

**What it costs the projection.** Per layer the NPU's share needs `0.438 x 3.3 ms` = **1.45 ms**
of repack on the GPU. The GPU's per-layer work becomes 19.5 + 1.45 = 21.0 ms against the NPU's
19.5 ms, so the FFN speedup goes 1.78x -> **1.66x**, and the prefill projection
`1/(0.33 + 0.67/1.66)` = **1.36x** rather than 1.42x.

That is the trade the instruction buys: **4.6 GiB not held, for about 4% of the projected
speedup.** Worth taking, and now a measured number rather than a preference.

Not yet done: the engine wiring. `atb_repack` is a prototype that reads the tensor from a file
and writes to stdout; the engine version reads the resident `layer.ffn_*.data` against its
type and writes into the double buffer. That is also where the split lands, since both need the
same per-layer hook.

## Regression hunt: every busy CPU core costs 2-3% of prefill

The standing pp2048 reference is 543.93, and the box now reads 456-498. Two hypotheses were
tested and one of them holds.

**Not thermal.** A full cooldown -- eight minutes idle, then the same benchmark -- gave 490.84
before and 492.94 after, no recovery. Temperatures during runs sit at 90-92 C edge, and the
platform profile is `balanced`, but neither is a step change.

**Not code.** This session's engine changes are additive and environment-gated: `+119` lines,
zero deletions, and every one of them behind a `getenv`. The hot kernel
(`HalfPrefillGemmKernel`) is untouched; the additions are a separate diagnostic kernel, a
separate repack kernel, and two guarded call sites.

**It is CPU load, and it is measurable.** Adding known busy cores and re-running the identical
benchmark:

| busy cores added | pp2048 | cost |
| ---: | ---: | ---: |
| 0 | 498.72 +/- 8.39 | -- |
| 1 | 486.79 +/- 4.99 | -2.4% |
| 2 | 476.96 +/- 4.33 | -4.4% |
| 4 | 466.90 +/- 1.60 | -6.4% |
| 8 | 375.93 +/- 6.50 | **-24.6%** |

**About 2-3% of prefill per busy core**, which is the signature of a power-capped SoC rather
than a CPU-starved host: this box caps at ~95 W (95.9 W observed at the peak) and CPU watts come
out of the same budget as the iGPU's. It also explains the earlier "cross-session drift" of 8.5%
-- it was never drift, it was whatever else the machine was doing.

**The engine is one of those busy cores.** The host thread spins in `libhsa-runtime64` for the
entire prefill (see the previous section), which is one busy core and so **~2.4% of pp2048** --
self-inflicted, in our control, and recoverable by replacing the spinning wait at the chunk
boundary with a blocking one. The desktop accounts for roughly another one to two cores
(measured concurrently: this GUI's renderer 63-100%, waterfox 24-38%, btop 10%, niri 5-14%),
which is a further 3-6%.

**This also settles the activation-packing plan.** Two sections ago I budgeted the A encode at
5.53 ms per layer on 16 CPU threads and called it hidden under the GPU's 59 ms. The cost model
was wrong: 8 busy cores already cost 24.6%, so 16 threads of encoding would cost far more than
the ~10% the split is playing for. **The A operand has to be encoded on the GPU**, next to the B
repack that already runs there for 1.45 ms/layer. The measurement does not just qualify that
correction, it makes the CPU path untenable.

**And a measurement-methodology note that applies to this whole document**: every pp2048 in it
was taken with a browser, a terminal and a compositor running, and this session's numbers were
additionally taken *while streaming results to a GUI on the same box*. At 2-3% per busy core
that is several percent of uncertainty on every figure. The 543.93 reference is not
reproducible on a loaded desktop, and the 8-core row shows the sensitivity is not linear at the
top end either.

## The repack hook, measured in situ

`GUFO_ATB_PACK=1` now repacks the layer's gate, up and down tensors into a 200 MB reusable
staging buffer inside `prefill_chunk.cpp`, at the FFN site, using the same kernel the tool
validated. Nothing consumes the output yet -- the split is the consumer -- so what it measures is
the cost the split will pay.

Kernel trace of a full prefill (rocprofv3 `--kernel-trace`):

| kernel | calls | total | per call |
| --- | ---: | ---: | ---: |
| `AtbRepackKernel` | 384 | 587.85 ms | **1.53 ms** |

384 calls is 65 layers x 3 tensors x 2 forwards (warmup + timed). So the in-situ cost is
**1.53 ms per tensor** against the standalone tool's 0.91-1.20 ms -- the gap is launch overhead
and the real geometry.

Per layer that is 3 x 1.53 = **4.59 ms** for the whole FFN; the NPU's 43.8% share is
**2.01 ms**. The GPU's per-layer work becomes 19.5 + 2.01 = 21.5 ms against the NPU's 19.5 ms, so
the FFN speedup goes 1.78x -> **1.62x** and the prefill projection
`1/(0.33 + 0.67/1.62)` = **1.34x** -- against 1.36x from the tool-level estimate.

**Caveat that applies to every number taken in this session**: the box runs with `iommu=pt`,
which was enabled for the NPU and costs about 10% of prefill (543.93 -> ~492 tok/s). The split's
relative costs -- the repack, the A encode, the balance point -- should carry over, but these
absolute pp2048 values are not comparable with the 543.93 reference, and a kernel fix for the
regression is being handled separately.

**Still to come**: the wiring. The repack writes bytes into a buffer nobody reads; the split
needs XRT inside the engine, the NPU launch per layer, the dma-buf import of its output, and the
row partition. That is the piece that would turn 1.34x into a measurement.

## The dma-buf handoff does not work -- and an unchecked import hangs the GPU

I built the handoff probe and it took the machine down. Recording both halves.

**The crash.** `tools/qwen27b/atb_handoff.hip` runs the ATB kernel on the NPU, exports its
output BO with `xrt::bo::export_buffer()` and imports it into HIP. The first version did not
check a single return code, so when the import failed it launched a GPU kernel on a null mapped
pointer. `dmesg`:

```
amdxdna_iommu_init: Enabled force_iova mode.
amd_iommu_report_page_fault: 1579 callbacks suppressed
amdgpu: ring gfx_0.0.0 timeout, signaled seq=6924296, emitted seq=6924299
amdgpu: GPU reset begin! ... GPU reset(1) succeeded!
```

IOMMU page faults, then a ring timeout, then a GPU reset. The lesson is not subtle and it is now
enforced in the code: **every call is checked before anything is launched**, and
`ATB_HANDOFF_SKIP_GPU=1` stops after the import and map so the API path can be probed without
running any kernel at all.

**The finding.** With the checks in place, the same run in probe mode:

```
NPU run complete
export -> fd 4
import -> out of memory (2)
HANDOFF ABORT at import
ring timeouts after: 0
```

`hipImportExternalMemory` on an `amdxdna`-exported dma-buf returns **`hipErrorOutOfMemory`**.
So the zero-copy handoff does not work as assumed earlier; the XRT BO is not importable into the
GPU's address space on this stack.

**This does not block the split.** The bound computed earlier still stands: the NPU's share of the
FFN output is 40 MB per layer against ~19.5 ms of NPU compute, so it needs 2.05 GB/s, and a
system-RAM copy at ~100 GB/s costs about **0.4 ms, 2%**. The handoff is a copy rather than a
mapping, and the projection absorbs it.

Still untested, and not on the critical path: whether a `/dev/dma_heap/system` dma-buf imported
by *both* drivers would give zero-copy, or whether HIP can export its own allocation for XRT to
import in the other direction. Either would recover the 2%; neither is needed to proceed.

## Both operands are now producible on the GPU, in the ATB layouts

`tools/qwen27b/atb_encode_a.hip` is the A side: an FP16 activation `[rows, K]` into the bfp16 A
L1 layout, same 2x1/8x8 shuffle and same shared-exponent encoder as the B pack, one thread per
8-element group. Validated against the CPU oracle in `atb_act_bench.cpp` on a 2048 x 5120
activation with injected outliers:

| check | result |
| --- | ---: |
| bytes | 11,796,480 |
| **exponent mismatches** | **0** |
| mantissa mismatches | 11.7%, one grid step |

Zero exponent mismatches is the layout proof, same as for B: the exponent byte encodes which 8
values share it, so a wrong grouping moves whole groups. The mantissa difference is the same
rounding-mode difference as B -- the CPU oracle truncates the mantissa to 7 bits before rounding,
the kernel rounds the exact value -- and all differences are adjacent grid points, with the GPU
the more accurate of the two.

So the operand side of the split is complete and verified:

| operand | producer | status |
| --- | --- | --- |
| B, weights | `atb_repack` (also in the engine, 1.53 ms/tensor) | layout-exact, all 7 FFN types |
| A, activations | `atb_encode_a` | layout-exact |

What remains for the split is three things and no unknowns: the **C decoder** (the ATB C layout
back into the engine's FP32 FFN rows, the inverse of the 2x2/8x8 shuffle plus a bfp16 decode),
**XRT inside the engine** (a skeleton now exists in `atb_handoff.hip`), and the **row
partition**. The handoff between them is a copy, bounded at 2%.

## The C decoder is bit-exact, so all three layout transforms are done

`tools/qwen27b/atb_decode_c.hip` is the last of them: the device's C, bfp16ebs8 in L1 tiles of
(512, 128) with 2x2 super-blocks of 8x8 row-major sub-blocks, back into the engine's FP32 FFN
rows. One thread per 8-element group; consecutive packed elements are 8 consecutive output columns
of one row, so a group writes 8 contiguous floats.

Its oracle is the vendored host's own verification path -- `bfp16ebs8ToFloat` followed by
`layout_inverse_C_L1_2x2_8x8block`, in `tools/qwen27b/atb_c_oracle.cpp`. On a 4096 x 2048
buffer with randomised mantissas:

```
C decode 4096x2048: elements=8388608 exact_mismatches=0 max_abs=0
```

**Zero on 8.4 M elements.** That is a stronger result than the encoders got, and for a good
reason: the decoder is pure arithmetic with no rounding choice, so there is nothing to disagree
about once the layout and the two's-complement handling are right. The encoders differ from their
oracles only in which grid point they round to.

So the layout layer is finished:

| transform | validation |
| --- | --- |
| B encode (weights) | 0 exponent mismatches, all 7 FFN types |
| A encode (activations) | 0 exponent mismatches |
| **C decode** | **bit-exact, 0 of 8,388,608** |

What is left for the split is now only the two things that cannot be validated in isolation:
**XRT inside the engine** and the **row partition**.


## The handoff is a map, not a copy: the NPU reads the iGPU's own memory

Three separate mistakes made the previous section conclude wrong. All three are cheap to state
because each was caught by a direct measurement.

**The launch was wrong.** The ERT start-CU opcode for these xclbins is the literal `3`, not
`kernel.group_id(0)`. `atb_handoff.hip` passed the group id, the kernel never ran, C stayed at
its zeroed contents, and the probe then compared the host's view of that buffer against the GPU's
view of the same all-zero buffer and reported `differing=0`. The "handoff must be a copy
(~2%)" conclusion was drawn from a test that could not have failed. With opcode 3 the same xclbin
produces C with every byte non-zero and `run state=4`.

**The dma-buf direction that fails is not the one the split needs.** `hipImportExternalMemory` on an
`amdxdna`-exported dma-buf does fail. The split needs the opposite: the iGPU allocates, the NPU
consumes. That direction works, and it needs no XRT feature at all --
`hsa_amd_portable_export_dmabuf` on a `hipMalloc` pointer yields an fd that `xrt::bo(device, fd)`
imports as a BO the ATB kernel accepts.

**Every alternative is worse or closed.** Measured on a 64 MB buffer:

| path | result |
| --- | --- |
| userptr BO over `hipHostMalloc` | rejected: "User pointer BO must be AMDXDNA_BO_SHARE type" |
| `hipHostRegister` on an XRT BO's mapping | invalid argument, all four flag sets |
| HIP -> XRT host-only BO | 22.8 GB/s in, 11.3 GB/s out |
| HIP <-> `hipHostMalloc` | 86 GB/s both ways |
| CPU memcpy pageable -> XRT BO | 27 GB/s |

An XRT host-only BO is slow memory with the read side at 11 GB/s: a copy handoff over gate, up and
down would have cost more per layer than the split saves. `hipHostMalloc` cannot be exported as a
dma-buf, so the memory has to be the iGPU's.

`tools/qwen27b/atb_npu_run.hip` is the resulting harness: `hipMalloc` A, B and C, export each,
import each, fill A and B with a GPU kernel and never the host, then check C twice -- through a
second GPU kernel, and by `mmap` of the dma-buf, which is a plain CPU load that sees exactly what
the NPU sees.

    M=2048 K=5120 N=10240  A=11.2 B=56.2 C=22.5 MB
    stress 40 iterations, 0 with mismatches
    iter 37 state=4 fill(A/B)=0/0 dram(A/B)=0/0 C: gpu=0 dram=0 late=0

`dram(A/B)` is the load-bearing number: it is compared **before** anything reads A or B back, so
zero mismatches there means the GPU's stores are in DRAM and not sitting in a dirty L2 line that
the NPU cannot see. That was the obvious failure mode for a non-coherent second device, and it does
not happen.

One caveat worth recording: an earlier revision of the harness failed 2 times in 5 with ~2000 C
bytes left at zero, and that failure has not reproduced in 81 runs since, 40 of them with new A and
B bytes every iteration. Repeating one fixed input set would have hidden it, because a stale value
equals the expected value; varying the data is what makes the comparison mean anything. If a
correctness fault shows up in the engine, this is the first thing to suspect.

### NPU time at the engine's real geometries

| M | K | N | best ms | TFLOPS |
| --- | --- | --- | --- | --- |
| 2048 | 5120 | 10240 | 6.653 | 32.3 |
| 2048 | 5120 | 9216 | 5.997 | 32.2 |
| 2048 | 17408 | 3072 | 6.667 | 32.9 |

The new xclbins are `build/npu_gu_10240` (gate/up slice) and `build/npu_dn_3072` (down slice), both
generated by `config3/n32_core.py` with the default 128x64x128 tile.

### What the split now costs per layer

Gate and up at N=10240 are 6.7 ms each and down at N=3072 is 6.7 ms, so 20.1 ms of NPU work per
layer. The GPU keeps gate/up rows [0, 7168) and down columns [0, 2048), which is 446 GFLOP against
the 1095 GFLOP the whole FFN costs, so about 17.8 ms at the 25 TF the GPU currently sustains. That
is a balanced split, and it puts the FFN floor near 20 ms instead of 43.6 ms. Nothing is copied, so
the only overhead left is the A encoder and the B repack, both GPU work the split was always going
to pay.

The down projection accumulates straight into `d_hidden` through
`LaunchBatchedQuantGEMMResidualFp16`, which is what makes a column split of it legal: the two sides
write disjoint columns of the same residual, so no partial sums are exchanged.

## XRT inside the engine: the split runs, and it is worth about 6%

`src/models/qwen/hip/atb_npu.hpp` and `.cpp` own the NPU side: the XRT device, one
hardware context and kernel per shape, and the A, B and C operands. Those operands are
`hipMalloc` allocations exported with `hsa_amd_portable_export_dmabuf` and imported as XRT
BOs, so the iGPU writes A and B in place and reads C out of them -- no stage of the split
copies a byte. `tools/qwen27b/atb_npu_run.hip` is the standalone proof of that path and is
where the ERT opcode and the handoff findings came from.

The engine's GEMM kernels index their output as `y[token * m + row]`, so a projection that
computes part of the output width writes *packed* rows, not a slice of a wide row. The GPU's
share therefore lands packed and is placed into full-width rows afterwards by
`LaunchAtbExpandHeadFp16` (gate/up) or `LaunchAtbAddHeadFp32` (down), while the NPU's C
decode writes the columns past it. Getting this wrong would have silently garbled every
activation, which is why the call site carries the reasoning.

The split is behind `GUFO_ATB_GU_XCLBIN/INSTS/NSLICE`, the same for `GUFO_ATB_DN_*`,
`GUFO_ATB_BATCH`, and `GUFO_ATB_TRACE` for per-phase timings. It is off by default.

**Two xclbins coexist.** Feeding the down instruction stream to the gate/up xclbin aborts the
queue (`qds_device::wait() unexpected command state`), which earlier looked like one shape per
process. That was wrong: a second hardware context with a different K and N is created and runs
correctly -- `tools/qwen27b/atb_two_kernels.hip` prepares both, runs both, and re-runs the
first. Only the instruction stream has to match its own xclbin.

### Measuring this box honestly

A width sweep taken in one sequence decays monotonically -- 487, 438, 420, 412 tok/s for
increasing N -- with the *width* changing far less than the decay. That is thermal and power
drift, the same effect that moves the pp2048 reference by 20% between sessions. **Any comparison
taken across sequences on this box is worthless.** Only paired or interleaved A/B inside one
session means anything, and the pair must alternate to share the thermal state.

Paired base/split/base/split, three rounds, N=8192 of the gate/up width:

| round | base tok/s | split tok/s | gain |
| --- | ---: | ---: | ---: |
| 1 | 479.93 | 509.82 | +6.2% |
| 2 | 417.92 | 449.29 | +7.5% |
| 3 | 390.53 | 412.76 | +5.7% |

The split wins every round by 5.7-7.5%. The absolute numbers fall 20% across the three rounds,
which is the drift; the paired difference does not.

### Where the rest of the projection went

Two things were ruled out by direct measurement:

| probe | result |
| --- | --- |
| fixed cost per NPU launch | none: N=1024 takes 0.770 ms (27.9 TF), N=10240 takes 6.652 ms (32.3 TF) |
| clock decay across idle gaps | none: 0/10/25/50 ms gaps all give 6.64-6.71 ms (32.0-32.3 TF) |

So the NPU is fine on its own, and the shortfall is all in how it is driven. The gate/up slice
runs while the GPU runs its own head of the same projection, but the *down* projection cannot
start until the NPU's gate and up have both completed, because its reduction spans the whole
intermediate width. That serialisation is what keeps the split near 6% instead of the ~30% a
free second engine would give: at N=8192 the per-layer wait on the NPU is about 13 ms, and the
GPU has nothing else it is allowed to run during it.

### The CPU cost is real and comes out of the same budget

gufo runs 4 threads. During the split two of them are each near 100%, for about 1.7 cores total;
the baseline spins one (the known `libhsa-runtime64` spin). System time per benchmark rises
from roughly 5 s to 9-11 s while user time falls, so the added cost is kernel time in the XRT
command path. Because the SoC is capped near 95 W, that CPU work competes with the engines it is
feeding -- the +6% above is measured *after* paying it.

Two levers follow from that, neither taken yet:

1. **Consolidate gate and up into one launch.** Their B slices can be concatenated into a single
   N = 2 x n_slice GEMM, halving both the submissions and the waits, and halving whatever
   per-command cost the XRT/amdxdna path carries.
2. **The libhsa spin costs a full core in both paths.** It is not the split's fault and not the
   split's cost, but at 95 W a core is a few percent of the budget, and it is paid on every
   prefill the engine has ever run.

## Queue-depth pacing was worth having on its own, and is now the default

The baseline path never blocks per layer: its only stream synchronisations are at the end of a
chunk. It therefore enqueues all 64 layers and then sits inside `libhsa-runtime64` on queue-full
backpressure. Measured over a pp2048 benchmark: **17-19 s of user time**, most of a core for the
whole run.

Bounding how many layers may be in flight removes it. Interleaved against no pacing, three rounds:

| depth | pp2048 (mean) | user time |
| --- | ---: | ---: |
| off | 380-395 | 17.5-18.8 s |
| 1 | 400.9 | 1.5-2.5 s |
| 2 | 402.3 | 2.6-3.1 s |
| 3 | 404.5 | 3.1-3.2 s |
| 6 | 401.9 | 3.6-4.9 s |

Depth 3 was fastest and is not sensitive, so `kPrefillPaceDepth` in `prefill_chunk.cpp` is
that value with no switch to set. The +2-3% is the cycles the spin was burning: on a 95 W-capped
SoC those are power and heat the engines do not get. Note this also means every pp2048 figure
recorded before it was measured with that core on fire.

## The split's two effects, separated

Interleaved in one session, the gate/up slice width against pp2048 and CPU:

| n_gu | pp2048 (r2 / r3) | vs baseline | user CPU |
| --- | --- | ---: | ---: |
| 0 | 395.05 / 392.42 | -- | 17.2 s |
| 1024 | 393.45 / 390.57 | ~0% | 2.4 s |
| 4096 | 410.33 / 411.69 | +4% | 2.3 s |
| 8192 | 435.17 / 434.58 | +10% | 2.1 s |

The CPU collapse appears at n=1024, where the NPU does almost nothing: it is the pacing, not the
NPU. The throughput gain tracks the NPU's share, which is the part the NPU is actually
responsible for.

## A misconfigured NPU degrades; it does not abort

XRT needs the memlock limit raised -- the hard limit here is 8 MB and XRT maps 64 MB host-only
BOs -- so the split needs `doas`. Without it `xrt::bo` throws, and the first version let that
escape and aborted the process. `Init` now catches everything including the failures XRT raises
from outside `std::exception`, and the engine falls back to the GPU path: unprivileged pp2048
prints the XRT error and completes at 491.61 tok/s.

## What is left to tune

The split is a first working implementation, not a tuned one. In rough order of expected value:

1. **Re-test the down offload.** It was rejected on a comparison that is now known to have been
   thermally confounded. The plumbing exists (`AtbRole::kDown`, `npu_dn_*` xclbins,
   `LaunchAtbAddHeadFp32`), and the question is genuinely open.
2. **Token split.** Give each engine a token subset so its gate/up -> SwiGLU -> down chain is
   independent: the NPU's gate/up then overlaps the GPU's down instead of blocking it, which is
   the serialisation that caps the split. Needs an M < 2048 xclbin and a row-offset decode, and
   the NPU needs the full weight tensors repacked.
3. **One launch for gate and up.** Their B slices concatenate into a single N = 2 x n_slice GEMM,
   halving submissions and waits. Worth doing on the CPU argument alone.
4. **Tune (n_gu, n_dn) properly**, interleaved, never across sequences.
5. **Move the B repack off the critical path.** It depends only on weights, so layer L+1's repack
   can run while layer L's NPU slice executes. It is also slow: 1.53 ms/tensor is ~93 GB/s against
   an iGPU that sustains ~200.
6. **The A encoder's gather.** One thread per 8-element group, with consecutive threads reading
   rows 10-35 KB apart.

## Correction: the power envelope is 80 W sustained, 95 W peak

Earlier sections call the package "power-capped at ~95 W". That is wrong, and tracing the rail at
2 s resolution through one `-r 3` run shows why the mistake was easy:

| t within the run | package |
| --- | ---: |
| idle | 15.4 W |
| 4 s | 91.5 W |
| 8 s | **95.9 W** (peak) |
| 10 s | 86.4 W |
| 12 s | 80.9 W |
| 14-16 s | **80.0 W** (sustained) |

**80 W sustained, with roughly an 8 s boost to 95 W.** Two consequences that explain a lot of
earlier confusion:

1. Reps inside one run are not equal. The first rep gets the boost and the rest do not, so a
   `-r 1` and an `-r 3` are measuring different regimes.
2. A run leaves Tctl near 89 C and the next run starts there, so back-to-back measurements ratchet
   upward until they plateau. This is the 20% decay seen in every uncooldowned sequence in this
   document.

Thermal decay after a run, same 2 s resolution: GPU edge recovers within ~5 s (90 to 56 C in two)
and package power within ~4 s, but Tctl has a fast component to about 52 C over ~25 s and then a
multi-minute tail (55.7 C at +10 s, 52.6 at +30 s, 51.1 at +60 s, 48.7 at +138 s).

### Measurement protocol from here on

Cooldown before every point, gated on the sensors rather than a fixed time: Tctl <= 51 C and
package power <= 20 W. Measured cost 30-39 s per point, so a five-point sweep is about eight
minutes: cheap next to being wrong.

Four baseline points taken that way:

| point | cooldown | start Tctl | pp2048 |
| --- | ---: | ---: | ---: |
| 1 | 0 s | 48 C | 546.73 |
| 2 | 30 s | 51 C | 556.69 |
| 3 | 36 s | 50 C | 543.89 |
| 4 | 39 s | 50 C | 545.79 |

Mean 548.3, spread 1.2%, against 20% drift for the same binary run back to back. Every comparison
in the tuning work that follows uses this protocol.

## The NPU's concurrency tax is DRAM bandwidth, not power

An earlier section concluded "this is a power split, not memory-bandwidth contention" from the fact
that the concurrent power ceiling did not exceed the GPU-solo ceiling. That inference does not
hold -- a bandwidth limit produces the same observation -- and direct measurement says it is
bandwidth.

**The power mode is a real clock control, and it is already at peak.** `xrt-smi configure --pmode`
takes default, powersaver, balanced, performance and turbo. Solo NPU at N=8192, K=5120:

| pmode | best ms | TFLOPS |
| --- | ---: | ---: |
| Powersaver | 12.048 | 14.26 |
| Balanced | 7.577 | 22.67 |
| Default | 5.326 | 32.25 |
| Performance | 5.305 | 32.38 |
| Turbo | 5.296 | 32.44 |

Powersaver and balanced are genuine throttles, so the knob works. But `default` is already at the
top level, so there is no headroom to raise, and under the split the three top modes are
indistinguishable: pp2048 at n_gu=7168 was 594.4 (default), 593.7 (performance), 590.5 (turbo).
Nothing to recover here.

**The NPU's best case is untouched by the engine; only its mean moves.** Same shape, one tight
loop, measured three ways:

| load | best | mean |
| --- | ---: | ---: |
| none | 5.310 ms | 5.381 ms |
| 8 CPU hogs | 5.309 ms | 5.801 ms (+7.8%) |
| the engine prefilling | 5.305 ms | 6.829 ms (+26.9%) |

So the ~20-26% "derate" is not a lower clock: the best run under full engine load is identical to
solo. About eight points of the mean is the measuring thread being descheduled (the CPU-hog
control), and the rest is interference in execution. The engine's own CPU use accounts for a
couple of those points, not eight.

**Which interference?** Two synthetic GPU loads, each held for the whole NPU loop:

| concurrent GPU load | NPU best | NPU mean | best TF |
| --- | ---: | ---: | ---: |
| none | 5.326 ms | 5.368 ms | 32.26 |
| compute (register-resident FMA, no DRAM) | 6.109 ms | 6.667 ms | 28.12 |
| memory (streaming copy) | 8.499 ms | 9.697 ms | 20.21 |

A saturating DRAM load costs the NPU 60% *on its best run*; an ALU-only load costs about 15%. The
NPU is dominated by DRAM bandwidth and latency, not by the power budget it shares with the GPU.
That also explains why powersaver and balanced look like clock throttles: fewer fetches per unit
time is what a lower clock buys on a memory-bound workload.

For the record, the engines do not take power from each other either: package power was 82.8 W mean
/ 95.2 W peak for the engine alone and 79.8 W mean / 92.6 W peak when the NPU loop ran alongside
it, and the engine lost 0.6% (545.34 to 541.87 tok/s).

**What this means for the split.** The width tuning already prices the contended rate, so the +8.4%
at n_gu=7168 is real. What is left is not a clock but DRAM pressure: the split adds a B repack that
writes and then re-reads the NPU's whole weight slice, which the engine would not otherwise touch.
That is the lever -- reduce or hide that traffic -- not the power mode.

### Correction: it is two effects, and the steady one is power after all

The section above reads as "bandwidth, not power". Measuring the whole *distribution* of per-command
times separates two mechanisms that the mean alone conflates. 2000 runs each, same shape:

| load | min | p10 | p25 | p50 | p75 | p90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 5.304 | 5.331 | 5.340 | 5.355 | 5.370 | 5.388 |
| compute (register-resident FMA) | 5.309 | 6.831 | 6.862 | 6.883 | 6.903 | 6.920 |
| memory (streaming copy) | 5.304 | 5.332 | 5.352 | 9.726 | 9.954 | 10.032 |

**A compute-only GPU load shifts the entire distribution, uniformly, by about 28%.** That kernel
never touches DRAM, so this is not bandwidth. It is shared-power arbitration, and the size says
which level the SMU picks: 22.67 / 32.25 = 0.70, i.e. a busy GPU appears to drop the NPU to close
to its `balanced` power mode (22.7 TF). That also explains why setting `--pmode performance` or
`turbo` changes nothing under load -- the driver is not choosing the level.

**A memory load is bimodal.** The bottom quartile of commands run at full solo speed (5.33, 5.35)
and the rest sit at ~9.7-10.0 ms, 83% slower. That is a collision effect: a command whose operand
fetches do not coincide with the GPU's bursts runs at full speed, and one that does stalls. The
best and the min stay pinned at 5.304 in every case, which is the signature of an intermittent
stall rather than a reduced clock.

So the ~20-27% seen under the engine is the *steady* component -- the engine's GPU work is steady,
so the NPU spends most commands in the arbitrated level -- with occasional extra stalls when the
engine's memory traffic spikes. Two different remedies, neither of which is a driver knob:

1. The steady 28% needs the SMU to give the NPU a better level while the GPU is busy. The available
   dial is the platform profile (`balanced` today), which is the user's setting and has not been
   touched.
2. The intermittent stalls need the NPU's fetches to stop colliding with the GPU's bursts, either by
   scheduling or by deepening the design's L3/L2 FIFOs so a command can ride out a latency spike.

## How XDNA2 power management actually works (from the running kernel's own source)

The driver source ships with this kernel, so this is read rather than inferred:
`/lib/modules/` + `$(uname -r)` + `/build/drivers/accel/amdxdna/`. Files that matter:
`aie2_pm.c` (mode mapping), `npu4_regs.c` (clock table), `aie_smu.c` (mailbox),
`aie2_solver.c` (per-context level), `aie2_pci.c` (the setter).

**Performance is a discrete DPM level, requested over the SMU mailbox.** `npu4_dpm_clk_table` has
eight levels of `{npuclk, hclk}` in MHz, from {396, 792} to {1267, 1800}. The modes
`xrt-smi configure --pmode` offers map onto them directly, and the measured throughput matches
the table to within a percent, which is what confirms the model:

| level | hclk | predicted vs level 7 | measured pmode |
| --- | ---: | ---: | --- |
| 0 | 792 | 0.44 | powersaver 14.26 TF |
| 3 (max/2) | 1267 | 0.70 | balanced 22.67 TF |
| 7 | 1800 | 1.00 | default 32.25 / performance 32.38 / turbo 32.44 TF |

`turbo` differs from `performance` only by disabling clock gating, and
`aie2_pm_set_mode` refuses it outright while a hardware context exists
(`if (ndev->hwctx_num) return -EINVAL`), so it can never help a running workload.

**The driver never learns what the SMU granted.** The NPU's SMU interface has exactly five messages:
`POWER_ON`, `POWER_OFF`, `SET_MPNPUCLK_FREQ`, `SET_HCLK_FREQ`,
`SET_SOFT_DPMLEVEL`, `SET_HARD_DPMLEVEL`. There is no query, no read-back, and sysfs exposes
only `vbnv`, `device_type` and `fw_version` while debugfs exposes only `carveout` and
`name`. `ndev->npuclk_freq` and `hclk_freq` are the *requested* values cached in software.
Whatever the SMU actually grants is invisible from the host, so the ~28% steady tax under GPU load
can only be inferred from timing -- which is what the distributions above do.

**The SMU is the arbiter, and the NPU has no lever above it.** The driver's highest request is already
the top level, sent by both `default` and `performance`. When the GPU is busy the SMU clamps
whatever it grants, and the clamp size we measured (0.70) is exactly the table's level 3. Nothing in
XRT, the driver, or the engine can override that.

**One real and non-obvious trap.** `aie2_solver.c` picks the level per context load:

    /* If no QoS parameters are passed, set it to the max DPM level */
    if (!is_valid_qos_dpm_params(rqos)) { level = max_dpm_level; goto set_dpm; }
    for (level = 0; level < max_dpm_level; level++) { ... qos_meet ... }

A context that advertises QoS parameters (`gops`, `fps`, `latency`) gets the *lowest* level
that just meets them. Our ATB design advertises none, so it correctly gets the top level -- but an
xclbin that ever grew QoS metadata would silently lose up to 56% of throughput with no error and no
readback to notice it by.

**So the steady 28% is closed.** It is a SMU decision inside a shared power budget, invisible and
un-influenceable from the NPU side, and the platform profile is the user's custom one.
What remains reachable is the *intermittent* half, which is on our side: the bimodal stalls when
the NPU's fetches collide with the GPU's memory bursts.

### The configuration layer is package-level, and it is not NPU-specific

The phrase "SMU contention" resolves to a concrete layer: `amd_pmf`'s package power limits. Its
custom-policy interface (`tee-if.c`) can set exactly this list:

    PMF_POLICY_SPL              sustained power limit
    PMF_POLICY_SPPT / FPPT      slow and fast package power tracking
    PMF_POLICY_SPPT_APU_ONLY / PMF_POLICY_PMF_PPT_APU_ONLY
    PMF_POLICY_STT_MIN, STT_SKINTEMP_APU, STT_SKINTEMP_HS2
    PMF_POLICY_P3T, PMF_POLICY_SYSTEM_STATE, 10 BIOS_OUTPUT_n slots

Every entry is a *whole-package* limit. **There is no NPU-specific policy item anywhere in it**, so
there is nothing to configure that would give the NPU a larger share while the GPU is busy. That
matches the driver: the NPU requests its DPM level (already the top one) and the SMU decides from
the package policy.

On this machine the tuning surfaces are not even present. `amd_pmf_dbgfs_register` only creates
`current_power_limits` when `pmf_if_version == PMF_IF_V1`, and `/sys/kernel/debug/amd_pmf/` is
empty; the `pb/update_policy` upload lives behind a build option that this kernel did not take,
since there is no `pb/` directory either. Consistent with the operator using a custom profile
applied elsewhere.

### The NPU's live power, and a number that does not fit the earlier story

`xrt-smi examine -r platform` reports the NPU's own power. Reading it while a loop runs gives
the missing half of the clamp evidence:

| condition | NPU power | NPU mean | best |
| --- | ---: | ---: | ---: |
| NPU only | **1.517 W** | 5.355 ms | 5.302 |
| + GPU compute | **1.119 W (0.74x)** | 5.877 ms | 5.302 |
| + GPU memory | **1.474 W (0.97x)** | 6.465 ms | 5.303 |

A busy GPU drops the NPU to 0.74x power, which is the level-3 clamp from the clock table measured
rather than inferred. A memory load leaves the clock alone (0.97x) and stalls it instead. Two
mechanisms, both now confirmed from hardware telemetry rather than from timing.

But **the NPU draws about 1.5 W at full tilt (32.4 TF)**. That sits badly with the earlier claim in
this document that the NPU is "31 TFLOPS for ~30 W" and that the two engines split a power budget:
a 1.5 W block is not squeezing an 80 W envelope. Whatever drops the NPU's clock when the GPU is
busy, it is not the NPU's own consumption. Two candidates remain open -- a policy reservation that
does not reflect the measured power, or a shared clock domain whose divider moves with the GPU's
state (which would explain why the drop lands on a discrete table level rather than varying
smoothly). Both are outside anything the engine, XRT or the NPU driver can set.

### Found it: the contention is configured through the ASUS WMI power limits, not amd_pmf

The NPU's granted clock is decided by the SMU from the package power limits, and on this machine
those limits are ordinary writable sysfs attributes. Two interfaces expose them, and they do not
agree with each other:

    /sys/devices/platform/asus-nb-wmi/          (mode 0666)
      ppt_pl1_spl           80
      ppt_pl2_sppt          80
      ppt_apu_sppt          80
      ppt_platform_sppt     80
      ppt_fppt              80

    /sys/devices/virtual/firmware-attributes/asus-armoury/attributes/
      ppt_pl1_spl  "Set the CPU slow package limit"      60   [28..80]
      ppt_pl2_sppt "Set the CPU fast package limit"      75   [32..92]
      ppt_pl3_fppt "Set the CPU fastest package limit"   86   [45..93]

The armoury interface is the ranged, documented one and goes up to 92 on the fast limit. Which of
the two the SMU actually takes is not yet established -- writing one and reading the other is the
test -- but the sustained package draw measured under GPU load was exactly **80.0 W**, which is the
WMI value and not any of the armoury ones.

**This is the lever, and it means a kernel patch is probably unnecessary.** The contention is not
hidden in the SMU: it is the package power limit, it is exposed by mainline `asus-nb-wmi`, and it
is bounded and reversible. Raising the fast/second limit (75 -> 92 by its own range, or the WMI
platform limit above 80) gives the SMU headroom, and the prediction from the measurements above is
concrete: the NPU's power should return from ~1.0 W toward 1.5 W and its mean command time from
~6.5-9.5 ms back toward 5.3, because the 0.64-0.67x power collapse is the level-3 clamp.

**It is a power change, so it is the operator's call**, and the cost is heat: the package already
runs at 80-85 W with Tctl 86-90 C and both fans at 6100 RPM during these runs, and skin temperature
limits (`STT_SKINTEMP_*`) are in the same policy family.

The precedence question is also worth settling first, because it decides whether the write should go
to `ppt_pl2_sppt` or `ppt_platform_sppt` / `ppt_apu_sppt`: the APU limit and the platform limit are
different domains, and the NPU lives inside the APU.

## The NPU draws about 21 W, not 1.5 W -- and the metric that said otherwise

An earlier section reports the NPU at ~1.0-1.5 W under load and calls it anomalous. It is an
artifact of how the number is produced. `amd_pmf` does not sample NPU power; it reads an
*accumulator* from the SMU metrics table and turns it into a rate:

    metrics.c: val_diff = curr_value - prev_value; return val_diff / counter_diff;
               data->npu_power = amd_pmf_q10_acc_to_mW(val);   /* acc/1024*1000, capped at u16 */

The value is the average over the interval **between two sensor reads**, not an instantaneous
power. A read taken during a short burst after a long idle is averaged against the idle time, and
the first read after the device has been unused returns `N/A` for the same reason. Polling during a
steady load gives a usable number, but it is not ground truth.

Package power is. Same shape (N=8192, K=5120) held for 16 s at 130 W:

| state | package | delta |
| --- | ---: | ---: |
| idle | 18.2 W | -- |
| NPU at full tilt, 32.39 TFLOPS | 39.5 W | **+21.3 W** |
| GPU compute load only | 87.0 W | -- |
| GPU compute load + NPU | 72.9 W | **-14.1 W** |
| GPU memory load only | 82.0 W | -- |
| GPU memory load + NPU | 80.0 W | **-2.0 W** |

**The NPU is not a 1.5 W device.** At full tilt it moves the package by ~21 W on an otherwise idle
box, which is ~1.5 TFLOPS/W for the 32.4 TF int8 GEMM and matches the older "31 TF for ~30 W"
figure that the 1.5 W reading had contradicted. The sensor was wrong; the old estimate was right.

**And its marginal cost on top of an already-active GPU is zero, even negative.** The 21 W is mostly
shared SoC infrastructure -- fabric, memory controllers, SLC -- that the GPU has already powered up.
Adding the NPU to a busy GPU adds no package power at all; it slightly *lowers* it, because the NPU
is throttled in the shared budget and the GPU's clock is pulled back with it.

That also means the split's power story is not "the NPU spends 21 W". It is "the NPU spends nothing
the GPU was not already spending, but it takes execution slots in a shared budget".

## The GPU does not stall more at higher power -- it barely clocks up at all

The suggestion was that extra package power cannot become throughput because the GPU is internally
data-movement bound, so more power just buys more stalls. Sampled at 0.1 s through a full prefill,
almost the opposite shows up: **throughput tracks the clock almost exactly, and it is the watts that
fail to become clock.**

Same benchmark, 0.5 s sampling of `pp_dpm_sclk` and `gpu_busy_percent`:

| limit | mean clock | max clock | pp2048 | TFLOPS | TFLOPS per MHz |
| --- | ---: | ---: | ---: | ---: | ---: |
| 80 W | 1799 MHz | 1875 MHz | 479.13 | 25.87 | 0.014378 |
| 130 W | 2016 MHz | 2221 MHz | 528.73 | 28.55 | 0.014161 |

+12.1% clock and +10.35% throughput: the work done per clock falls by **1.5%**, i.e. there is
essentially no extra stalling. What is broken is the conversion of watts into clock. Mean package
draw went 71.3 -> 88.5 W (+24%) for +12% clock, the GPU never reaches its top level (2900 MHz; the
sampled maximum is 2221), and `gpu_busy` sits at ~85% at both limits.

### The boost window is the clearest case: 14 W buys a *lower* clock

A single `-r 1` prefill at `ppt=80`, 0.1 s resolution, reproduces the original test case -- the ramp
to ~96 W and the step back to the 80 W cap -- and shows what the boost actually buys:

| t (s) | package | gpu busy | sclk |
| ---: | ---: | ---: | ---: |
| 1.1 | 23.9 W | 5% | 1445 MHz |
| 3.8 | 85.0 W | 87% | 2039 MHz |
| 7.6-9.6 | **95.8-96.0 W** | 100% | 2023-2054 MHz |
| 9.69 | 94.8 W | 100% | 2128 MHz |
| 10.13 | **81.8 W** | 88% | **2135 MHz** |

The plateau ends at ~9.6 s and the package steps down to the cap. Through the whole 95.9 W plateau
the clock is 2023-2054 MHz; the moment the power falls, the clock **rises** to 2135 MHz. The 14
extra watts of boost are not going to the GPU clock at all.

**Consequence for the split.** If the GPU is not throttling on data movement but has a steep V/f
curve near its ceiling, then 130 W buying only +6.4% is expected, and the headroom is in *replacing*
GPU work with NPU work -- which is exactly why the split's gain grows from +5.1% at 80 W to +8.5% at
130 W -- not in feeding the GPU more watts.

### Measurement regime, and why the earlier baselines disagreed

`-r 3` averages three reps inside one process and only the first gets the boost. A single `-r 1` run
at 80 W reads **530.30 tok/s**, while the sensor-gated `-r 3` protocol on the same box, same session,
same binary reads **493.7** -- a 7% regime difference with nothing to do with clocks or power
settings. The 543.93 standing reference and the 15 s-gap numbers (477) are further protocols again;
only same-protocol comparisons mean anything.

Sensor-gated (`Tctl <= 51 C`, package <= 20 W), fan at the original curve:

| limit | baseline | split (n_gu=7168) | split gain |
| --- | ---: | ---: | ---: |
| 80 W | 504.03 / 483.44 | 517.53 / 520.57 | **+5.1%** |
| 130 W | 523.38 / 526.88 | 570.19 / 569.03 | **+8.5%** |

The gate needs relaxing: this box idles at 52 C, so a `Tctl <= 51` gate can never open and the wait
loop spins until its timeout. A run that "never started" is this bug, not a hang.

## The ASUS fan curve is a two-step write, and maxing it changes nothing

`asus-nb-wmi`'s `asus_custom_fan_curve` hwmon has a trap that silently discards an edit. From the
running kernel's `asus-wmi.c`:

    fan_curve_store()       -> data->enabled = false;         /* on every point write */
    fan_curve_write()       -> if (!data->enabled) return 0;  /* line 3700 */
    fan_curve_enable_show() -> out = data->enabled ? 1 : 2;

Writing any `pwm*_auto_point*_pwm` disables the curve, and `fan_curve_write` then returns without
touching the EC. The curve only takes effect on `echo 1 > pwm*_enable` **after** the last point
write, and `pwm*_enable` reads **1 when enabled and 2 when disabled** -- the reverse of the usual
hwmon reading, which makes the two easy to confuse. There are two independent curves (`pwm1` = CPU,
`pwm2` = GPU) and 255 is 100% (`100 * percents[i] / 255`, line 3807).

The first attempt at this experiment wrote the points and left the curve disabled, so it measured
nothing. Done correctly, on both fans:

| fans | peak RPM | peak Tctl | peak package | 80 W base | 130 W base |
| --- | ---: | ---: | ---: | ---: | ---: |
| EC automatic | 8200 / 8500 | 95.2 C | 110.2 W | 477.6 / 477.1 | 512.3 / 509.6 |
| both curves at 100% | 8900 / 8900 | **95.25 C** | **103.3 W** | 479.7 | 520.0 |

100% adds 400 RPM but no cooling: peak Tctl is identical to two decimals and the peak package is
*lower*. The EC's automatic control is already near saturation under load, so the fan is not a
lever. The original curve is saved in `build/atb/fan_curve_backup.txt` and has been restored.

## z13ctl is the same surface, with the profile as the live path

`/usr/local/bin/z13ctl` (Go, socket at `$XDG_RUNTIME_DIR/z13ctl/z13ctl.sock`) drives the same
`asus-nb-wmi` PPT files for `tdp`, the `platform-profile` attribute for `profile`, and the same
`asus_custom_fan_curve` hwmon for `fancurve`. It is not a different power path -- setting 80 through
it produces the identical `ppt_*` = 80 state, and the trace above is that setting.

### The clamp does not release at 93 W, and it is a cliff, not a ramp

The NPU's unmet need under contention is ~0.5 W by its own metric, so the package limit needed to
release it should be trivial. It is not. Same shape under the real engine prefill (baseline path, no
split), 2000 commands, ppt = 93 W:

| state | best | p50 | mean | TFLOPS |
| --- | ---: | ---: | ---: | ---: |
| NPU solo | 5.306 | 5.355 | 5.359 | 32.38 |
| under engine, **80 W** | 5.305 | -- | 6.829 | -- |
| under engine, **93 W** | 6.199 | 6.925 | **6.960** | 27.72 |
| under GPU compute load, **130 W** | 5.302 | -- | **5.347** | 32.3 |

The time-ordered deciles at 93 W are flat -- `6.902 6.931 6.958 6.908 6.933 6.926 6.990 7.020 7.049
6.986` -- so there is no partial release, and none as the engine's boost budget expires and its
package draw falls from 111 to ~80 W. The command time is pinned at ~6.9 ms for the whole run.

That is the shape of the injustice exactly as the operator put it: **the SMU withholds ~0.5 W from
the NPU and charges 29-33% of its throughput (level 7 -> level 3, 0.70x), while the GPU spends an
extra 12-30 W for +1.6% of baseline throughput.** The exchange rate on the NPU side is ~60x better
than on the GPU side, and the arbiter spends the budget on the wrong one.

The engine's own baseline is indifferent to the limit in this range -- 530.30 (`-r 1`, 80 W),
538.83 (`-r 1`, 93 W), 531.54 +/- 9.90 (`-r 3`, 93 W) -- so the only thing that changes with the
limit is whether the NPU is allowed to work. **The knee is between 93 and 130 W, and the optimum
standing limit for a split workload is therefore above the optimum for a baseline workload.**

### The NPU's budget is not a separate domain -- the platform/APU split is inert

The "trick the firmware" route was to feed the NPU out of a different PPT domain, since the five
`ppt_*` files are documented as different limits (platform SPPT vs APU-only SPPT vs sustained SPL).
Tested one variable at a time from the all-80 state, NPU running under the real engine prefill,
1500 commands each:

| arm | spl / sppt / apu / plat / fppt | engine pp2048 | NPU best | NPU mean | deciles | package |
| --- | --- | ---: | ---: | ---: | --- | --- |
| A control | 80 / 80 / 80 / 80 / 80 | 457.20 +/- 21.17 | 6.947 | 7.666 | flat ~7.65 | 70.8 mean / **80.0 max** |
| B platform | 80 / 80 / 80 / **130** / 80 | 457.73 | 6.921 | 7.649 | flat ~7.65 | 70.8 / **80.0** |
| C apu | 80 / 80 / **130** / 80 / 80 | 459.21 | 6.903 | 7.647 | flat ~7.65 | 71.1 / **80.0** |
| D all | **130 / 130 / 130 / 130 / 130** | **499.50 +/- 5.28** | 5.314 | **5.618** | 6.084 -> **5.472** | 92.3 / 120.9 |

**B and C are inert.** Raising `ppt_platform_sppt` alone, or `ppt_apu_sppt` alone, changes neither
the engine, nor the NPU (7.649 and 7.647 against the control's 7.666), nor the package -- which sits
at exactly 80.0 W in all three, the value of `ppt_pl1_spl`. The sustained limit is the one that
binds, and it binds both engines together: there is no domain in which the NPU can be fed while the
GPU stays constrained. **The "trick the firmware" idea is closed by measurement.**

Only all five at 130 releases the clamp (mean 5.618 against a solo 5.359), and its deciles fall
monotonically, 6.084 -> 5.472, settling at solo speed. Note the engine under *concurrent* NPU load is
+9.2% at 130 W (499.50 vs 457.20), larger than the +6.4% it gains with no NPU running -- when the
NPU is also competing for the budget, opening the budget matters more.

## The SMU reports the clamp directly, and it says the NPU is at 0.9 W

The claim two sections ago -- "whatever the SMU actually grants is invisible from the host" -- is
wrong, and the correction is the strongest evidence in this document. `amdgpu`'s `gpu_metrics`
sysfs file is a 264-byte `struct gpu_metrics_v3_0` (`kgd_pp_interface.h`), world-readable, and for
an APU it is the whole SMU metrics table -- including a first-class IPU block and the throttle
counters:

    uint16_t average_ipuclk_frequency;   // time filtered target IPUCLK [MHz]
    uint16_t average_mpipu_frequency;    // time filtered target MPIPUCLK [MHz]
    uint16_t average_ipu_activity[8];    // per-column busy % [0-100]
    uint16_t average_ipu_power;          // time filtered IPU power [mW]
    uint16_t average_ipu_reads, average_ipu_writes;      // [MB/sec]
    uint32_t average_socket_power, average_apu_power, average_gfx_power, average_all_core_power;
    uint16_t stapm_power_limit, current_stapm_power_limit;
    uint16_t average_gfxclk_frequency, average_fclk_frequency, average_uclk_frequency;
    uint16_t current_gfx_maxfreq;        // GFXCLK limit enforced on GFX [MHz]
    uint32_t throttle_residency_prochot, _spl, _fppt, _sppt, _thm_core, _thm_gfx, _thm_soc;
    uint32_t time_filter_alphavalue;     // metrics table alpha filter time constant [us]

There is a second path to the same data. `npu4_update_counters()` (`npu4_regs.c`) overwrites the
driver's *cached request* with the SMU's granted values before every clock query:

    ret = AIE2_GET_PMF_NPU_METRICS(&npu_metrics);
    ndev->npuclk_freq = npu_metrics.mpnpuclk_freq;   // granted MP-NPU
    ndev->hclk_freq   = npu_metrics.npuclk_freq;     // granted H clock
    ndev->curr_tops   = NPU4_DPM_TOPS(ndev, ndev->hclk_freq);

so the `DRM_AMDXDNA_QUERY_CLOCK_METADATA` / `QUERY_RESOURCE_INFO` ioctls return the granted level,
not the request (`build/atb/npu_clk.c`, 30 lines, no privileges). `NPU4_DPM_TOPS` is
`4096 * cols * hclk / 1e6`, which is where the `tops_max=58` at 1800 MHz comes from.

**Measured through gpu_metrics at ppt = 93 W, engine prefill with the NPU running alongside:**

| quantity | value |
| --- | --- |
| IPUCLK / MPIPUCLK granted | **1285 / 985 MHz** (peak 1286 / 986) |
| DPM level 3 is | {npuclk 975, hclk 1267} |
| DPM level 7 is | {npuclk 1267, hclk 1800} |
| IPU per-column busy | **99% on all 8 columns** |
| **IPU power (SMU's own field)** | **0.94 W** (peak) |
| GFXCLK | 1860-1979 MHz |
| current_gfx_maxfreq during load | **1705-1985 MHz** (never 2900; 2900 only at idle) |
| socket / gfx / all-core power | 93.0 W (at the cap) / ~22 W / ~4-5 W |
| STAPM limit | **66 / 66 W** (below the 93 W being drawn) |
| throttle residency delta over the run | SPL +12207, FPPT +17087, SPPT +25, thm_gfx +1538, PROCHOT 0 |

**1285 / 1267 = level 3. 985 / 975 = level 3.** The granted clock is the level-3 row of the DPM
table to within 1.5%, which is the clamp the timing data inferred, now read straight out of the
SMU's own table. The NPU is at **99% utilization on every column and 0.94 W** while held at level 3,
and the GFX is simultaneously capped by the SMU at 1705-1985 MHz against a 2900 MHz ceiling, with
SPL *and* FPPT *and* GFX-thermal residency all accumulating.

That is the operator's claim as a table: **the arbiter withholds well under a watt from a
99%-utilised IPU, costs it 29-33% of throughput, and simultaneously caps the GFX it is feeding at
60-68% of its ceiling.** The exchange rate is ~60x better on the IPU side and the arbiter spends
the budget on the wrong one.

Two measurement notes. The field is explicitly **time filtered** (`time_filter_alphavalue` 1000000,
tau ~1 s): through the same run IPUCLK reads 499, 697, 994, 1186 and only then settles at 1270-1286,
so a single sample during a short load understates it -- the same trap that made the NPU look like a
1.5 W part. And the IPU power block sums to ~27 W of the 93 W drawn (gfx 22 + cores 4.5 + ipu 0.9),
so the SMU itself does not attribute the remaining ~66 W to any engine.

### What the firmware blobs did and did not yield

Both are present and both were unpacked (`/lib/firmware`, zstd on this distro):

* **SMU / PMFW** -- `amdgpu/smu_14_0_3.bin`, 333236 bytes, header parses as
  `{size 0x515b4, hdr 0x2c, seg 0x4fe00/0x4ff00, extra 0x16b4}`; the 0x4ff00 field is exactly the
  size of `smu_14_0_3_kicker.bin` (327424) and 327424 + 5812 = 333236. It contains plain data
  tables: at 0x7552 there is a clock ladder `1000, 1100, ... 1800`. But there are no symbols, no
  strings and no ELF, and the IPU clock values do not appear as a table -- the driver sends
  MPNPUCLK/HCLK numerically, so the SMU has no IPU table to find. Nothing about the arbitration
  policy is recoverable this way; it is custom-ISA code.
* **NPU** -- `amdnpu/17f0_11/npu.sbin.1.1.2.65`, 429680 bytes. Not ELF, no strings except
  `Release 1.1.2.65` glued to a long hex blob, i.e. a signed/obfuscated image. Not disassemblable
  in practice.

**The interface headers were the prize, not the binaries.** `smu14_driver_if_v14_0_0.h` documents the
IPU as a first-class power domain with a clock, a power, per-column busy and read/write bandwidth --
and yet no field anywhere sets its share. That is the whole finding in one sentence: the mechanism
is documented, the telemetry is complete, and the allocation has no knob.

## What the kernel can and cannot do about it

The firmware arbitrates, but the kernel *feeds* it, and one kernel driver on this box is a live power
policy engine rather than a passthrough: **`amd_pmf`**, which is loaded and is what supplies the NPU
metrics to amdxdna (`amd_pmf 94208 1 amdxdna`).

**The live lever is `platform_profile`.** The profile class core's `_store_class_profile` walks every
registered handler and calls `profile_set` on each whose choices include the bit, so one write drives
*both* providers present here:

    /sys/class/platform-profile/platform-profile-0  name=amd-pmf    (AMDI0105:00, SPS)
    /sys/class/platform-profile/platform-profile-1  name=asus-wmi   (throttle_thermal_policy)

and `amd_pmf` maps the profile onto a power mode and then pushes the package limits itself
(`sps.c`):

| platform_profile | amd_pmf mode | Smart PC TA slider |
| --- | --- | --- |
| performance / balanced_performance | `POWER_MODE_PERFORMANCE` | `AMD_PMF_TA_BEST_PERFORMANCE` |
| **balanced (current)** | `POWER_MODE_BALANCED_POWER` | `AMD_PMF_TA_BETTER_PERFORMANCE` |
| low_power / quiet | `POWER_MODE_POWER_SAVER` | `AMD_PMF_TA_BEST_BATTERY` |

`amd_pmf_set_sps_power_limits()` then writes SPL, FPPT, SPPT, **SPPT_APU_ONLY**, STT_MIN and the
STT_LIMIT entries through APMF to the EC/SMU. So the profile is a kernel-mediated knob on exactly the
limits that clamp the NPU, and the box is sitting in the middle of the three.

Two things follow that matter. First, **`stapm_power_limit` reads 66/66 W while the package draws
93 W**, so something other than the WMI ppt value is producing the sustained limit, and SPS/STT is
the prime suspect -- which makes the profile a direct test rather than a side issue. Second, the
Smart PC / CNQF path that `smart_pc_support=Y` enables is **not** registered here: its debugfs
(`pb/update_policy`, `current_power_limits`) is absent, so SPS is what is actually live. And
`metrics_table_loop_ms` defaults to 1000, which is the ~1 s time constant behind every filtered
Averaging field in this document.

**The PMF interface does carry NPU-specific rail limits** -- the first NPU-specific quantities found
anywhere in the stack (`pmf.h`):

    vddcr_npu_set_voltage, vddcr_npu_telemetry_voltage, vddcr_npu_telemetry_power
    tdc_vddcr_npu_limit, tdc_vddcr_npu_fused_limit,
    tdc_vddcr_npu_max_irm_limit, tdc_vddcr_npu_max_pbo_limit
    tdc_vddcr_npu_value_acc, tdc_vddcr_npu_residency_acc
    npuclk_freq, npu_power, npu_busy[8], mpnpuclk_freq, npu_reads, npu_writes   (enact table input)

The NPU has its own **VDDCR_NPU rail with a TDC (current) limit** and the enact table receives its
clock, power and per-column busy. Whether `tdc_vddcr_npu_limit` is what bounds it is unknown, but it
is the one NPU-specific limit that exists, and it is consumed by the signed TA rather than exposed as
a sysfs attribute.

**So: yes, the kernel affects this, and no, it cannot redirect the split.** Every kernel-side knob --
`platform_profile`, SPS's SPL/FPPT/SPPT/STT set, CNQF, `throttle_thermal_policy`, the five
`ppt_*` -- moves the size of the whole-package budget, and the budget is shared. The decision that
divides that budget between GFX and IPU lives in the SMU firmware's arbitration, and where Smart PC
is active, in an AMD-signed TA. The NPU driver's own vocabulary is five SMU messages, of which the
clock request is already the top DPM level, and the SMU grants less than asked. There is no kernel
path that asks for the IPU's share, because no such request exists in the interface.

## Could a kernel patch feed the NPU more?

The direct route does not exist, and the reason is now precise rather than rhetorical.

**The SMU's clock messages cannot name the NPU.** `SetSoftMinByFreq`, `SetSoftMaxByFreq`, `SetHardMinByFreq`,
`SetHardMaxByFreq`, `GetDpmFreqByIndex`, `GetMinDpmFreq` and `GetMaxDpmFreq` are all addressed by a
`PPCLK_e` clock id, and that enum is (`smu14_driver_if_v14_0.h`):

    PPCLK_GFXCLK, PPCLK_SOCCLK, PPCLK_UCLK, PPCLK_FCLK, PPCLK_DCLK_0, PPCLK_VCLK_0,
    PPCLK_DISPCLK, PPCLK_DPPCLK, PPCLK_DPREFCLK, PPCLK_DCFCLK, PPCLK_DTBCLK

**There is no IPU or NPU clock id.** So the amdgpu SMU driver has no register in its vocabulary for the
NPU clock -- it can neither set it nor read it. The NPU driver's vocabulary is the five messages
already described, and its clock request is already the top DPM level. A patch can therefore ask
louder, but there is nothing quieter to ask.

**One direct bit does exist.** The PMFW feature set carries `FEATURE_IPU_DPM_BIT = 19` (and
`FEATURE_DS_IPUCLK_BIT = 58`), and `PPSMC_MSG_SetAllowedFeaturesMaskLow/High` is sent by the driver --
so a patch could **exclude bit 19 and disable the SMU's IPU DPM block**, on the theory that the
NPU driver's own `SET_MPNPUCLK_FREQ` then stands. That is the only direct patch with a mechanism, and
it is a gamble: with IPU DPM off the clock may pin anywhere, including lower, and the NPU firmware's
own DVFS may depend on the feature. Worth knowing it exists; not worth betting the IPU on without a
way to read the grant, which the `gpu_metrics` decoder now provides.

**The indirect route is real, and the kernel says which one it intends.** The GFX branch above
deliberately picks the *soft* messages, and the comment states the reason: *"SoftMin lets PMFW
throttle gfxclk; HardMin would override SoftMax."* The kernel is structured to hand the arbitration to
the firmware. The clamp on the NPU is that same arbitration, so the way to un-clamp the NPU from the
kernel is to stop exhausting the budget with the GPU.

The duty sweep already proved the mechanism: at 75% GPU duty the package peaks at 84.3 W, under the
93 W cap, and the NPU's execution floor is still 5.30 ms; at 100% duty the package sits exactly on the
cap and the floor collapses to 6.83 ms. **Hold the package under the cap and the NPU keeps level 7.**

And that needs no patch, because the GFX side is already exposed and writable:
`/sys/class/drm/card1/device/power_dpm_force_performance_level` (currently `auto`) and
`pp_dpm_sclk`, both root-writable. Capping GFX to keep the package clear of the cap is the one trade
the numbers say should be net-positive -- the GPU returns +1.6% per 13 W near its ceiling while the
NPU returns 29-42% for the sub-watt it was denied -- but the optimum is an empirical balance point,
because capping GFX also slows the GPU's share of the split.

Caveat that keeps this honest: at ppt 93 W the metrics attribute only 22 W to GFX, 4.5 W to the cores
and 0.9 W to the IPU, leaving **~66 W unattributed**. If the budget is being consumed by that
unattributed part rather than by the GFX clock, capping GFX will not un-clamp anything. That is the
experiment, and it is falsifiable in one run: cap GFX, watch `average_ipuclk_frequency` in
`gpu_metrics` -- 1267 means still level 3, 1800 means the NPU got its level back.

### Measured: capping GFX releases the clamp completely, level 3 -> level 7

Three arms, NPU under the engine, `gpu_metrics` sampled throughout (`build/atb/gfxcap.sh`):

| GFX setting | peak IPUCLK | peak MPIPUCLK | peak package | engine pp2048 | NPU mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| `auto` (control) | 1327 | 1007 | **93.0 W** | 498.63 | 7.442 |
| `manual` + `pp_dpm_sclk=1` (write rejected) | 1290 | 988 | 93.0 W | 492.50 | 7.579 |
| **`low`** (600 MHz) | **1810** | **1267** | **61.0 W** | 147.99 | **5.365** |

DPM level 7 is {npuclk 1267, hclk 1800}; level 3 is {975, 1267}. The `low` arm lands exactly on the
**level-7** row (1267 / 1810) and the NPU's mean returns to its solo value (5.365 against 5.351
solo, best 5.309). **The clamp is released by nothing but package headroom**: 61 W peak -> level 7,
93 W at the cap -> level 3.

The cost is the problem. `low` pins GFX at 600 MHz and the engine falls from 498.63 to 147.99, a
70% loss, so the control is useless as a setting.

### The intermediate cap is not reachable -- and that is the patch worth writing

* `echo 1 > pp_dpm_sclk` -> **EINVAL**. `amdgpu_set_pp_dpm_sclk` exists only as the sysfs wrapper in
  `amdgpu_pm.c:1114`; **there is no smu14 implementation**, so the per-level selection has nothing to
  call.
* `echo "s 1 1600" > pp_od_clk_voltage` -> **EINVAL**, even with `ppfeaturemask=0xffffffff`.
* `smu_v14_0_0_force_clk_levels` handles SOCCLK, FCLK, VCLK, DCLK, VCLK1 and DCLK1 and **explicitly
  omits GFXCLK** (default: -EINVAL).
* But `smu_v14_0_0_set_soft_freq_limited_range` **does** handle SMU_GFXCLK, mapping it to
  `SMU_MSG_SetSoftMinGfxclk` / `SMU_MSG_SetSoftMaxGfxClk`, with the comment *"SoftMin lets PMFW
  throttle gfxclk"*.

So the message path is implemented and simply not wired to any user interface. **A small patch
exposing `SMU_MSG_SetSoftMaxGfxClk` is the one kernel patch with a mechanism** -- not to address the
NPU, which no SMU clock id can name, but to leave the package the headroom that releases it.

### Why that patch probably still loses, and what to do instead

The NPU carries about 29% of prefill (43.8% of the FFN, which is ~67% of prefill). A 38.7% NPU
improvement cuts roughly **7.8%** of total time; a GFX cap costing more than ~8% of GPU throughput
gives that straight back. Only a cap sitting right at the threshold would be free, and the threshold
is unknown -- so the patch would buy the answer, not the win.

**Raising the package limit is strictly better.** It buys the same level-7 grant (the all-130 W arm
already measured NPU mean 5.618 with the engine running), costs the GPU nothing by construction, and
instead adds +9.2% to the engine under concurrent NPU load. Capping GFX pays for the NPU's level with
GPU throughput; raising the limit pays with heat. There is no third option, because the arbiter
inside the SMU is the only thing that decides, and it decides on the package budget.

## Research pass: the clamp has a ~45 s time constant, and the missing watts are DRAM

Four questions were open after the GFX-cap experiment. Three are now answered, and two of the
answers change the conclusion.

### 1. The clamp lifts on its own under sustained load

A single long engine run (`-r 18`) with six sequential NPU chunks, `gpu_metrics` sampled at 2 Hz:

| chunk (sequential) | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NPU best (ms) | 6.904 | 6.895 | 5.498 | 5.478 | 5.399 | **5.370** |
| NPU mean (ms) | 7.600 | 7.589 | 7.229 | 6.491 | 6.157 | **6.016** |

and the granted clock over the same window, package pinned at 93.0 W throughout:

| t (s) | IPUCLK | MPIPUCLK | gfxclk | socclk | fclk | gfxmax | thm_gfx residency |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 9385 | 1277 | 980 | 1852 | 1104 | 1600 | 1769 | 109288 |
| 9405 | 1399 | 1050 | 1862 | 1163 | 1644 | 1838 | 117412 |
| 9415 | 1594 | 1143 | 1728 | 1434 | 1864 | 1749 | 127430 |
| 9425 | **1637** | **1177** | 1653 | 1466 | 1885 | 1564 | 137447 |

**The SMU walks the NPU from level 3 back toward level 7 over ~40-45 s, at constant package power,
while it simultaneously throttles GFX** (1887 -> 1653 MHz, `thm_gfx` residency climbing steadily) and
raises SOCCLK and FCLK (1104 -> 1466, 1600 -> 1885). The best-case command time returns to 5.370 ms,
within 0.4% of solo. This is a rebalancing, not a warm-up: the NPU is not getting faster by itself, it
is being given back part of the budget as the GFX thermally yields.

This is why the earlier warm-up test was refuted -- that one sampled 28 s of an NPU *solo* run, and
this effect needs a mixed load sustained for roughly twice as long. **Every comparison in this
document that ran for under ~45 s was measuring the clamp's initial state, not its steady state.**

### 2. Two mechanisms, now with the granted clock as direct evidence

`gpu_load` synthetic load, NPU concurrent, 1200 commands each:

| GPU load | NPU best | NPU mean | granted IPUCLK/MPIPUCLK | peak package | DRAM read/write |
| --- | ---: | ---: | --- | ---: | ---: |
| compute (register-resident FMA) | 6.446 | 6.877 | **1413 / 1059 (level 3)** | 93.0 W | 42.7 / 3.4 GB/s |
| memory (streaming copy) | **8.493** | **9.711** | **1798 / 1260 (level 7)** | 88.8 W | 52.8 / 53.2 GB/s |

**The memory load leaves the NPU at level 7 and still costs it 81%** (9.711 against 5.351 solo) --
that is pure bandwidth starvation, and it is worse than the power clamp. The compute load clamps the
clock to level 3 and costs 29%. The engine's prefill does both. The earlier "two effects" correction
is therefore right in kind: there is a power clamp *and* a bandwidth fight, and they are separable
now because the granted clock distinguishes them.

### 3. The unattributed ~63 W is DRAM and fabric traffic -- and the NPU drives half of it

`gpu_metrics` detail lines at package 93.0 W:

    apu=93.0  sys=4.2  dgpu=0.0 | core_power_sum=6.5
    dram r/w = 63138 / 12097 MB/s      <-- 75 GB/s total
    ipu  r/w = 37084 /  2471 MB/s      <-- 39 GB/s of it is the NPU's own fetches
    gfx power ~25 W, IPU power ~0.9 W

So the breakdown of the 93 W is roughly: GFX 25 W, cores 4-6 W, IPU 0.9 W, and **~60 W in the memory
subsystem and fabric**, at ~75 GB/s. The LPDDR5X is on-package, so its power is inside `average_apu_power`
and is not attributed to any engine. Note also `average_sys_power` is 2-8 W while `average_apu_power` is
93 W, so the "sys power" field is not the system total and should not be read as one.

**This is the most important correction in the pass.** The IPU's *compute* cost really is ~1 W, but its
*fetch* cost is not: the NPU pulls 39 GB/s, and that traffic is charged to the memory subsystem, which
is inside the same package budget the SMU enforces. So when the SMU clamps the NPU it is not behaving
absurdly -- it is responding to a shared resource that the NPU itself loads heavily. The operator's
"shafted for half a watt" is exactly right about the *attribution* and incomplete about the *physics*:
the watt the NPU is denied is small, but the memory power it causes is what the arbiter is really
rationing.

It also explains why the `low` GFX cap released the clamp: at 600 MHz GFX the package fell to 61 W
because the GPU stopped generating traffic, leaving DRAM headroom.

### 4. Upstream has no NPU control to borrow

The NPU metrics interface is brand new -- `[PATCH V3 1/2] platform/x86/amd/pmf: Introduce new interface
to export NPU metrics`, reviewed January 2026 (Lizhi Hou, Shyam Sundar S K, Mario Limonciello). It is a
**metrics export only**, and its v2 path copies `m_table_v2` fields with no accumulation, while the
M80H/Strix Halo path uses the `_acc` deltas. AMD has never exposed an NPU power or clock *control*:
the entire control surface in `amdxdna_accel.h` is get/set power mode
(`DEFAULT/LOW/MEDIUM/HIGH/TURBO`) plus `set_dft_dpm_level`, and `aie2_pm_init` already walks
`dpm_clk_tbl` to find `max_dpm_level` and requests it at init. Nothing to borrow, nothing that was
left disabled.

### Where this leaves the patch

It lowers its value further. The SMU already performs the GFX-to-IPU rebalance that a soft GFX cap
would force -- it just takes ~45 s and it does it by letting the GFX thermally throttle. A patch
exposing `SMU_MSG_SetSoftMaxGfxClk` would make that happen immediately and controllably, but it would
not produce a state the hardware cannot already reach, and the steady-state numbers above are mostly
what a cap would buy.

The practical consequences that follow are cheaper than a kernel rebuild:

* **Measure the steady state, not the first 30 s.** Any NPU-under-engine number taken under ~45 s is
  the clamped transient. The +8.5% split gain measured earlier is a lower bound.
* **The limit is still the honest lever** (130 W buys the level back immediately rather than after 45 s),
  and the memory traffic is the one the split should attack -- the B repack writes and re-reads the
  NPU's whole weight slice, which is exactly the resource that is actually rationed.

### Correction: the clamp recovery is a 93 W effect, not a time constant

The section above concluded the clamp "lifts on its own over ~45 s". **At 80 W it does not.** Two
protocols, both under a continuous engine load:

| protocol at 80 W | result |
| --- | --- |
| one continuous 14000-command context (~100 s) | deciles flat: `7.391 7.262 7.375 7.411 7.482 7.462 7.520 7.514 7.530 7.290` |
| six 1200-command contexts, then three more after a 40 s idle pause | means `7.167 7.446 7.498 7.503 7.566 7.601`, then `7.496 7.604 7.591` |

Granted IPUCLK sat at 1236-1387 (level 3) for the whole 114 s of the second protocol, and the 40 s pause
-- well past the 5 s runtime-autosuspend delay, so contexts really were recreated after a real resume --
changed nothing. Neither re-creating the context nor waiting recovers it.

So the recovery seen at 93 W (7.600 -> 6.016 across six chunks, IPUCLK 1277 -> 1637) is a function of
the **power limit**, not of elapsed time and not of context lifetime. At 93 W the SMU can afford to
promote the NPU progressively as the GFX settles; at 80 W it cannot, and the clamp is permanent for as
long as the engine is busy.

Both bullets above are therefore wrong for the standing 80 W setting. There is no transient to wait
out, so the split's +8.5% measured at 80 W is already a steady-state figure rather than a lower bound;
and 130 W does not "buy the level back after 45 s", it buys it at all, which 80 W never does.

### The request path, from the code

The level is asked for once and then held. `aie_smu_set_dpm` issues `AIE_SMU_SET_HARD_DPMLEVEL`
followed by `AIE_SMU_SET_SOFT_DPMLEVEL`, and its caller is guarded:

    aie2_xrs_set_dft_dpm_level(): if (ndev->pw_mode != POWER_MODE_DEFAULT || ndev->dpm_level == dpm_level) return 0;

so a solver call at context creation is a **no-op** once `ndev->dpm_level` already holds the target, and
`aie2_solver.c` picks `max_dpm_level` whenever no QoS parameters are present -- which is our case. The
only re-sends are `aie2_pm_init`'s resume branch and `aie2_pm_set_mode`. The SMU therefore receives one
hard-level request at device init (or on resume), grants what it grants, and is never asked again.

That makes **`aie2_pm_set_mode` the one untested lever**: a power-mode transition re-issues the hard
level, so re-requesting `performance` (mode HIGH, `dpm_level = max_dpm_level`) mid-run would ask the
SMU a second time, at a moment when the GFX has settled. Whether a second request is granted better
than the first is unknown, and it is the cheapest remaining question about the clamp.

### Measured: re-requesting the top level changes nothing

That question is now answered. Under a continuous engine load at 80 W, seven NPU chunks with a power
mode change before each -- every change forcing a fresh `SET_HARD_DPMLEVEL`:

| chunk | pmode before it | IPUCLK peak | best ms | mean ms | TFLOPS |
| --- | --- | ---: | ---: | ---: | ---: |
| A baseline | default | 1355 | 6.765 | 7.408 | 25.40 |
| B | performance | 1347 | 6.839 | **7.419** | 25.12 |
| C | balanced | 1267 | 7.584 | 7.698 | 22.65 |
| D | performance | 1334 | 6.850 | **7.444** | 25.08 |
| E | powersaver | 858 | 12.036 | **12.108** | 14.27 |
| F | performance | 1320 | 6.859 | 7.546 | 25.05 |
| G | turbo | 1383 | 6.824 | 7.201 | 25.18 |

**The knob works, and the measurement chain reproduces the document's own earlier numbers.** `balanced`
and `powersaver` return 7.584 ms / 22.65 TF and 12.036 ms / 14.27 TF against the no-load table recorded
much earlier in this document (7.577 / 22.67 and 12.048 / 14.26) -- a 0.1% agreement that validates both
the pmode path and the instrument.

**And re-requesting the top level does nothing.** Three separate `performance` chunks (B, D, F) return
7.419, 7.444 and 7.546 ms against the 7.408 ms baseline -- identical within noise. The IPUCLK peaks tell
the same story: every performance chunk granted 1320-1383 MHz, which is level 3's hclk (1267) plus filter
lag, **not** level 7's 1800. The driver asks for level 7 each time and the SMU answers level 3 each time.

So the clamp is a **continuous arbitration, not a latched decision**, and no request lifts it. The
complete lever set is now closed:

* The level request is already the maximum (`aie2_solver.c` picks `max_dpm_level` with no QoS).
* Re-requesting it via the power-mode path changes nothing (measured above).
* Context lifetime changes nothing, and at 80 W neither does elapsed time (measured).
* Only two things release it: **package headroom** -- a 600 MHz GFX cap produced 1810/1267, level 7, at
  61 W -- or **a higher package limit**, 130 W.

There is no software route to the IPU's share. Whatever the SMU decides is what runs.

## Digging the SMU itself: what is and is not in there

Four passes over the interface and the binary, exhaustively rather than by sampling.

**The control plane contains no IPU field at all.** Scanning every PMFW interface header
(`smu14_driver_if_v14_0.h`, `smu14_driver_if_v14_0_0.h`, `smu_v14_0_0_pmfw.h`, `smu_v14_0_2_ppsmc.h`)
for any identifier containing `ipu` or `npu` returns exactly eight hits, and they are only:

    FEATURE_IPU_DPM_BIT 19, FEATURE_DS_IPUCLK_BIT 58        (feature bits)
    IpuclkFrequency, MpipuclkFrequency, IpuPower, IpuBusy[], IpuReads, IpuWrites   (metrics)

`PPTable_t` -- the entire table the driver hands the SMU -- has **none**, and `PPCLK_e` enumerates
eleven clocks with the IPU absent. There is no NPU power, current, voltage or allocation field anywhere
in what the kernel can say to the SMU. The "put the NPU in the CPU pool" idea is closed for good: the
vocabulary for it does not exist.

**The driver does not switch IPU DPM on -- the SMU does.** `smu_v14_0_0_dpm_features` does contain
`SMU_FEATURE_BIT_INIT(FEATURE_IPU_DPM_BIT)`, but its only consumer is `smu_v14_0_0_is_dpm_running()`,
which *reads* the enabled mask back. And the 14.0.3 path, `smu_v14_0_2_init_allowed_features()`, is one
line: `smu_feature_list_set_all(smu, SMU_FEATURE_LIST_ALLOWED)`. Everything is allowed; nothing is
selectively enabled by the driver.

**The firmware is neither compressed nor encrypted, and its tables are plain.** Shannon entropy over
every 4 KB window of the 333236-byte blob: **no window exceeds 7.5 bits/byte**, 42 of 81 fall below 5.0,
and `0x4000`/`0x5000` are all-zero. The header parses cleanly
(`{0x515b4, 0x2c, 2, 0xe, 0x684f00, 0x4fe00, 0x100, 0x2771d83b, 0x20000, 0x4ff00, 0x16b4}`; and
327424 + 5812 = 333236), and clock ladders sit in the open:

    @0x7534  300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800
    @0x7554  {200 400 500 600 700 750 800 2000}   also at 0x7564, 0x7574   (three domains)
    @0x756a  {600 700 750 800 870 900 1050 1100}
    @0x757a  {100 450 550 770 870 1000}
    @0xc808  a ladder containing 1285 and 1797

That last one is the interesting one: **1285 and 1797 are the clocks we measured the SMU actually
apply** (1285 under engine load, 1797-1810 with headroom), whereas the driver's own `npu4_dpm_clk_tbl`
says 1267 and 1800. The firmware has its own level table and the driver's copy is an approximation of
it. So the IPU level table is in the SMU image, findable, and slightly different from what the kernel
believes.

**But there is no ISA.** No symbols, no strings, no ELF, and no public documentation of the PMFW core.
The tables can be read; the arbitration policy that consumes them cannot be disassembled in practice.
Data yes, control flow no.

**The one patchable bit.** `FEATURE_IPU_DPM_BIT` (19) is the only IPU-specific thing in the SMU's
feature set, and `smu_cmn.c` already has the senders (`SMU_MSG_DisableSmuFeaturesLow/High`). A patch
that clears bit 19 and disables the feature is the single experiment the interface admits. It must
*disable* rather than merely disallow: the allowed list is already all, and the feature is already on.

Two outcomes, and they are both informative. If the SMU's IPU DPM block is what clamps the clock, then
with it off the NPU should keep whatever the driver last asked for -- level 7, 1800 -- under full engine
load, which would be the win this whole hunt has been after. If the feature is load-bearing, the NPU
clock will stick low or the DPM requests will start failing, and it reverts. Nothing in between is
likely, and both are one reboot from reverting.

### Aside found on the way

`amdgpu` exposes the IOMMU PerfOpt work as a module parameter:
`module_param_named(iommu_perfopt, amdgpu_iommu_perfopt, int, 0444)`. So the `iommu=pt` regression
thread has a documented knob in the driver as well as the boot parameter, which is worth knowing if
that ~10% ever needs re-testing.

## Measured: disabling IPU DPM does not release the clamp

The experiment was run and the lever is closed. The patch (kept at `/home/q/ipudpm-experiment.patch`)
made `smu_v14_0_set_allowed_mask()` clear `FEATURE_IPU_DPM_BIT` from the allowed mask and then send
`DisableSmuFeaturesLow` with that bit -- clearing alone is not enough, because the SMU enables IPU DPM
itself and the allowed list is already "all".

NPU under the engine at 80 W, after rebooting into the patched module:

| | stock | patched |
| --- | ---: | ---: |
| NPU mean | 7.408 / 7.599 / 7.666 ms | **7.220 ms** |
| granted IPUCLK peak | 1236-1387 MHz | **1388 MHz** |
| deciles | flat ~7.4 | flat ~7.2 |
| NPU best | 5.31 ms | 5.346 ms |
| package peak | 80.0 W | 80.6 W |

The granted clock is **still level 3**, and the best-case command still runs at full speed while the mean
stays clamped -- behaviourally identical. The 2-4% on the mean and the +8% on the engine are the
fresh-boot effect (the engine also read 538 against 479-498 immediately pre-reboot), not a mechanism
change.

**The IPU DPM feature block is not what arbitrates the NPU clock.** The clamp lives elsewhere in SMU
firmware we cannot read, and the search has now exhausted every software path: the level request is
already maximum, re-requesting does nothing, context lifetime does nothing, time at 80 W does nothing,
no clock id names the IPU, the control plane has no IPU field, and the one IPU-specific feature bit does
nothing. Package headroom and the package limit are the whole lever set.

One honest caveat: the feature mask cannot be read back, so the negative admits two readings -- the
feature is not the clamp, or the disable was ignored. The granted clock is unchanged at level 3 either
way, so the lever is dead regardless; distinguishing them would need a diagnostic build that prints the
message return codes and enabled mask, and would not change the outcome.

### How to build and boot a patched amdgpu module on this box

Worth recording, because four separate things silently defeat the obvious approach.

1. **It is a module rebuild, not a kernel rebuild.** `amdgpu` is `amdgpu.ko.zst`, and the source tree is
   `/home/q/linux-7.3rc` with a git HEAD matching the running kernel. Build with
   `make -j$(nproc) M=drivers/gpu/drm/amd/amdgpu`. **Do not** use `make .../amdgpu.ko`: that takes the
   `single_modules` path, regenerates `Module.symvers` from scratch and fails with ~166 undefined
   `drm`/`ttm` symbols.
2. **Vermagic.** A dirty tree appends `-dirty`, and `CONFIG_MODVERSIONS` is *not* set, so
   `same_magic()` is a plain `strcmp` and the module is refused. The string comes from
   `include/generated/utsrelease.h` -- **not** `include/config/kernel.release`, which the `M=` build
   ignores. `CONFIG_MODULE_FORCE_LOAD=y` exists as a fallback but taints.
3. **The initramfs carries its own copy.** `HOOKS` includes `kms`, so `amdgpu.ko.zst` is inside the image
   and is what loads for early KMS. Replacing only `/lib/modules` silently runs the old code. Regenerate
   with `mkinitcpio -g <path> -k <kver>` (`/etc/mkinitcpio.d/` is empty here, so there are no presets to
   use `-P` with).
4. **The limine entry is hash-verified.** `module_path: ...#<hash>`; refresh with
   `limine-entry-tool --add-kernel linux-perfopt <initramfs> <vmlinuz>`, which updates in place (entry
   count stays 1) and changes exactly one line. The hash is limine's own 512-bit digest rather than
   sha512 of the file, and a stale one still booted here, so it is not enforced -- but regenerating is
   cheap and the diff against a backup is the cleanest way to confirm the edit.

Verify by md5-ing the module, **extracting the initramfs and md5-ing the module inside it**, and
diffing `limine.conf` against a saved copy. An `M=` build also lacks the `intree` modinfo tag, so the
kernel logs `loading out-of-tree module taints kernel`; a full in-tree `make modules` avoids that.

### Correction: the GPU *can* be capped smoothly from userspace

The earlier claim that "no moderate GFX cap is reachable" was wrong, and the reason is one line in
`smu_v14_0_od_edit_dpm_table`:

    /* Only allowed in manual mode */
    if (smu_dpm->dpm_level != AMD_DPM_FORCED_LEVEL_MANUAL)
        return -EINVAL;

My probe never set `manual` first, so the OD write bailed before parsing. The table it edits is
`gfx_actual_soft_max_freq`, committed with `SMU_MSG_SetHardMinGfxClk` + **`SMU_MSG_SetSoftMaxGfxClk`** --
the exact soft cap the driver's own comment describes ("SoftMin lets PMFW throttle gfxclk"). So:

    echo manual > /sys/class/drm/card1/device/power_dpm_force_performance_level
    echo s 1 <mhz> > /sys/class/drm/card1/device/pp_od_clk_voltage     # s=SCLK, 0=min, 1=max
    echo c > /sys/class/drm/card1/device/pp_od_clk_voltage             # commit
    # restore:  echo r > .../pp_od_clk_voltage ; echo auto > .../power_dpm_force_performance_level

Verified: `s 1 1600` holds the GFX at 1594-1596 MHz with `gfxmax=1600`. (`pp_dpm_sclk` remains
unwritable -- smu14 has no `set_pp_dpm_sclk`, and `force_clk_levels` omits GFXCLK -- but OD covers it.)
`pp_od_clk_voltage` also accepts `m`/`f`/`p` (MCLK/FCLK/CCLK) and `r`; TDC is **not** among the
types this ASIC's `od_edit_dpm_table` handles (they fall through to `-ENOSYS`), so a current cap is not
exposed -- the soft-max clock is the practical equivalent.

**The trade curve at 80 W**, NPU under the engine, capped in manual mode:

| GFX soft-max | engine pp2048 | NPU mean | granted IPUCLK | gfxclk | pkg peak |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `auto` | 533.7 +/- 18.4 | 7.247 ms | 1379 (level 3) | 2297 | 95.9 W |
| manual, 2900 | 538.0 +/- 10.9 | 7.122 ms | 1399 (level 3) | 2256 | 93.1 W |
| 2000 | 508.5 +/- 9.0 | 6.506 ms | 1584 | 2000 | 92.9 W |
| 1600 | 425.8 +/- 3.6 | **5.614 ms** | 1776 | 1600 | 91.3 W |
| 1200 | 325.0 +/- 0.6 | **5.421 ms** | **1809 (level 7)** | 1200 | 82.8 W |

`manual` alone is neutral (538.0 against 533.7), which isolates the effect to the cap rather than the
mode. The clamp releases monotonically and is fully gone by 1600 (NPU solo is 5.35 ms). **2000 MHz is the
cheap point: engine -5.4%, NPU +8.6%.**

**Whether that pays depends on the split's work share, and on these numbers it probably does not.** The
NPU carries ~29% of prefill, so at cap 2000 the engine's 5.4% loss on 71% of the work (+3.8% time) very
nearly cancels the NPU's 8.6% gain on 29% (-2.5% time) -- roughly 1% net negative. At 1600 the engine's
21% loss is clearly not covered. The cap only becomes attractive if the balance point moves far enough
toward the NPU to exploit it, which is a *split* measurement, not a baseline one: the right experiment is
the split throughput at several caps, re-finding n_gu at each.

### The complete GPU tunable surface, and which of it works

Enumerated live rather than from the source. **Writable and functional:**

| control | scope | works? |
| --- | --- | --- |
| `power_dpm_force_performance_level` | auto / low / high / manual / `profile_*` | yes |
| `pp_od_clk_voltage` | SCLK min/max (in `manual`) -> `SetHardMinGfxClk` + `SetSoftMaxGfxClk` | **yes, the only lever that releases the NPU** |
| `pp_dpm_socclk` | force SOCCLK level (in `manual`) | yes, no NPU budget effect |
| `pp_dpm_fclk` | force FCLK level (in `manual`) | yes, no NPU budget effect |
| `pp_dpm_dclk`, `_vclk`, `_dclk1`, `_vclk1` | force VCN/DCN levels | yes |
| `pp_force_state`, `thermal_throttling_logging`, `reset_method`, `reset` | misc | yes |
| `resource0/2/4/5` (0600) | **raw MMIO to the GPU BARs -- i.e. the SMU mailbox registers** | the only userspace route to arbitrary SMU messages, no patch needed |
| module params at boot | `ppfeaturemask`, `smu_pptable_id`, `pg_mask`, `cg_mask`, `bapm`, `iommu_perfopt`, ... | boot-time only (0444) |
| debugfs | `amdgpu_smu_debug`, `amdgpu_benchmark`, `amdgpu_compute_sched_mask`, ... | misc |

**Rejected:** `pp_dpm_sclk` (no smu14 `set_pp_dpm_sclk`) and `pp_dpm_mclk`. **Read-only:** all of amdgpu's hwmon -- `power1_input/average`, `temp1_input`, `freq1_input`, `in0/in1` -- with **no fan and no power cap**. **Not exposed anywhere:** TDC/current (`SetThrottlerMask` 0x3A is defined but never called by the driver; the OD table's TDC type falls through to `-ENOSYS`; no `power1_cap` in hwmon).

**FCLK and SOCCLK forcing does nothing for the NPU budget**, contrary to the hope that the ~60 W of unattributed package power was fabric power that could be turned down:

| arm | engine pp2048 | NPU mean | granted IPUCLK | pkg peak |
| --- | ---: | ---: | ---: | ---: |
| `auto` | 530.7 +/- 23.2 | 7.415 ms | 1367 | 96.0 W |
| FCLK 1400 | 535.6 +/- 13.9 | 7.217 ms | 1396 | 93.8 W |
| FCLK 1200 | 522.0 +/- 9.1 | 7.213 ms | 1404 | 94.7 W |
| SOCCLK 883 | 530.6 +/- 21.2 | 7.383 ms | 1357 | 96.0 W |

Peak package power moves by ~2 W and the grant stays at level 3. So the unattributed power is not fabric clock power either, and **the GFX soft-max remains the one control that shifts the balance** -- moving the grant to 1776-1809 and the NPU to 5.4-5.6 ms against these arms' 7.2-7.4.

For completeness, the scope of what a userspace lever can reach is now bounded on both sides: the NPU side has one feature bit that does nothing, and the GPU side has exactly one control that does something, which is `SMU_MSG_SetSoftMaxGfxClk` through the OD table. Everything else either does not exist, is read-only, or is rejected.
