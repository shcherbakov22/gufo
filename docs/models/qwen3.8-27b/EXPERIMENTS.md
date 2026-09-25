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
