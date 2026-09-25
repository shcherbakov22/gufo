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


