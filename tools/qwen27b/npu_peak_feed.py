#!/usr/bin/env python3
"""Isolate the NPU GEMM dataflow bottleneck: MAC rate vs L1 loads per MAC.

The whole-array GEMM captures ~15% of the MAC-array peak for every datatype,
which points at the operand feed rather than the MAC.  This probe measures the
int8 mac_8x8_8x8 rate with 0, 1 and 2 vector L1 loads per MAC, all with the
same register-resident accumulator set, so it separates load-issue pressure
from kernel scheduling and from DMA.
"""
from __future__ import annotations

import argparse

import aie.iron as iron
import numpy as np
from aie.iron import (
    CompileTime,
    ExternalFunction,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    Worker,
    jit,
)
from aie.utils.benchmark import run_iters

_SRC = r"""
#include <aie_api/aie.hpp>

constexpr int N = 128;

extern "C" {

// 0 loads per MAC: the pure issue-rate reference.
void peak_reg(int8 *__restrict in, int8 *__restrict out, int32_t reps) {
  v64int8 a = *((v64int8 *)in);
  v64int8 b0 = *((v64int8 *)(in + 64));
  v64int8 b1 = *((v64int8 *)(in + 128));
  v64acc32 c0 = {}, c1 = {};
  for (int32_t r = 0; r < reps; ++r) {
    c0 = mac_8x8_8x8(a, b0, c0);
    c1 = mac_8x8_8x8(a, b1, c1);
    c0 = mac_8x8_8x8(a, b0, c0);
    c1 = mac_8x8_8x8(a, b1, c1);
    c0 = mac_8x8_8x8(a, b0, c0);
    c1 = mac_8x8_8x8(a, b1, c1);
    c0 = mac_8x8_8x8(a, b0, c0);
    c1 = mac_8x8_8x8(a, b1, c1);
  }
  v64int8 *po = (v64int8 *)out;
  po[0] = ssrs(c0, 0);
  po[1] = ssrs(c1, 0);
}

// 1 load per MAC: B stays register-resident, A is streamed from L1.
void peak_ld1(int8 *__restrict in, int8 *__restrict out, int32_t reps) {
  v64int8 *pa = (v64int8 *)in;
  v64int8 b = *((v64int8 *)(in + 64 * N));
  v64acc32 c0 = {}, c1 = {};
  for (int32_t r = 0; r < reps; ++r) {
    for (int i = 0; i < N; ++i) {
      v64int8 a = pa[i];
      c0 = mac_8x8_8x8(a, b, c0);
      c1 = mac_8x8_8x8(a, b, c1);
    }
  }
  v64int8 *po = (v64int8 *)out;
  po[0] = ssrs(c0, 0);
  po[1] = ssrs(c1, 0);
}

// 2 loads per MAC: both operands streamed from L1.
void peak_ld2(int8 *__restrict in, int8 *__restrict out, int32_t reps) {
  v64int8 *pa = (v64int8 *)in;
  v64int8 *pb = (v64int8 *)(in + 64 * N);
  v64acc32 c0 = {}, c1 = {};
  for (int32_t r = 0; r < reps; ++r) {
    for (int i = 0; i < N; ++i) {
      v64int8 a = pa[i];
      v64int8 b = pb[i];
      c0 = mac_8x8_8x8(a, b, c0);
      c1 = mac_8x8_8x8(a, b, c1);
    }
  }
  v64int8 *po = (v64int8 *)out;
  po[0] = ssrs(c0, 0);
  po[1] = ssrs(c1, 0);
}

}
"""

IN_BYTES = 2 * 128 * 64
OUT_BYTES = 128
N = 128


@jit
def peak(
    inp: In,
    out: Out,
    *,
    func: CompileTime[ExternalFunction],
    reps: CompileTime[int],
):
    in_ty = np.ndarray[(IN_BYTES,), np.dtype[np.int8]]
    out_ty = np.ndarray[(OUT_BYTES,), np.dtype[np.int8]]
    of_in = ObjectFifo(in_ty, name="in")
    of_out = ObjectFifo(out_ty, name="out")

    def core_body(of_in, of_out, fn, reps):
        ei = of_in.acquire(1)
        eo = of_out.acquire(1)
        fn(ei, eo, reps)
        of_in.release(1)
        of_out.release(1)

    worker = Worker(
        core_body,
        fn_args=[of_in.cons(), of_out.prod(), func, reps],
        stack_size=8192,
    )

    def sequence(ii, oi, ik, ok):
        ik.fill(ii)
        ok.drain(oi, wait=True)

    rt = Runtime(sequence, [in_ty, out_ty, of_in.prod(), of_out.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


# macs issued per outer iteration, per variant
SPEC = {
    "reg": ("peak_reg", 8),
    "ld1": ("peak_ld1", 2 * N),
    "ld2": ("peak_ld2", 2 * N),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=8192)
    ap.add_argument("--iters", type=int, default=4)
    args = ap.parse_args()

    in_ty = np.ndarray[(IN_BYTES,), np.dtype[np.int8]]
    out_ty = np.ndarray[(OUT_BYTES,), np.dtype[np.int8]]

    for label, (sym, macs_per_iter) in SPEC.items():
        func = ExternalFunction(
            sym,
            source_string=_SRC,
            arg_types=[in_ty, out_ty, np.int32],
            inline=True,
        )
        inp = iron.randint(-128, 127, (IN_BYTES,), dtype=np.int8, device="npu")
        out = iron.zeros((OUT_BYTES,), dtype=np.int8, device="npu")
        bench = run_iters(
            peak, inp, out, func=func, reps=args.reps, warmup=1, iters=args.iters
        )
        stats = bench.npu if bench.npu is not None else bench.e2e
        scope = "NPU" if bench.npu is not None else "e2e"
        t = stats.avg_us * 1e-6
        macs = float(args.reps) * macs_per_iter * 512.0
        per_tile = 2.0 * macs / t * 1e-12
        print(f"{label:4s} scope={scope} avg_us={stats.avg_us:.1f} "
              f"per-tile TOPS={per_tile:.4f} 32-tile={32.0 * per_tile:.2f}")


if __name__ == "__main__":
    main()