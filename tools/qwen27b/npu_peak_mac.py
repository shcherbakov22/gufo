#!/usr/bin/env python3
"""Pure MAC-issue roofline probe for the XDNA2 NPU (mlir-aie 1.4.3 IRON).

Operands are loaded into registers before the loop and four independent
accumulator chains are kept live, so the loop measures the mac_8x8_8x8 issue
rate with no L1 traffic and no DMA.  Input is one 192-byte FIFO element
because a compute tile exposes only 2 input / 2 output DMA channels and has
64 KB of L1.
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

extern "C" {
void peak_mac(int8 *__restrict in, int8 *__restrict out, int32_t reps) {
  v64int8 a = *((v64int8 *)in);
  v64int8 b0 = *((v64int8 *)(in + 64));
  v64int8 b1 = *((v64int8 *)(in + 128));
  v64acc32 acc0 = {};
  v64acc32 acc1 = {};
  v64acc32 acc2 = {};
  v64acc32 acc3 = {};
  for (int32_t r = 0; r < reps; ++r) {
    acc0 = mac_8x8_8x8(a, b0, acc0);
    acc1 = mac_8x8_8x8(a, b1, acc1);
    acc2 = mac_8x8_8x8(a, b0, acc2);
    acc3 = mac_8x8_8x8(a, b1, acc3);
    acc0 = mac_8x8_8x8(a, b0, acc0);
    acc1 = mac_8x8_8x8(a, b1, acc1);
    acc2 = mac_8x8_8x8(a, b0, acc2);
    acc3 = mac_8x8_8x8(a, b1, acc3);
  }
  v64int8 *po = (v64int8 *)out;
  po[0] = ssrs(acc0, 0);
  po[1] = ssrs(acc2, 0);
  po[2] = ssrs(acc1, 0);
  po[3] = ssrs(acc3, 0);
}
}
"""

IN_BYTES = 192
OUT_BYTES = 256
MACS_PER_ITER = 8


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
    )

    def sequence(ii, oi, ik, ok):
        ik.fill(ii)
        ok.drain(oi, wait=True)

    rt = Runtime(sequence, [in_ty, out_ty, of_in.prod(), of_out.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=262144)
    ap.add_argument("--iters", type=int, default=5)
    args = ap.parse_args()

    in_ty = np.ndarray[(IN_BYTES,), np.dtype[np.int8]]
    out_ty = np.ndarray[(OUT_BYTES,), np.dtype[np.int8]]
    func = ExternalFunction(
        "peak_mac",
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
    macs = float(args.reps) * MACS_PER_ITER * 512.0
    flops = 2.0 * macs
    per_tile = flops / t * 1e-12
    print(f"scope={scope} avg_us={stats.avg_us:.1f} min_us={stats.min_us:.1f}")
    print(f"macs per kernel call   = {macs:.4e}")
    print(f"per-tile TOPS          = {per_tile:.4f}")
    print(f"32-tile array TOPS     = {32.0 * per_tile:.2f}")


if __name__ == "__main__":
    main()