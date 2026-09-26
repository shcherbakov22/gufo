#!/usr/bin/env python3
"""bf16 MAC-issue roofline probe for the XDNA2 NPU (2 accumulators).

AIE2P has no fp16 mmul.  bf16 has two implementations selected by
AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16: native (two mac_4x8_8x8_bf16) or
emulated via bfp16 (convert both operands, then mac_8x8_8x8T_conf).
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

void peak_bf16(int8 *__restrict in, float *__restrict out, int32_t reps) {
  using MMUL = aie::mmul<8, 8, 8, bfloat16, bfloat16, accauto>;
  using T = bfloat16;
  aie::vector<T, MMUL::size_A> A = aie::load_v<MMUL::size_A>((const T *)in);
  aie::vector<T, MMUL::size_B> B =
      aie::load_v<MMUL::size_B>((const T *)(in + 2 * MMUL::size_A));
  aie::vector<float, MMUL::size_C> z = aie::zeros<float, MMUL::size_C>();
  MMUL c0(z), c1(z);
  for (int32_t r = 0; r < reps; ++r) {
    c0.mac(A, B);
    c1.mac(A, B);
    c0.mac(A, B);
    c1.mac(A, B);
    c0.mac(A, B);
    c1.mac(A, B);
    c0.mac(A, B);
    c1.mac(A, B);
  }
  aie::store_v(out, c0.template to_vector<float>());
  aie::store_v(out + MMUL::size_C, c1.template to_vector<float>());
}

}
"""

IN_BYTES = 256
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
    out_ty = np.ndarray[(OUT_BYTES,), np.dtype[np.float32]]
    of_in = ObjectFifo(in_ty, name="in")
    of_out = ObjectFifo(out_ty, name="out")

    def core_body(of_in, of_out, fn, reps):
        ei = of_in.acquire(1)
        eo = of_out.acquire(1)
        fn(ei, eo, reps)
        of_in.release(1)
        of_out.release(1)

    worker = Worker(core_body, fn_args=[of_in.cons(), of_out.prod(), func, reps], stack_size=8192)

    def sequence(ii, oi, ik, ok):
        ik.fill(ii)
        ok.drain(oi, wait=True)

    rt = Runtime(sequence, [in_ty, out_ty, of_in.prod(), of_out.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1048576)
    ap.add_argument("--iters", type=int, default=4)
    args = ap.parse_args()

    in_ty = np.ndarray[(IN_BYTES,), np.dtype[np.int8]]
    out_ty = np.ndarray[(OUT_BYTES,), np.dtype[np.float32]]
    variants = [
        ("bf16-native", None),
        ("bf16-emul-bfp16", ["-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16=1"]),
    ]

    for label, flags in variants:
        func = ExternalFunction(
            "peak_bf16",
            source_string=_SRC,
            arg_types=[in_ty, out_ty, np.int32],
            compile_flags=flags,
            stack_size_override=4096,
            inline=False,
        )
        inp = iron.randint(-128, 127, (IN_BYTES,), dtype=np.int8, device="npu")
        out = iron.zeros((OUT_BYTES,), dtype=np.float32, device="npu")
        bench = run_iters(
            peak, inp, out, func=func, reps=args.reps, warmup=1, iters=args.iters
        )
        stats = bench.npu if bench.npu is not None else bench.e2e
        scope = "NPU" if bench.npu is not None else "e2e"
        t = stats.avg_us * 1e-6
        macs = float(args.reps) * MACS_PER_ITER * 512.0
        flops = 2.0 * macs
        per_tile = flops / t * 1e-12
        print(f"{label:16s} scope={scope} avg_us={stats.avg_us:.1f} "
              f"per-tile TOPS={per_tile:.4f} 32-tile={32.0 * per_tile:.2f}")


if __name__ == "__main__":
    main()