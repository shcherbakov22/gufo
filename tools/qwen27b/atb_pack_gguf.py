#!/usr/bin/env python3
"""Repack every FFN weight tensor in a GGUF shard into the ATB bfp16 B operand.

Drives tools/qwen27b/atb_pack over each tensor, in parallel, reading the raw
quantisation blocks straight out of a memory map. This is the one-off cost the
NPU FFN path pays at load, and the size it costs in memory.

    tools/qwen27b/atb_pack_gguf.py <model.gguf> [--out DIR] [--jobs N] [--limit N]

Without --out the packed bytes are discarded, which measures time and size
without writing ~18 GiB to disk.
"""

import argparse
import concurrent.futures
import mmap
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "gufo"))
import gguf  # noqa: E402

FFN = ("ffn_gate", "ffn_up", "ffn_down")
PACKER = os.environ.get("ATB_PACK", "build/atb_pack")
BLOCK_BYTES = {11: 110, 12: 144, 16: 66, 17: 74, 18: 98, 21: 110, 23: 136}


def load(path):
    with open(path, "rb") as handle:
        data = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        state = gguf._parse_gguf_data(data)
    work = []
    for t in state["tensors"]:
        parts = t["name"].split(".")
        if len(parts) >= 3 and parts[0] == "blk" and parts[2] in FFN:
            work.append(
                {
                    "name": t["name"],
                    "type": t["type"],
                    "rows": int(t["shape"][0]),
                    "cols": int(t["shape"][1]),
                    "offset": state["data_offset"] + t["offset"],
                }
            )
    return data, work


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    buf, work = load(args.model)
    if args.limit:
        work = work[: args.limit]
    if not work:
        print("no FFN tensors found", file=sys.stderr)
        return 1
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
    print(
        "packing %d FFN tensors with %d jobs (out=%s)"
        % (len(work), args.jobs, args.out or "discard")
    )

    def one(spec):
        rows, cols = spec["rows"], spec["cols"]
        block_bytes = BLOCK_BYTES[spec["type"]]
        nbytes = rows * (cols // 256) * block_bytes
        blob = bytes(memoryview(buf)[spec["offset"] : spec["offset"] + nbytes])
        keep = args.out is not None
        dest = open(args.out / (spec["name"] + ".bfp16"), "wb") if keep else subprocess.DEVNULL
        proc = subprocess.run(
            [PACKER, "-", str(spec["type"]), str(rows), str(cols)],
            input=blob,
            stdout=dest,
            stderr=subprocess.DEVNULL,
        )
        if keep:
            dest.close()
        if proc.returncode != 0:
            raise RuntimeError("pack failed for %s" % spec["name"])
        return rows * cols, rows * cols * 9 // 8

    total_weights = total_bytes = 0
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for weights, packed in pool.map(one, work):
            total_weights += weights
            total_bytes += packed
    elapsed = time.monotonic() - started

    print(
        "packed %.3f G weights -> %.2f GiB at 9 bits/weight in %.1f s wall (%.2f G weights/s)"
        % (
            total_weights / 1e9,
            total_bytes / 2**30,
            elapsed,
            total_weights / 1e9 / elapsed,
        )
    )
    print(
        "FFN as shipped 7.05 GiB; packed %.2f GiB; net change %+.2f GiB"
        % (total_bytes / 2**30, (total_bytes / 2**30) - 7.05)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
