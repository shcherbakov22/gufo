#!/usr/bin/env python3
"""Dump one FFN weight tensor per quantization type from a GGUF shard.

The ATB repack tools take raw quantisation blocks, and this is how a real
tensor's blocks are obtained. For each type present in the shard's FFN it picks
a tensor (preferring ffn_gate) and writes the raw bytes plus a manifest of
"type W_rows W_cols path" lines, one per tensor.

    tools/qwen27b/atb_extract.py <model.gguf> --out DIR
"""

import argparse
import mmap
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "gufo"))
import gguf  # noqa: E402

FFN = ("ffn_gate", "ffn_up", "ffn_down")
# type -> bytes per 256 weights, for every type the ATB repack understands.
BLOCK_BYTES = {
    11: 110,
    12: 144,
    13: 176,
    14: 210,
    16: 66,
    17: 74,
    18: 98,
    21: 110,
    22: 82,
    23: 136,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    with open(args.model, "rb") as handle:
        data = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        state = gguf._parse_gguf_data(data)
    chosen = {}
    for t in state["tensors"]:
        parts = t["name"].split(".")
        if len(parts) < 3 or parts[0] != "blk" or parts[2] not in FFN:
            continue
        if t["type"] not in BLOCK_BYTES:
            continue
        best = chosen.get(t["type"])
        if best is None or (parts[2] == "ffn_gate" and best[0] != "ffn_gate"):
            chosen[t["type"]] = (parts[2], t["name"], t["offset"], [int(x) for x in t["shape"]])

    lines = []
    for type_id in sorted(chosen):
        which, name, offset, (rows, cols) = chosen[type_id]
        raw = cols // 256 * BLOCK_BYTES[type_id]
        blob = data[
            state["data_offset"] + offset : state["data_offset"] + offset + rows * raw
        ]
        path = args.out / ("type%d.raw" % type_id)
        Path(path).write_bytes(blob)
        lines.append("%d %d %d %s" % (type_id, rows, cols, path))
        print("%-8s %-26s %s %dx%d -> %s"
              % (gguf.GGML_TYPE_NAME.get(type_id, type_id), name, which, rows, cols, path))
    (args.out / "manifest.txt").write_text("\n".join(lines) + "\n")
    print("manifest: %s (%d tensors)" % (args.out / "manifest.txt", len(lines)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
