#!/usr/bin/env python3
"""Decode and scan an AMD SMU firmware image (smu_14_0_3.bin).

The container is smc_firmware_header_v1_0:
    header      ucode_array_offset_bytes            payload ...
                (0x100 for smu_14_0_3)              (ucode_size_bytes)
followed by a second section (0x16B4 bytes at 0x4FF00 in smu_14_0_3.bin).

The driver loads this itself, over SMN, with **no verification of contents**:
smu_v14_0_load_microcode() writes the payload dwords to MP1_SRAM (0x03c00004)
and then resets/releases the SMU core via smnMP1_PUB_CTRL (SMN 0x3B10D10), and
the only check anywhere is amdgpu_ucode_validate(), which just compares
fw->size to size_bytes.  The header's crc32 is never read (declared
0x2771D83B, actual payload 0x261BF6BD).

The core is Xtensa little-endian with its address space starting at 0, per
public RE work (bc250-collective/amd_smu_reverse_engineering); it strips the
same 0x100-byte header before loading into Ghidra.

Usage:
  smu_fw.py header <file>
  smu_fw.py sections <file>
  smu_fw.py find <file> <value> [--bits 16|32]
  smu_fw.py ptr-runs <file>
"""
import struct
import sys
import zlib

FIELDS = ("size_bytes", "header_size_bytes", "hv_major", "hv_minor", "ip_major",
          "ip_minor", "ucode_version", "ucode_size_bytes",
          "ucode_array_offset_bytes", "crc32", "ucode_start_addr")


def header(path):
    with open(path, "rb") as fh:
        raw = fh.read()
    values = struct.unpack_from("<IIHHHHIIIII", raw, 0)
    print("file size: %d" % len(raw))
    for name, value in zip(FIELDS, values):
        print("  %-24s = 0x%08X (%d)" % (name, value, value))
    off = values[8]
    payload = raw[off:]
    print("  crc32 declared/total/payload = 0x%08X / 0x%08X / 0x%08X" % (
        values[9], zlib.crc32(raw) & 0xFFFFFFFF, zlib.crc32(payload) & 0xFFFFFFFF))
    print("  payload lands at SMN 0x%X + file-offset" % 0x3C00000)
    return raw, values


def sections(path):
    raw, values = header(path)
    off, size = values[8], values[7]
    print("  section 0 (payload): 0x%05X .. 0x%05X  (%d bytes)" % (off, off + size, size))
    rest = len(raw) - (off + size)
    if rest:
        print("  section 1 (extra):   0x%05X .. 0x%05X  (%d bytes)" % (
            off + size, len(raw), rest))


def find(path, value, bits=16):
    with open(path, "rb") as fh:
        raw = fh.read()
    fmt = "<H" if bits == 16 else "<I"
    width = bits // 8
    hits = [o for o in range(0, len(raw) - width, 2)
            if struct.unpack_from(fmt, raw, o)[0] == value]
    print("%d u%d hits for %d: %s" % (len(hits), bits, value, ["0x%X" % h for h in hits[:20]]))


def ptr_runs(path, lo=0x1000, hi=0x4FE00, stride=8, min_entries=12):
    """Find runs of plausible Xtensa code pointers -- candidate dispatch tables."""
    with open(path, "rb") as fh:
        raw = fh.read()[0x100:]

    def plausible(v):
        return lo <= v < hi and (v & 3) == 0

    runs = []
    for start in range(0, len(raw) - 4, 4):
        if not plausible(struct.unpack_from("<I", raw, start)[0]):
            continue
        count = 0
        while start + stride * count + 4 <= len(raw) and \
                plausible(struct.unpack_from("<I", raw, start + stride * count)[0]):
            count += 1
            if count > 400:
                break
        if count >= min_entries:
            runs.append((count, start))
    runs.sort(reverse=True)
    shown = []
    for count, start in runs:
        if any(abs(start - s) < 0x40 for s in shown):
            continue
        shown.append(start)
        first = [hex(struct.unpack_from("<I", raw, start + stride * i)[0])
                 for i in range(min(8, count))]
        print("  payload+0x%05X stride=%d entries=%d first=%s" % (
            start, stride, count, first))


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    cmd, path = argv[1], argv[2]
    if cmd == "header":
        header(path)
    elif cmd == "sections":
        sections(path)
    elif cmd == "find":
        bits = 32 if "--bits" in argv and "32" in argv else 16
        find(path, int(argv[3], 0), bits)
    elif cmd == "ptr-runs":
        ptr_runs(path)
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
