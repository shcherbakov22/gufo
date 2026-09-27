#!/usr/bin/env python3
"""Probe the AIE SMU queue (MP1 C2PMSG_0/60/61) from userspace over SMN.

amdxdna drives the SMU through this queue (npu4_regs.c smu_regs_off): CMD is
MP1_C2PMSG_0, ARG and OUT share MP1_C2PMSG_60, RESP is MP1_C2PMSG_61 -- the same
C2PMSG space amdgpu and amd_pmf use, with its own message namespace
(POWER_ON/OFF 0x3/0x4, SET_MPNPUCLK_FREQ 0x5, SET_HCLK_FREQ 0x6,
SET_SOFT_DPMLEVEL 0x7, SET_HARD_DPMLEVEL 0x8).

POWER_ON/OFF are NEVER sent: they power-cycle the NPU.

Reads are printed first; the command probe runs only if the queue looks sane.
"""
import sys, time
sys.path.insert(0, "/home/q/gufo/tools/qwen27b")
from smn import smn_read, smn_write

CMD, ARG, RESP = 0x3B10900, 0x3B109F0, 0x3B109F4
FORBIDDEN = {0x3, 0x4}


def snap(tag):
    print("  %-12s cmd=%08x arg=%08x resp=%08x" % (tag, smn_read(CMD), smn_read(ARG), smn_read(RESP)))


def send(op, arg, timeout=0.5):
    """amdgpu-style MP1 queue protocol: drain, clear, arg, cmd, poll."""
    assert op not in FORBIDDEN, "refusing power command 0x%x" % op
    t0 = time.time()
    while time.time() - t0 < 0.3 and smn_read(RESP):
        time.sleep(0.001)
    smn_write(RESP, 0)
    smn_write(ARG, arg)
    smn_write(CMD, op)
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = smn_read(RESP)
        if r:
            return r, smn_read(ARG)
        time.sleep(0.001)
    return None, smn_read(ARG)


print("=== stage 1: read the queue (no writes) ===")
for i in range(3):
    snap("read %d" % i)
    time.sleep(0.1)

print()
print("=== stage 2: command probe (POWER_ON/OFF excluded) ===")
for op, name, arg in [
    (0x99, "control/bogus", 0),
    (0x7, "SET_SOFT_DPMLEVEL", 7),
    (0x8, "SET_HARD_DPMLEVEL", 7),
    (0x5, "SET_MPNPUCLK_FREQ", 1267),
    (0x6, "SET_HCLK_FREQ", 1800),
    (0x5, "SET_MPNPUCLK_FREQ", 1267),
    (0x6, "SET_HCLK_FREQ", 1800),
]:
    st, out = send(op, arg)
    note = {0x1: "OK", 0xFC: "busy", 0xFD: "guard reject", 0xFE: "no handler"}.get(st, "")
    print("  op=0x%02x %-18s arg=%-5d -> status=%s out=%s  %s" % (
        op, name, arg,
        hex(st) if st is not None else "TIMEOUT",
        hex(out) if out is not None else None, note))

print()
print("=== stage 3: queue state after ===")
snap("after")
