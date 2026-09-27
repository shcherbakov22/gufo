#!/usr/bin/env python3
"""Userspace SMU mailbox clients.

Two independent front ends onto the same PMFW, both reachable as root with no
kernel patch and no reboot:

MP1 mailbox (ryzen_smu sysfs)
    smu_args (24 bytes = six u32) then mp1_smu_cmd (one byte = op).  The
    response is in mp1_smu_cmd (1 = SMU_Return_OK) and the returned arguments
    in smu_args.  Verified against SMU14 ids: 0x02 GetSmuVersion -> 0x0A640200
    (10.100.2.0), 0x03 GetDriverIfVersion -> 0x19, 0x01 TestMessage -> 1.

Queue mailbox (raw SMN)
    The firmware has queue_descriptor_table_offs_0[8]: eight queues, each a
    {CMD, ARG, RSP} register triple, which is why several C2PMSG sets exist
    (NPU C2PMSG_0/60/61 at 0x3B10900/9F0/9F4, ryzen_smu C2PMSG_10/30/38 at
    0x3B10928/978/998, PMF at 0x3B10A18/A58/A78).  queue_dispatch answers
    0xFE "no handler", 0xFD "guard reject", 0xFC "busy" -- so a 0xFE reply is a
    real answer, not a failure.  That is exactly what the PMF/SPS message set
    returns on this Strix Halo, i.e. the platform does not implement it.

Usage:
  smu_mailbox.py mp1 <op-hex> [args...]
  smu_mailbox.py queue <cmd> <arg> <rsp> <op-hex> [args...]
  smu_mailbox.py pmf-get <op-hex>          # uses the PMF queue triple
"""
import os
import struct
import sys
import time

from smn import smn_read, smn_write

BASE = "/sys/kernel/ryzen_smu_drv/"
PMF_QUEUE = (0x3B10A18, 0x3B10A58, 0x3B10A78)      # {CMD, ARG, RSP}, v1 layout


def _write(path, data):
    fd = os.open(BASE + path, os.O_WRONLY)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)


def _read(path, size):
    fd = os.open(BASE + path, os.O_RDONLY)
    try:
        out = b""
        while len(out) < size:
            chunk = os.read(fd, size - len(out))
            if not chunk:
                break
            out += chunk
        return out
    finally:
        os.close(fd)


def mp1(op, *args):
    """Send *op* with up to six u32 args; return (status, six returned args)."""
    packed = list(args) + [0] * (6 - len(args))
    _write("smu_args", struct.pack("<6I", *packed))
    _write("mp1_smu_cmd", struct.pack("<I", op))
    status = struct.unpack("<I", _read("mp1_smu_cmd", 4))[0]
    out = struct.unpack("<6I", _read("smu_args", 24))
    return status, out


def queue(op, cmd, arg, rsp, value=0, timeout=0.5):
    """Drive one firmware queue and return (status, arg) -- 1 = OK, 0xFE = no handler."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if smn_read(rsp):
            break
        time.sleep(0.001)
    smn_write(rsp, 0)
    smn_write(arg, value)
    smn_write(cmd, op)
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = smn_read(rsp)
        if status:
            return status, smn_read(arg)
        time.sleep(0.001)
    return None, None


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    cmd = argv[1]
    if cmd == "mp1":
        status, out = mp1(int(argv[2], 16), *[int(a, 16) for a in argv[3:]])
        print("status=%d args=%s" % (status, [hex(v) for v in out]))
    elif cmd == "queue":
        c, a, r = (int(argv[2], 16), int(argv[3], 16), int(argv[4], 16))
        status, out = queue(int(argv[5], 16), c, a, r,
                            int(argv[6], 16) if len(argv) > 6 else 0)
        print("status=%s arg=%s" % (status, None if out is None else hex(out)))
    elif cmd == "pmf-get":
        c, a, r = PMF_QUEUE
        status, out = queue(int(argv[2], 16), c, a, r)
        print("PMF  status=%s (1=OK, 0xFE=no handler) arg=%s" % (
            None if status is None else hex(status),
            None if out is None else hex(out)))
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
