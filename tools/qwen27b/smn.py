#!/usr/bin/env python3
"""Raw SMN access through ryzen_smu's sysfs attribute.

The module exposes /sys/kernel/ryzen_smu_drv/smn on the classic SMN index/data
pair in PCI config space (0xC4 index, 0xC8 data):
  * write one u32  -> read that SMN address; the result appears on /sys read
  * write two u32  -> write the second word to the address in the first

Gotcha that cost a sweep: sysfs advances the file position after a write, so a
re-read on the same descriptor returns EOF (0 bytes) and every address looks
zero.  Always lseek(fd, 0, 0) before reading, or open a fresh descriptor.  The
first sweep of 0x3B10000-0x3B11000 reported "0 of 1024 populated" this way
while the module was demonstrably reading the registers fine.

Usage:
  smn.py read <hexaddr>
  smn.py write <hexaddr> <hexval>
  smn.py sweep <hexlo> <hexhi> [--step N] [--all]

Needs root: the attribute is 0644 root-owned.
"""
import os
import struct
import sys

SMN = "/sys/kernel/ryzen_smu_drv/smn"


def smn_read(addr, _fd=None):
    """Return the u32 at SMN *addr*."""
    fd = _fd if _fd is not None else os.open(SMN, os.O_RDWR)
    try:
        os.write(fd, struct.pack("<I", addr))
        os.lseek(fd, 0, 0)              # sysfs left f_pos at 4 after the write
        data = os.read(fd, 4)
        if len(data) != 4:
            raise IOError("short SMN read at 0x%X (%d bytes)" % (addr, len(data)))
        return struct.unpack("<I", data)[0]
    finally:
        if _fd is None:
            os.close(fd)


def smn_write(addr, value, _fd=None):
    """Write *value* to SMN *addr*."""
    fd = _fd if _fd is not None else os.open(SMN, os.O_RDWR)
    try:
        os.write(fd, struct.pack("<II", addr, value))
    finally:
        if _fd is None:
            os.close(fd)


def sweep(lo, hi, step=4, show_all=False):
    fd = os.open(SMN, os.O_RDWR)
    try:
        for addr in range(lo, hi, step):
            try:
                value = smn_read(addr, fd)
            except Exception as exc:                 # noqa: BLE001 - report and continue
                print("0x%05X  <error %s>" % (addr, exc))
                continue
            if value or show_all:
                print("0x%05X = 0x%08X  (%d)" % (addr, value, value))
    finally:
        os.close(fd)


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    cmd = argv[1]
    if cmd == "read":
        print("0x%08X" % smn_read(int(argv[2], 16)))
    elif cmd == "write":
        smn_write(int(argv[2], 16), int(argv[3], 16))
    elif cmd == "sweep":
        step = int(argv[4], 0) if len(argv) > 4 else 4
        sweep(int(argv[2], 16), int(argv[3], 16), step, "--all" in argv)
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
