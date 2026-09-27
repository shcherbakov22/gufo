#!/usr/bin/env python3
"""Classify the amdgpu.fw_load_type=0 boot.

Run this after rebooting into linux-perfopt with the edited cmdline.  It decides
whether the driver-side (direct, unverified) SMU firmware load path works, which
is the precondition for patching smu_14_0_3.bin at all:

  * direct path works -> the driver writes MP1_SRAM itself, with no signature
                         check, so a modified image is viable.
  * direct path fails -> the PSP path is mandatory and the firmware route is
                         closed; go back to the mailbox / ppt levers.

Usage:
    doas python3 /home/q/smu/verify-fwloadtype.sh
"""
import os
import subprocess as sp
import sys


def run(cmd):
    return sp.run(cmd, shell=True, capture_output=True, text=True).stdout


def read(path):
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return None


print("=== boot parameters ===")
print("cmdline      :", read("/proc/cmdline"))
param = read("/sys/module/amdgpu/parameters/fw_load_type")
print("fw_load_type :", param)

dmesg = run("dmesg") or run("doas dmesg")
if not dmesg.strip():
    print("!! could not read dmesg (try with doas)")
    sys.exit(1)

print()
print("=== relevant dmesg lines ===")
keys = ("smu driver if version", "smu fw version", "SMU is initialized",
        "load microcode", "psp", "autoload", "Fatal error during GPU init",
        "amdgpu: failed")
for line in dmesg.splitlines():
    if any(k.lower() in line.lower() for k in keys):
        print(" ", line)

failed = "Load microcode failed" in dmesg
ok = "SMU is initialized successfully" in dmesg
cards = run("ls -d /sys/class/drm/card[0-9]* 2>/dev/null").split()

print()
print("=== verdict ===")
if param != "0":
    print("INCONCLUSIVE: fw_load_type is %r, not 0.  The perfopt entry carrying the edited"
          " cmdline was not the one booted (check the limine menu selection)." % param)
elif failed and not ok:
    print("DIRECT PATH FAILS -- 'Load microcode failed' and no 'SMU is initialized successfully'.")
    print("The MP1_SRAM write path cannot bring the SMU up, so the PSP is mandatory and the")
    print("firmware-patch route is closed.  Revert the cmdline (see below) and go back to the")
    print("mailbox / ppt levers.")
elif ok and not failed:
    print("DIRECT PATH WORKS -- the SMU came up with the PSP bypassed, so its firmware can only")
    print("have come from the driver's own MP1_SRAM writes.  That path validates nothing but the")
    print("file size, so a modified smu_14_0_3.bin is viable.  Next step: an inert validation")
    print("patch (change ucode_version, confirm dmesg reflects it), then the real edit.")
else:
    print("INCONCLUSIVE: found neither the failure nor the success marker.")
    print("DRM cards present:", cards or "none")

print()
print("to revert:  doas cp /boot/limine.conf.pre-fwloadtype /boot/limine.conf")
