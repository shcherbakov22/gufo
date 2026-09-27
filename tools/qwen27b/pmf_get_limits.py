#!/usr/bin/env python3
"""Read the PMF/SPS power + skin-temperature limits straight from the SMU.

The M20H driver only ever *writes* these (SET_PMF_PPT etc.); the GET_* readbacks
exist in the firmware table but the kernel never issues them on PMF_IF_V2.  We
replay amd_pmf_send_cmd()'s handshake against the PMF queue triple in raw SMN:

    poll resp != 0        (consume the previous response)
    resp <- 0
    arg0 <- 0
    msg  <- op            (this write is the trigger)
    poll resp != 0        (1 = OK, 0xFE = no handler, 0xFD = guard reject)
    sleep, then read arg0 (GET payload lands here late)

Read-only: no SET_* op is ever issued.
"""
import sys, time
sys.path.insert(0, "/home/q/gufo/tools/qwen27b")
from smu_mailbox import queue, PMF_QUEUE

GETS = [
    (0x0B, "GET_SPL"),
    (0x0D, "GET_SPPT"),
    (0x0F, "GET_FPPT"),
    (0x1E, "GET_SPPT_APU_ONLY"),
    (0x1F, "GET_STT_MIN_LIMIT"),
    (0x20, "GET_STT_LIMIT_APU"),
    (0x21, "GET_STT_LIMIT_HS2"),
]
SANITY = [(0x02, "GetSmuVersion"), (0x01, "TestMessage")]


def get(op, settle=0.05):
    c, a, r = PMF_QUEUE
    status, arg = queue(op, c, a, r, 0)
    if status != 1:
        return status, arg, None
    time.sleep(settle)
    from smn import smn_read
    return status, arg, smn_read(a)


def main():
    print("=== sanity (does the PMF queue answer at all?) ===")
    for op, name in SANITY:
        st, arg, late = get(op)
        print("  %-18s op=0x%02x status=%s arg=%s late=%s" % (
            name, op, hex(st) if st else st,
            hex(arg) if arg is not None else None,
            hex(late) if late is not None else None))
    print("\n=== power / STT limits ===")
    for op, name in GETS:
        st, arg, late = get(op)
        note = ""
        if st == 0xFE:
            note = "no handler"
        elif st == 0xFD:
            note = "guard reject"
        elif st == 1 and late is not None:
            if name.startswith("GET_STT"):
                note = "%d cC  (raw 0x%08x)" % (late >> 8, late)
            else:
                note = "%d W  (raw 0x%08x)" % (late, late)
        print("  %-18s op=0x%02x status=%s arg=%s late=%s  %s" % (
            name, op, hex(st) if st else st,
            hex(arg) if arg is not None else None,
            hex(late) if late is not None else None, note))


if __name__ == "__main__":
    main()
