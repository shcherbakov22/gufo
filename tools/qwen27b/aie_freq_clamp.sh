#!/bin/bash
set -u
cd /home/q/gufo
M=/home/q/Downloads/Qwen3.8-27B-IQ4_XS-3.84bpw.gguf
S=/tmp/aiefreq.log; : > $S

./build/release/gufo bench -m $M -p 2048 -n 0 -r 40 > /tmp/aiefreq_bench.log 2>&1 &
BP=$!
doas build/atb/npu_run.sh 45000 dev 1 > /tmp/aiefreq_npu.log 2>&1 &
NP=$!
sleep 10
echo "### A: clamped baseline" >> $S
doas /home/q/smu/sps_sample.sh 8 0.4 A >> $S
echo "### sending frequency requests mid-run" >> $S
doas python3 - >> $S 2>&1 <<'PY'
import sys, time
sys.path.insert(0, '/home/q/gufo/tools/qwen27b')
from smn import smn_read, smn_write
CMD, ARG, RESP = 0x3B10900, 0x3B109F0, 0x3B109F4
def send(op, arg, timeout=0.5):
    t0 = time.time()
    while time.time()-t0 < 0.3 and smn_read(RESP):
        time.sleep(0.001)
    smn_write(RESP, 0); smn_write(ARG, arg); smn_write(CMD, op)
    t0 = time.time()
    while time.time()-t0 < timeout:
        r = smn_read(RESP)
        if r: return r, smn_read(ARG)
        time.sleep(0.001)
    return None, smn_read(ARG)
for op, arg, n in ((0x8, 7, 'HARD_DPMLEVEL 7'), (0x7, 7, 'SOFT_DPMLEVEL 7'),
                   (0x5, 1267, 'MPNPUCLK 1267'), (0x6, 1800, 'HCLK 1800')):
    st, out = send(op, arg)
    print("  %-16s status=%s out=%s" % (n, hex(st) if st else st, hex(out) if out is not None else None))
PY
echo "### B: after frequency request" >> $S
doas /home/q/smu/sps_sample.sh 22 0.4 B >> $S
wait $NP; wait $BP
echo AIEFREQ_DONE >> $S
