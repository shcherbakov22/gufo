#!/bin/bash
# Rotated n_gu sweep for the gate/up NPU offload.
# The down offload is closed (4/4 losses), so the split is 1-D: how much gate/up
# width should the NPU take?  The recorded sweep stopped at 8192; the 10240 arm in
# the down-offload A/B measured +17% over base, so the curve may still be rising.
# Latin-square order again -- thermal drift here moves pp2048 by ~35%.
set -u
cd /home/q/gufo
M=/home/q/Downloads/Qwen3.8-27B-IQ4_XS-3.84bpw.gguf
B=/home/q/npu/atb/build
OUT=/tmp/atb_gu_sweep.log
: > $OUT

run_gu() {
  local gu="$1" reps="${2:-3}"
  doas sh -c "ulimit -l unlimited; cd /home/q/gufo && export GUFO_ATB_BATCH=2048 GUFO_ATB_GU_XCLBIN=$B/npu_gu_$gu.xclbin GUFO_ATB_GU_INSTS=$B/npu_gu_$gu.bin GUFO_ATB_GU_NSLICE=$gu; ./build/release/gufo bench -m $M -p 2048 -n 0 -r $reps" 2>&1 \
    | grep -E "pp2048" | sed 's/.*| *\([0-9.]*\) .*/\1/'
}

ARMS=(8192 10240 12288 14336 15360)
N=${#ARMS[@]}

for round in $(seq 1 $N); do
  for k in $(seq 0 $((N-1))); do
    i=$(( (k + round - 1) % N ))
    gu=${ARMS[$i]}
    t0=$(date +%s)
    r=$(run_gu "$gu" 3)
    t1=$(date +%s)
    printf 'round=%d pos=%d gu=%-6s %s  (%ds)\n' "$round" "$k" "$gu" "$r" "$((t1-t0))" >> $OUT
  done
  echo "### round $round done"
done
echo ATB_GU_SWEEP_DONE >> $OUT
