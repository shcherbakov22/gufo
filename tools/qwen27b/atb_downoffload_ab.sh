#!/bin/bash
# Interleaved A/B for the FFN split: does the DOWN offload help on top of gate/up?
#
# The down offload was rejected earlier on a comparison since found to be
# thermally confounded, so the question is open again.  Thermal drift on this box
# moves pp2048 by ~20% between sequences, so only differences measured inside one
# session mean anything.  Arms are therefore rotated per round (a Latin-square
# order) so every arm visits every position once -- that cancels both drift and
# ordering bias.
set -u
cd /home/q/gufo
M=/home/q/Downloads/Qwen3.8-27B-IQ4_XS-3.84bpw.gguf
B=/home/q/npu/atb/build
OUT=/tmp/atb_dn_ab.log
: > $OUT

run_arm() {   # gu dn reps
  local gu="$1" dn="$2" reps="${3:-3}" envs=""
  [ "$gu" != "0" ] && envs="$envs GUFO_ATB_GU_XCLBIN=$B/npu_gu_$gu.xclbin GUFO_ATB_GU_INSTS=$B/npu_gu_$gu.bin GUFO_ATB_GU_NSLICE=$gu"
  [ "$dn" != "0" ] && envs="$envs GUFO_ATB_DN_XCLBIN=$B/npu_dn_$dn.xclbin GUFO_ATB_DN_INSTS=$B/npu_dn_$dn.bin GUFO_ATB_DN_NSLICE=$dn"
  doas sh -c "ulimit -l unlimited; cd /home/q/gufo && export GUFO_ATB_BATCH=2048 $envs; ./build/release/gufo bench -m $M -p 2048 -n 0 -r $reps" 2>&1 \
    | grep -E "pp2048" | sed 's/.*| *\([0-9.]*\) .*/\1/'
}

ARMS=("base:0:0" "gu10240:10240:0" "gu+dn2048:10240:2048" "gu+dn3072:10240:3072")
N=${#ARMS[@]}

for round in 1 2 3 4; do
  for k in $(seq 0 $((N-1))); do
    i=$(( (k + round - 1) % N ))
    IFS=: read -r label gu dn <<< "${ARMS[$i]}"
    t0=$(date +%s)
    r=$(run_arm "$gu" "$dn" 3)
    t1=$(date +%s)
    printf 'round=%d pos=%d arm=%-9s gu=%-6s dn=%-5s %s  (%ds)\n' "$round" "$k" "$label" "$gu" "$dn" "$r" "$((t1-t0))" >> $OUT
  done
  echo "### round $round done"
done
echo ATB_DN_AB_DONE >> $OUT
