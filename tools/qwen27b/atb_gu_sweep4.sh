#!/bin/bash
# n_gu sweep for the gate/up NPU offload, at the current (130 W) setting.
# The down offload is closed (4/4 losses), so the split is 1-D: how much gate/up
# width should the NPU take?  Arms rotated per round (Latin square) so every arm
# visits every position once.  15 s gap before every run, -r 3.
#
# Extraction: the result row is "| model | size | params | backend | pp2048 | V +/- S |".
# Use the sed that is proven in atb_downoffload_ab.sh.  Do NOT switch this to an
# awk program with backslash-escaped dollars: inside single quotes those reach awk
# literally, awk then treats them as escaped text, and the field never matches --
# which yields EMPTY results rather than an error.
set -u
cd /home/q/gufo
M=/home/q/Downloads/Qwen3.8-27B-IQ4_XS-3.84bpw.gguf
B=/home/q/npu/atb/build
TCTL=/sys/class/hwmon/hwmon6/temp1_input
OUT=/tmp/atb_gu_sweep4.log
: > $OUT
echo "ppt=$(cat /sys/devices/platform/asus-nb-wmi/ppt_pl1_spl)W" >> $OUT

run_gu() {
  local gu="$1" reps="${2:-3}" envs=""
  if [ "$gu" != "0" ]; then
    envs="GUFO_ATB_GU_XCLBIN=$B/npu_gu_$gu.xclbin GUFO_ATB_GU_INSTS=$B/npu_gu_$gu.bin GUFO_ATB_GU_NSLICE=$gu"
  fi
  doas sh -c "ulimit -l unlimited; cd /home/q/gufo && export GUFO_ATB_BATCH=2048 $envs; ./build/release/gufo bench -m $M -p 2048 -n 0 -r $reps" 2>&1 \
    | grep -E "pp2048" | sed 's/.*| *\([0-9.]*\) .*/\1/'
}

ARMS=(8192 10240 12288 14336 15360)
N=${#ARMS[@]}

for round in $(seq 1 $N); do
  for k in $(seq 0 $((N-1))); do
    i=$(( (k + round - 1) % N ))
    gu=${ARMS[$i]}
    sleep 15
    T0=$(( $(cat $TCTL)/1000 ))
    t0=$(date +%s)
    r=$(run_gu "$gu" 3)
    t1=$(date +%s)
    printf 'round=%d pos=%d gu=%-6s startTctl=%dC  %s  (%ds)\n' "$round" "$k" "$gu" "$T0" "${r:-EMPTY}" "$((t1-t0))" >> $OUT
  done
  echo "### round $round done"
done
echo ATB_GU_SWEEP4_DONE >> $OUT
