#!/bin/bash
# Lean tuning protocol: ONE run per config, alternating with an adjacent baseline
# run, so every candidate is compared against a baseline measured minutes apart
# rather than against a number from another session (which drifts 14%).
# 15 s gap before every run.  -r 3.  Extraction via the proven sed.
set -u
cd /home/q/gufo
M=/home/q/Downloads/Qwen3.8-27B-IQ4_XS-3.84bpw.gguf
B=/home/q/npu/atb/build
TCTL=/sys/class/hwmon/hwmon6/temp1_input
OUT=/tmp/lean_gu.log
: > $OUT
echo "ppt=$(cat /sys/devices/platform/asus-nb-wmi/ppt_pl1_spl)W" >> $OUT

one() {   # gu  (0 = splitless)
  local gu="$1" envs=""
  if [ "$gu" != "0" ]; then
    envs="GUFO_ATB_GU_XCLBIN=$B/npu_gu_$gu.xclbin GUFO_ATB_GU_INSTS=$B/npu_gu_$gu.bin GUFO_ATB_GU_NSLICE=$gu"
  fi
  doas sh -c "ulimit -l unlimited; cd /home/q/gufo && export GUFO_ATB_BATCH=2048 $envs; ./build/release/gufo bench -m $M -p 2048 -n 0 -r 3" 2>&1 \
    | grep -E "pp2048" | sed 's/.*| *\([0-9.]*\) .*/\1/'
}

pair() {  # candidate
  local gu="$1"
  sleep 15; local tb=$(( $(cat $TCTL)/1000 )); local b=$(one 0)
  sleep 15; local tc=$(( $(cat $TCTL)/1000 )); local c=$(one $gu)
  local pct=""
  if [ -n "$b" ] && [ -n "$c" ]; then pct=$(awk -v b="$b" -v c="$c" 'BEGIN{printf "%+.1f%%", 100*(c-b)/b}'); fi
  printf 'gu=%-6s base=%-8s (Tctl %dC)  cand=%-8s (Tctl %dC)  %s\n' "$gu" "${b:-EMPTY}" "$tb" "${c:-EMPTY}" "$tc" "$pct" >> $OUT
  echo "  gu=$gu -> ${c:-EMPTY} vs base ${b:-EMPTY} $pct"
}

for gu in 8192 10240 12288; do pair $gu; done
echo LEAN_GU_DONE >> $OUT
