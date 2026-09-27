#!/bin/bash
# Paired A/B between two n_gu values: alternate, 1 run each, 15 s gap, so each
# comparison is against a run minutes apart instead of a number from another
# session.  Same proven sed extraction.
set -u
cd /home/q/gufo
M=/home/q/Downloads/Qwen3.8-27B-IQ4_XS-3.84bpw.gguf
B=/home/q/npu/atb/build
TCTL=/sys/class/hwmon/hwmon6/temp1_input
OUT=/tmp/lean_ab.log
: > $OUT
echo "ppt=$(cat /sys/devices/platform/asus-nb-wmi/ppt_pl1_spl)W" >> $OUT

one() {
  local gu="$1"
  doas sh -c "ulimit -l unlimited; cd /home/q/gufo && export GUFO_ATB_BATCH=2048 GUFO_ATB_GU_XCLBIN=$B/npu_gu_$gu.xclbin GUFO_ATB_GU_INSTS=$B/npu_gu_$gu.bin GUFO_ATB_GU_NSLICE=$gu; ./build/release/gufo bench -m $M -p 2048 -n 0 -r 3" 2>&1 \
    | grep -E "pp2048" | sed 's/.*| *\([0-9.]*\) .*/\1/'
}

A=${1:-8192}; Bv=${2:-10240}
for rep in 1 2 3; do
  sleep 15; ta=$(( $(cat $TCTL)/1000 )); a=$(one $A)
  sleep 15; tb=$(( $(cat $TCTL)/1000 )); b=$(one $Bv)
  d=""
  if [ -n "$a" ] && [ -n "$b" ]; then d=$(awk -v a="$a" -v b="$b" 'BEGIN{printf "%+.1f%%", 100*(a-b)/b}'); fi
  printf 'rep=%d  %s=%-8s (Tctl %dC)   %s=%-8s (Tctl %dC)   %s vs %s: %s\n' \
    "$rep" "$A" "${a:-EMPTY}" "$ta" "$Bv" "${b:-EMPTY}" "$tb" "$A" "$Bv" "$d" >> $OUT
  echo "rep $rep: $A=$a  $Bv=$b  $d"
done
echo LEAN_AB_DONE >> $OUT
