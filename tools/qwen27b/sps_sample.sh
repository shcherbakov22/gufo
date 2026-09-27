#!/bin/bash
# Refresh the PMF/SPS metrics table (via the amdxdna NPU sensor ioctl) and print
# one machine-readable SAMPLE line per iteration.
# usage: sps_sample.sh <n> [interval_s] [label]
N=${1:-10}; IV=${2:-0.25}; L=${3:-sps}
B=/home/q/gufo/build/atb/amdxdna_power
S=/sys/kernel/debug/amd_pmf/sps_metrics
for i in $(seq 1 $N); do
  $B 1 10 >/dev/null 2>&1
  printf '%s %s %s\n' "$(date +%s.%N)" "$L" "$(grep '^SAMPLE' $S)"
  sleep $IV
done

