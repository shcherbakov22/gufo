set -u
cd /home/q/gufo
P=build/atb/amdxdna_power
echo "=== idle baseline ==="
$P 3 1000
echo
echo "=== NPU + GPU-compute contention ==="
doas /home/q/gufo/build/atb/npu_run.sh 2000 dev 1 > /tmp/npu_meas.log 2>&1 &
NP=$!
sleep 1
/home/q/gufo/build/atb/gpu_load compute 300 > /dev/null 2>&1 &
GL=$!
# sample the NPU telemetry while both are running
$P 25 1000 > /tmp/pwr.log 2>&1
# keep sampling until the NPU finishes
wait $NP
kill $GL 2>/dev/null
echo "--- NPU runner result ---"
grep -E "npu:|NPU_RUN" /tmp/npu_meas.log | head -4
echo
echo "--- NPU telemetry during contention (POWER mW, columns) ---"
awk '{print}' /tmp/pwr.log | head -30
echo
echo "--- POWER stats while busy (nonzero only) ---"
python3 - <<'EOF'
vals=[]
for ln in open("/tmp/pwr.log"):
    for tok in ln.split():
        if tok.startswith("POWER="):
            v=int(tok.split("=")[1].rstrip("mW")); vals.append(v)
busy=[v for v in vals if v>0]
print("samples=%d nonzero=%d mean_busy=%.1f max=%d" % (len(vals), len(busy),
      (sum(busy)/len(busy)) if busy else 0, max(vals) if vals else 0))
EOF
