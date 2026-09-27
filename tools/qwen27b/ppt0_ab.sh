set -u
cd /home/q/gufo
GM=build/atb/gpu_metrics.py
setppt() {
  doas python3 -c "
import sys; sys.path.insert(0,'/home/q/gufo/tools/qwen27b')
from smu_mailbox import mp1
st,out = mp1(0x32, $1)
print('  SetPptLimit(%s) -> status=%d echo=0x%X' % ($1, st, out[0]))
"
}
run_case() {
  local ppt=$1 tag=$2
  echo "================ PPT0=$ppt  ($tag) ================"
  setppt $ppt
  sleep 3
  ( for i in $(seq 1 45); do python3 $GM 1 1 -v 2>/dev/null; sleep 1; done ) > /tmp/gm_$tag.txt &
  S=$!
  doas /home/q/gufo/build/atb/npu_run.sh 2000 dev 1 > /tmp/npu_$tag.log 2>&1 &
  NP=$!
  sleep 2
  /home/q/gufo/build/atb/gpu_load compute 300 > /dev/null 2>&1 &
  GL=$!
  wait $NP
  kill $GL 2>/dev/null
  sleep 1
  kill $S 2>/dev/null
  grep -E "npu:|NPU_RUN|median|mean" /tmp/npu_$tag.log | head -6
  sleep 8
}
run_case 80  a80
run_case 200 b200
echo "================ analysis ================"
python3 - <<'EOF'
import re
for tag in ("a80","b200"):
    ipu=[]; pkg=[]; gfx=[]
    try:
        for ln in open("/tmp/gm_%s.txt" % tag):
            m=re.search(r"IPU=(\S+)", ln); p=re.search(r"pkg=\s*([0-9.]+)", ln); g=re.search(r"gfx=(\d+)", ln)
            if m and m.group(1).isdigit() and int(m.group(1))>0: ipu.append(int(m.group(1)))
            if p: pkg.append(float(p.group(1)))
            if g and int(g.group(1))>0: gfx.append(int(g.group(1)))
    except FileNotFoundError:
        print(tag, "no samples"); continue
    f=lambda v: ("n=%d mean=%.1f min=%d max=%d" % (len(v), sum(v)/len(v), min(v), max(v))) if v else "none"
    print("%-5s IPUclk(busy) %-40s" % (tag, f(ipu)))
    print("      package      %-40s" % f(pkg))
    print("      gfxclk       %-40s" % f(gfx))
EOF
