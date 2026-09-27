import shutil, sys, difflib
src = "/home/q/smu/limine.conf.orig"
dst = "/home/q/smu/limine.conf.new"
lines = open(src).read().split("\n")
try:
    i = lines.index("  //linux-perfopt")
except ValueError:
    sys.exit("could not find //linux-perfopt marker")
# the cmdline for this entry is the first cmdline: line after the marker
j = next(k for k in range(i, len(lines)) if lines[k].lstrip().startswith("cmdline:"))
if "amdgpu.fw_load_type" in lines[j]:
    sys.exit("already present: " + lines[j])
lines[j] = lines[j].rstrip() + " amdgpu.fw_load_type=0"
open(dst, "w").write("\n".join(lines))
print("marker line %d, cmdline line %d" % (i + 1, j + 1))
print()
for line in difflib.unified_diff(open(src).read().splitlines(True),
                                 open(dst).read().splitlines(True),
                                 fromfile="limine.conf (current)", tofile="limine.conf (new)",
                                 n=2):
    print(line.rstrip())
