import re, sys
H="/home/q/linux-7.3rc/drivers/gpu/drm/amd/pm/swsmu/inc/pmfw_if/smu14_driver_if_v14_0.h"
TARGETS=("PFE_Settings_t","SkuTable_t","CustomSkuTable_t","BoardTable_t")
PAT=re.compile(r"^\s*(uint8_t|int8_t|uint16_t|int16_t|uint32_t|int32_t|uint64_t|int64_t|float)\s+([A-Za-z_]\w*)\s*(\[\s*([A-Za-z_0-9]+)\s*\])?\s*;")
FMT={"uint8_t":"%u","int8_t":"%d","uint16_t":"%u","int16_t":"%d","uint32_t":"%u",
     "int32_t":"%d","uint64_t":"%llu","int64_t":"%lld","float":"%g"}
CAST={"uint8_t":"unsigned","int8_t":"int","uint16_t":"unsigned","int16_t":"int",
      "uint32_t":"unsigned","int32_t":"int","uint64_t":"unsigned long long",
      "int64_t":"long long","float":"double"}
out={}; cur=[]; ifdefs=0
for ln in open(H).read().split("\n"):
    s=ln.strip()
    if s.startswith("#if"): ifdefs+=1
    if s.startswith("typedef struct"): cur=[]; continue
    m=re.match(r"^}\s*([A-Za-z_]\w*)\s*;", s)
    if m:
        if m.group(1) in TARGETS: out[m.group(1)]=cur
        cur=[]; continue
    if s.startswith("//") or not s: continue
    mm=PAT.match(ln)
    if mm: cur.append((mm.group(1), mm.group(2), mm.group(4)))

KEY=re.compile(r"Limit|Tdc|Temperature|Stt|Stapm|Ppt|Tau|Ctf|Power", re.I)
L=["#include <stdio.h>","#include <stdint.h>","#include <stddef.h>","#include <stdlib.h>",
   '#include "smu14_driver_if_v14_0.h"',"int main(void){",
   '  FILE *f=fopen("/home/q/npu/fw/smu_14_0_3.bin","rb");',
   '  if(!f){perror("open");return 1;}',
   "  unsigned char *b=malloc(0x16B4);",
   "  fseek(f,0x4FF00,SEEK_SET); fread(b,1,0x16B4,f); fclose(f);",
   "  PPTable_t *p=(PPTable_t*)(b+1344);",
   '  printf("sizeof(PPTable_t)=%zu   blob=0x16B4   PPTable_t at +1344\\n\\n",sizeof(PPTable_t));']
n=0; listing=[]
for st in TARGETS:
    L.append('  printf("---- ' + st + ' ----\\n");')
    for (ty, fn, arr) in out.get(st, []):
        if not KEY.search(fn): continue
        n+=1
        listing.append("%-18s %-10s %s" % (st, ty, fn + ("[%s]" % arr if arr else "")))
        path = "p->" + st + "." + fn
        sizep = "sizeof(((PPTable_t*)0)->" + st + "." + fn + ")"
        if arr:
            L.append("  { size_t n_=" + sizep + "/sizeof(((PPTable_t*)0)->" + st + "." + fn + "[0]);")
            L.append('    printf("  %-40s @%5zu =", "' + fn + '", offsetof(' + st + ',' + fn + '));')
            L.append('    for(size_t k_=0;k_<n_ && k_<8;k_++) printf(" ' + FMT[ty] + '",(' + CAST[ty] + ')' + path + '[k_]);')
            L.append('    printf("  (n=%zu)\\n", n_); }')
        else:
            L.append('  printf("  %-40s @%5zu = ' + FMT[ty] + '\\n", "' + fn + '", offsetof(' + st + ',' + fn + '), (' + CAST[ty] + ')' + path + ');')
L.append("  return 0; }")
open("/home/q/smu/ppt_dump.c","w").write("\n".join(L))
print("conditional blocks: %d ; generated %d fields" % (ifdefs, n), file=sys.stderr)
print("\n".join(listing))
