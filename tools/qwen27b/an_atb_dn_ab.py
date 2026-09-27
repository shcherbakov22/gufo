
import re, collections, statistics as st
rows=[]
for ln in open("/tmp/atb_dn_ab.log"):
    m=re.match(r'round=(\d+) pos=(\d+) arm=(\S+)\s+gu=(\d+)\s+dn=(\d+)\s+([0-9.]+)', ln)
    if m:
        rows.append(dict(round=int(m.group(1)), pos=int(m.group(2)), arm=m.group(3),
                         gu=int(m.group(4)), dn=int(m.group(5)), v=float(m.group(6))))
print("measurements: %d" % len(rows))
if not rows:
    print("(no data yet)"); raise SystemExit
arms=[]
for r in rows:
    if r['arm'] not in arms: arms.append(r['arm'])
print()
print("=== by round (within-round comparison is the valid one) ===")
print("%-6s %s" % ("round", "  ".join("%-10s"%a for a in arms)))
for rd in sorted(set(r['round'] for r in rows)):
    cells=[]
    for a in arms:
        vs=[r['v'] for r in rows if r['round']==rd and r['arm']==a]
        cells.append("%-10s" % ("%.1f"%vs[0] if vs else "-"))
    print("%-6d %s" % (rd, "  ".join(cells)))
print()
print("=== per-arm ===")
base_by_round={r['round']:r['v'] for r in rows if r['arm']=='base'}
for a in arms:
    vs=[r['v'] for r in rows if r['arm']==a]
    pct=[]
    for r in rows:
        if r['arm']==a and r['round'] in base_by_round:
            pct.append(100.0*(r['v']-base_by_round[r['round']])/base_by_round[r['round']])
    line="%-10s n=%d mean=%.1f" % (a, len(vs), sum(vs)/len(vs))
    if a!='base' and pct:
        line += "   vs base (paired per round): " + " ".join("%+.1f%%"%p for p in pct)
        if len(pct)>1:
            line += "  | mean %+.2f%%  median %+.2f%%" % (sum(pct)/len(pct), st.median(pct))
    print(line)
