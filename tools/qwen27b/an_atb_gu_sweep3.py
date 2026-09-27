
import re, collections, statistics as st
rows=[]
for ln in open("/tmp/atb_gu_sweep3.log"):
    m=re.match(r'round=(\d+) pos=(\d+) gu=(\d+)\s+startTctl=(\d+)C\s+([0-9.]+)', ln)
    if m: rows.append(dict(round=int(m.group(1)), pos=int(m.group(2)), gu=int(m.group(3)),
                           tctl=int(m.group(4)), v=float(m.group(5))))
print("measurements: %d" % len(rows))
if not rows: raise SystemExit
gus=sorted(set(r['gu'] for r in rows))
print()
print("%-6s %s" % ("round", "  ".join("%-8s"%g for g in gus)))
for rd in sorted(set(r['round'] for r in rows)):
    cells=[]
    for g in gus:
        vs=[r['v'] for r in rows if r['round']==rd and r['gu']==g]
        cells.append("%-8s"%("%.1f"%vs[0] if vs else "-"))
    print("%-6d %s" % (rd, "  ".join(cells)))
print()
best=None
for g in gus:
    vs=[r['v'] for r in rows if r['gu']==g]
    if vs:
        m=sum(vs)/len(vs)
        print("gu=%-6d n=%d mean=%7.2f  min=%.1f max=%.1f  spread=%.1f%%" % (
            g, len(vs), m, min(vs), max(vs), 100*(max(vs)-min(vs))/m))
        if best is None or m>best[1]: best=(g,m)
if best: print("\nbest mean: gu=%d  %.2f" % best)
