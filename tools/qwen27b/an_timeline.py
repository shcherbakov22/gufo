
import sqlite3, collections, sys
db=sqlite3.connect(sys.argv[1] if len(sys.argv)>1 else "/tmp/rocprof/tr_results.db"); c=db.cursor()
tabs=[r[0] for r in c.execute("select name from sqlite_master where type='table'")]
kd=[t for t in tabs if t.startswith("rocpd_kernel_dispatch")][0]
ks=[t for t in tabs if t.startswith("rocpd_info_kernel_symbol")][0]
rows=list(c.execute("select d.start,d.end,d.stream_id,s.kernel_name from %s d left join %s s on d.kernel_id=s.id order by d.start"%(kd,ks)))
tot=collections.Counter(); cnt=collections.Counter()
for st,en,sid,nm in rows: tot[nm]+=en-st; cnt[nm]+=1
print("dispatches %d span %.3f s"%(len(rows),(rows[-1][0]-rows[0][0])/1e9))
for nm,t in tot.most_common(8): print("  %-30s %8.1f ms x%d (%.3f each)"%(str(nm)[:30],t/1e6,cnt[nm],t/1e6/cnt[nm]))
byst=collections.defaultdict(list)
for st,en,sid,nm in rows: byst[sid].append((st,en,nm))
for sid,ev in sorted(byst.items(), key=lambda kv:-sum(e-s for s,e,_ in kv[1])):
    ev.sort(); busy=sum(e-s for s,e,_ in ev); span=ev[-1][1]-ev[0][0]
    gaps=sorted(((ev[i][0]-ev[i-1][1]),ev[i-1][2],ev[i][2]) for i in range(1,len(ev)))
    print("stream %s: busy %.1f ms span %.1f ms idle %.1f%% gaps>0.5ms=%d"%(sid,busy/1e6,span/1e6,100*(1-busy/span),sum(1 for g in gaps if g[0]>500000)))
    for g,a,b in sorted(gaps,reverse=True)[:5]: print("   gap %7.3f ms after %s"%(g/1e6,str(a)[:40]))
rp=[(st,en,nm) for st,en,sid,nm in rows if nm and "Repack" in nm]
if rp: print("Repack: n=%d total %.1f ms (%.3f each)"%(len(rp),sum(e-s for s,e,_ in rp)/1e6,sum(e-s for s,e,_ in rp)/1e6/len(rp)))
