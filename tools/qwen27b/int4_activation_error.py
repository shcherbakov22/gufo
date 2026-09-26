import sys, mmap, numpy as np
sys.path.insert(0, '/home/q/gufo/tools/gufo')
import gguf

PATH = '/home/q/Downloads/Qwen3.8-27B-IQ4_XS-3.84bpw.gguf'
f = open(PATH, 'rb')
mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
st = gguf._parse_gguf_data(mm)

# pick a real Q4_K ffn_gate (the tensor that consumes the dumped activation)
cand = None
for t in st['tensors']:
    if t['type'] == 12 and t['name'].endswith('ffn_gate.weight') and t['shape'][-1] == 5120:
        cand = t; break
if cand is None:
    for t in st['tensors']:
        if t['type'] == 12 and t['shape'][-1] == 5120 and t['shape'][0] >= 256:
            cand = t; break
print('weight tensor:', cand['name'], cand['type_name'], cand['shape'])

ROWS = 256
row_bytes = 5120 // 256 * 144
base = st['data_offset'] + cand['offset']
raw = np.frombuffer(mm, dtype=np.uint8, count=ROWS*row_bytes, offset=base)
W = gguf._dequant_q4_K(raw).reshape(ROWS, 5120).astype(np.float32)

A = np.fromfile('/tmp/act.bin', dtype=np.float16).astype(np.float32).reshape(2048, 5120)
TOK = 256
A = A[:TOK].copy()

def rms(x): return np.sqrt((x.astype(np.float64)**2).mean())

# per-32 block stats, weights and activations
def stats(M, block, tag):
    n = M.shape[1]; nb = n // block
    B = M[:, :nb*block].reshape(M.shape[0], nb, block)
    amax = np.abs(B).max(axis=2); r = np.sqrt((B.astype(np.float64)**2).mean(axis=2))
    m = r > 0
    q = (amax[m]/r[m])
    print('%-12s per-%d amax/rms: mean %.2f median %.2f p90 %.2f max %.2f' %
          (tag, block, q.mean(), np.median(q), np.percentile(q,90), q.max()))

stats(W, 32, 'weights')
stats(A, 32, 'activations')

def quant(A, bits, block=32):
    n = A.shape[1]; nb = n // block
    B = A[:, :nb*block].reshape(A.shape[0], nb, block)
    amax = np.abs(B).max(axis=2, keepdims=True)
    top = (1 << (bits-1)) - 1
    sc = np.where(amax > 0, amax/top, 1.0)
    c = np.clip(np.rint(B/sc), -top-1, top)
    out = A.copy()
    out[:, :nb*block] = (c*sc).reshape(A.shape[0], nb*block)
    return out

def relerr(Y, R): return np.abs(Y-R).mean() if False else np.sqrt(((Y-R).astype(np.float64)**2).mean())/rms(R)

W64 = W.astype(np.float64); A64 = A.astype(np.float64)
Yref = W64 @ A64.T
print()
print('%-34s %-12s' % ('GEMM variant (weight side identical)', 'rel err'))
for tag, Ab in [('fp16/fp32 activations', A64),
                ('int8 per-32 activations', quant(A, 8)),
                ('int4 per-32 activations', quant(A, 4)),
                ('int4 per-16 activations', quant(A, 4, 16)),
                ('int4 per-8 activations', quant(A, 4, 8))]:
    Y = W64 @ Ab.astype(np.float64).T
    print('%-34s %-12.5f' % (tag, relerr(Y, Yref)))

# weight-grid error, for the sqrt(2) comparison
Wq = quant(W, 4)
print()
print('%-34s %-12.5f' % ('weight 4-bit grid rel err', rms(W-Wq)/rms(W)))
