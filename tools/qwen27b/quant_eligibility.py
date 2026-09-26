#!/usr/bin/env python3
"""Report a GGUF int4 eligibility from its header alone.

The int4 WMMA needs both operands on a *linear* grid, so what matters is each
tensor stored type, not the shard name -- Unsloth's UD-Q3_K_XL is 12% linear.
The GGUF header carries every tensor type and sits at the front of the file, so
an HTTP Range request for the first few MB answers the question without
downloading the model.

Usage:
  tools/qwen27b/quant_eligibility.py <file.gguf> [more.gguf ...]
  tools/qwen27b/quant_eligibility.py --hf <repo> <name.gguf> ...
"""
from __future__ import annotations
import collections, struct, subprocess, sys

# Linear (code proportional to value) -> int4-eligible, per IsInt4Eligible.
ELIGIBLE = {'Q4_0', 'Q4_1', 'Q4_K', 'Q3_K', 'Q2_K', 'Q2_0', 'TQ1_0', 'TQ2_0', 'Q1_0'}
TYPES = {0:'F32',1:'F16',2:'Q4_0',3:'Q4_1',6:'Q5_0',7:'Q5_1',8:'Q8_0',9:'Q8_1',
         10:'Q2_K',11:'Q3_K',12:'Q4_K',13:'Q5_K',14:'Q6_K',15:'Q8_K',16:'IQ2_XXS',
         17:'IQ2_XS',18:'IQ3_XXS',20:'IQ4_NL',21:'IQ3_S',22:'IQ2_S',23:'IQ4_XS',30:'BF16'}
FIXED = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}

def parse(path):
    data = open(path, 'rb').read()
    if data[:4] != b'GGUF': raise ValueError('not a GGUF file')
    off = 4
    version, = struct.unpack_from('<I', data, off); off += 4
    tensors, = struct.unpack_from('<Q', data, off); off += 8
    pairs, = struct.unpack_from('<Q', data, off); off += 8
    def read_str():
        nonlocal off
        n, = struct.unpack_from('<Q', data, off); off += 8
        s = data[off:off + n].decode('utf-8', 'replace'); off += n
        return s
    def skip(type_id):
        nonlocal off
        if type_id == 8: read_str(); return
        if type_id == 9:
            elem, = struct.unpack_from('<I', data, off); off += 4
            count, = struct.unpack_from('<Q', data, off); off += 8
            if elem == 8:
                for _ in range(count): read_str()
            else:
                off += FIXED[elem] * count
            return
        off += FIXED[type_id]
    for _ in range(pairs):
        read_str()
        type_id, = struct.unpack_from('<I', data, off); off += 4
        skip(type_id)
    counts, elements = collections.Counter(), collections.Counter()
    total = 0
    for _ in range(tensors):
        read_str()
        dims, = struct.unpack_from('<I', data, off); off += 4
        count = 1
        for _ in range(dims):
            dim, = struct.unpack_from('<Q', data, off); off += 8; count *= dim
        type_id, = struct.unpack_from('<I', data, off); off += 4
        off += 8
        name = TYPES.get(type_id, 'T%d' % type_id)
        counts[name] += 1; elements[name] += count; total += count
    return version, tensors, counts, elements, total

def report(label, path):
    version, tensors, counts, elements, total = parse(path)
    eligible = sum(v for k, v in elements.items() if k in ELIGIBLE)
    print('%-24s v%d %4d tensors  %.2fG el  eligible %5.1f%%' % (
        label, version, tensors, total / 1e9, 100.0 * eligible / total))
    for name, count in elements.most_common(8):
        print('    %-9s %4d tensors %7.3fG %5.1f%%' % (
            name, counts[name], count / 1e9, 100.0 * count / total))

def main(argv):
    if len(argv) > 2 and argv[1] == '--hf':
        repo = argv[2]
        for name in argv[3:]:
            url = 'https://huggingface.co/%s/resolve/main/%s' % (repo, name)
            out = '/tmp/gguf-header-%s' % name.rsplit('/', 1)[-1]
            subprocess.run(['curl', '-sL', '-r', '0-25165823', '-o', out, url], check=True)
            report(name, out)
    else:
        for path in argv[1:]: report(path, path)
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))