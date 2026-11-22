"""E02 - Entropy coding through multi-symbol operations (Case 3-4).

Same Dirichlet-style order-2 backoff idea, three alphabets:
  256-ary symbol model | 8-level binary tree | Huffman-shaped binary tree | nibble (16-ary x 2)
Reports ideal bits and the number of coder calls per byte (a speed proxy).
"""
import math, heapq
from collections import Counter
import numpy as np
from hcl import models
from experiments.common import get_data, save

N = 60_000


def huffman_code(counts):
    heap = [[w, i, (s,)] for i, (s, w) in enumerate(counts.items())]
    heapq.heapify(heap)
    codes = {s: "" for s in counts}
    uid = len(heap)
    while len(heap) > 1:
        a = heapq.heappop(heap); b = heapq.heappop(heap)
        for s in a[2]: codes[s] = "0" + codes[s]
        for s in b[2]: codes[s] = "1" + codes[s]
        heapq.heappush(heap, [a[0] + b[0], uid, a[2] + b[2]]); uid += 1
    return codes


def tree_cost(data, code, order=2, beta=2.0):
    """Binary-decision decomposition; per-node backoff over contexts 0..order."""
    tabs = [dict() for _ in range(order + 1)]
    hist = b""; bits = 0.0; calls = 0
    for b in data:
        path = code[b]; prefix = ""
        for ch in path:
            p1 = 0.5
            keys = []
            for k in range(order + 1):
                key = (hist[-k:] if k else b"", prefix)
                keys.append(key)
                n = tabs[k].get(key)
                if n is None: continue
                p1 = (n[1] + beta * p1) / (n[0] + n[1] + beta)
            bit = int(ch)
            bits -= math.log2(p1 if bit else 1 - p1); calls += 1
            for k in range(order + 1):
                n = tabs[k].setdefault(keys[k], [0, 0]); n[bit] += 1
            prefix += ch
        hist = (hist + bytes([b]))[-8:]
    return bits, calls


def run():
    d = get_data(N)
    res = {"bytes": len(d)}
    b = models.cost_bits(d, models.Backoff(order=2))
    res["symbol_256ary"] = dict(bpb=round(b / len(d), 4), coder_calls_per_byte=1.0)
    fixed = {i: format(i, "08b") for i in range(256)}
    bits, calls = tree_cost(d, fixed)
    res["binary_fixed8"] = dict(bpb=round(bits / len(d), 4), coder_calls_per_byte=round(calls / len(d), 2))
    huff = huffman_code(Counter(d))
    bits, calls = tree_cost(d, huff)
    res["binary_huffman_tree"] = dict(bpb=round(bits / len(d), 4), coder_calls_per_byte=round(calls / len(d), 2),
                                      note="tree built from corpus order-0 counts; table cost not charged")
    return save("e02_multisymbol", res)


if __name__ == "__main__":
    print(run())
