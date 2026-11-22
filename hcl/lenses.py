"""Reversible preprocessing as LENSES (the functorial view, Chapter 15).

A lens is a pair  get : S -> (V, C)   and   put : (V, C) -> S  with the laws
    PutGet:  put(get(s)) == s                  (nothing is lost: lossless)
    GetPut:  get(put(v, c)) == (v, c)          (the view/complement split is canonical)
V is what the downstream model sees; C (the complement) is the residue that must be paid for.
Lenses compose, and composition preserves both laws - that is the whole 'functor' we need.
"""
from __future__ import annotations
import re
from collections import Counter


class Lens:
    def get(self, s): raise NotImplementedError
    def put(self, v, c): raise NotImplementedError


class Compose(Lens):
    def __init__(self, *lenses): self.ls = lenses
    def get(self, s):
        comps = []
        for l in self.ls:
            s, c = l.get(s); comps.append(c)
        return s, tuple(comps)
    def put(self, v, c):
        for l, ci in zip(reversed(self.ls), reversed(c)):
            v = l.put(v, ci)
        return v


class CaseLens(Lens):
    """View: ASCII-lowercased text. Complement: gaps between upper-case letters."""
    def get(self, s: bytes):
        gaps, last = [], -1
        for i, b in enumerate(s):
            if 65 <= b <= 90:
                gaps.append(i - last); last = i
        return s.lower(), tuple(gaps)
    def put(self, v: bytes, c):
        out, pos = bytearray(v), -1
        for g in c:
            pos += g; out[pos] -= 32
        return bytes(out)


class WordDictLens(Lens):
    """View: frequent whole words replaced by one byte >= 0x80. Complement: the dictionary itself."""
    def __init__(self, k=100, minlen=4):
        self.k, self.minlen = k, minlen
    def get(self, s: bytes):
        assert all(b < 128 for b in s), "dictionary bytes must be unused in the source"
        cnt = Counter(re.findall(rb"[a-z]{%d,}" % self.minlen, s))
        gain = {w: (len(w) - 1) * c for w, c in cnt.items()}
        words = tuple(sorted(sorted(gain, key=gain.get, reverse=True)[:self.k]))
        table = {w: bytes([128 + i]) for i, w in enumerate(words)}
        pat = re.compile(rb"(?<![a-z])(" + b"|".join(map(re.escape, words)) + rb")(?![a-z])") if words else None
        v = pat.sub(lambda m: table[m.group(1)], s) if pat else s
        return v, words
    def put(self, v: bytes, c):
        out = bytearray()
        for b in v:
            out += c[b - 128] if b >= 128 else bytes([b])
        return bytes(out)


def check_laws(lens: Lens, samples):
    ok1 = ok2 = 0
    for s in samples:
        v, c = lens.get(s)
        ok1 += lens.put(v, c) == s
        ok2 += lens.get(lens.put(v, c)) == (v, c)
    return dict(samples=len(samples), put_get=ok1, get_put=ok2)
