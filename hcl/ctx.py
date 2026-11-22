"""Context-engineering toolkit: a shared parser State, hierarchical context experts,
and a geometric (logistic-style) mixer.  Every geometry/topology/harmonic experiment
in the casebook is 'one more expert' plugged into this same ensemble, so gains are
measured against identical machinery and identical controls.
"""
from __future__ import annotations
import math
import numpy as np

OPEN = {ord("{"): "{", ord("["): "["}
CLOSE = {ord("}"): "{", ord("]"): "["}


class State:
    """Decoder-side parser state: everything here is a function of already-decoded bytes."""

    def __init__(self):
        self.hist = b""
        self.cur = bytearray()          # current line
        self.prev = b""                 # previous line
        self.stack = []                 # open '{{' / '[[' pairs  (planar, non-crossing)
        self.last_consumed = False
        self.word = b""; self.pword = b""; self.ppword = b""
        self.n = 0
        self.buf = bytearray()

    def lag(self, k):
        return self.buf[-k] if len(self.buf) >= k else 0

    # convenience views --------------------------------------------------
    def above(self, dx=0):
        j = len(self.cur) + dx
        return self.prev[j] if 0 <= j < len(self.prev) else 0

    def left(self, k=1):
        return self.hist[-k] if len(self.hist) >= k else 0

    def update(self, b):
        last = self.hist[-1] if self.hist else 0
        if b in OPEN and last == b and not self.last_consumed:
            self.stack.append(OPEN[b]); self.last_consumed = True
        elif b in CLOSE and last == b and not self.last_consumed:
            if self.stack and self.stack[-1] == CLOSE[b]:
                self.stack.pop()
            self.last_consumed = True
        else:
            self.last_consumed = False
        if b == 10:
            self.prev, self.cur = bytes(self.cur), bytearray()
        else:
            self.cur.append(b)
        if (65 <= b <= 90) or (97 <= b <= 122):
            self.word += bytes([b | 32])
        elif self.word:
            self.ppword, self.pword, self.word = self.pword, self.word, b""
        self.hist = (self.hist + bytes([b]))[-16:]
        self.buf.append(b)
        self.n += 1


class CtxExpert:
    """Hierarchical-Dirichlet backoff over a coarse->fine list of context keys."""

    def __init__(self, name, keyfn, beta=2.0):
        self.name, self.keyfn, self.beta = name, keyfn, beta
        self.tabs = []

    def probs(self, st: State):
        keys = self.keyfn(st)
        while len(self.tabs) < len(keys):
            self.tabs.append({})
        self.keys = keys
        p = np.full(256, 1 / 256)
        for lvl, k in enumerate(keys):
            row = self.tabs[lvl].get(k)
            if row is not None:
                p = (row[0] + self.beta * p) / (row[1] + self.beta)
        return p

    def update(self, b):
        for lvl, k in enumerate(self.keys):
            row = self.tabs[lvl].get(k)
            if row is None:
                row = [np.zeros(256), 0.0]; self.tabs[lvl][k] = row
            row[0][b] += 1; row[1] += 1


def order_keys(maxo):
    return lambda st: [st.hist[-k:] if k else b"" for k in range(maxo + 1)]


class Ensemble:
    """experts[0] is the base model; others are 'candidate innovations' and controls."""

    def __init__(self, experts, lr=0.005, mode="geometric", init_base=True):
        self.st = State(); self.experts = experts; self.lr = lr; self.mode = mode
        K = len(experts)
        self.w = np.zeros(K)
        if mode == "geometric":
            if init_base: self.w[0] = 1.0
            else: self.w[:] = 1.0 / K
        self.p = None

    def probs(self):
        P = np.array([e.probs(self.st) for e in self.experts])
        self.P = P
        if self.mode == "base_only":
            self.p = P[0]; return self.p
        self.L = np.log(P)
        s = self.w @ self.L; z = np.exp(s - s.max()); p = z / z.sum()
        self.p = 0.999999 * p + 0.000001 / 256
        return self.p

    def update(self, b):
        if self.mode != "base_only":
            g = -self.L[:, b] + self.L @ self.p
            self.w -= self.lr * np.clip(g, -5, 5)
        for e in self.experts:
            e.update(b)
        self.st.update(b)


def run_bits(data: bytes, ens: Ensemble) -> float:
    total = 0.0
    for b in data:
        p = ens.probs()
        total -= math.log2(p[b])
        ens.update(b)
    return total
