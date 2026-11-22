"""Predictive byte models. Every model exposes probs() -> np.ndarray(256) and update(byte).

Costs are *ideal* code lengths (-log2 p) computed score-before-update, which is
what the decoder can reproduce.  coders.encode_with_model verifies the ideal cost
against a real integer arithmetic coder.
"""
from __future__ import annotations
import math
import numpy as np

ALPHA = 256


class Uniform:
    def probs(self):
        return np.full(256, 1 / 256)

    def update(self, b):
        pass


class Backoff:
    """Interpolated (hierarchical-Dirichlet style) order-k byte model.

    p_0 = (c+1)/(t+256);  p_k = (c_k + beta * p_{k-1}) / (t_k + beta).
    `experts()` returns each order's distribution for the mixing experiments.
    """

    def __init__(self, order=3, beta=2.0, orders=None):
        self.orders = list(orders) if orders is not None else list(range(order + 1))
        self.beta = beta
        self.tables = {k: {} for k in self.orders}
        self.hist = b""

    def _ctx(self, k):
        return self.hist[-k:] if k else b""

    def experts(self):
        prev = np.full(256, 1 / 256)
        out = []
        for k in self.orders:
            row = self.tables[k].get(self._ctx(k))
            if row is None:
                p = prev
            else:
                c, t = row
                p = (c + self.beta * prev) / (t + self.beta)
            out.append(p)
            prev = p
        return out

    def probs(self):
        return self.experts()[-1]

    def update(self, b):
        for k in self.orders:
            key = self._ctx(k)
            row = self.tables[k].get(key)
            if row is None:
                row = [np.zeros(256), 0.0]
                self.tables[k][key] = row
            row[0][b] += 1
            row[1] += 1
        self.hist = (self.hist + bytes([b]))[-8:]


def cost_bits(data: bytes, model) -> float:
    """Total ideal bits, score-before-update."""
    total = 0.0
    for b in data:
        p = model.probs()
        total -= math.log2(p[b])
        model.update(b)
    return total


def cost_trace(data: bytes, model) -> np.ndarray:
    out = np.empty(len(data))
    for i, b in enumerate(data):
        p = model.probs()
        out[i] = -math.log2(p[b])
        model.update(b)
    return out


def bpb(bits: float, n: int) -> float:
    return bits / n
