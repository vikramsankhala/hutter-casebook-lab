"""Entropy coders used throughout the casebook.

  * ExactArithmetic   - Fraction-based interval narrowing (teaching, Chapter 3)
  * RangeEncoder/Decoder - 32-bit integer arithmetic coder with underflow
                          handling and *multi-symbol* cumulative-frequency input
  * rans_encode/decode - unbounded-integer range ANS (Chapter 4)

All coders consume integer cumulative frequencies, so a binary alphabet is just
the special case of two symbols.
"""
from __future__ import annotations
from fractions import Fraction
import math
import numpy as np


# ---------------------------------------------------------------- ideal cost
def ideal_bits(p: float) -> float:
    return -math.log2(p)


def entropy(p) -> float:
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


# ----------------------------------------------------------- exact fractions
class ExactArithmetic:
    """Interval narrowing with exact rationals. probs: dict symbol -> Fraction."""

    def __init__(self, probs):
        self.symbols = list(probs)
        self.cum, c = {}, Fraction(0)
        for s in self.symbols:
            self.cum[s] = (c, c + probs[s])
            c += probs[s]
        assert c == 1

    def encode(self, seq):
        lo, hi = Fraction(0), Fraction(1)
        for s in seq:
            w = hi - lo
            a, b = self.cum[s]
            lo, hi = lo + w * a, lo + w * b
        return lo, hi

    def decode(self, point, n):
        lo, hi, out = Fraction(0), Fraction(1), []
        for _ in range(n):
            w = hi - lo
            for s in self.symbols:
                a, b = self.cum[s]
                if lo + w * a <= point < lo + w * b:
                    out.append(s)
                    lo, hi = lo + w * a, lo + w * b
                    break
        return out

    @staticmethod
    def shortest_codeword(lo, hi):
        """Shortest dyadic k/2^m inside [lo,hi): returns (bits, value)."""
        m = 0
        while True:
            m += 1
            k = math.ceil(lo * 2 ** m)
            if Fraction(k, 2 ** m) < hi:
                return m, Fraction(k, 2 ** m)


# ------------------------------------------------------------ integer range
PREC = 32
TOP = (1 << PREC) - 1
HALF = 1 << (PREC - 1)
QTR = 1 << (PREC - 2)


class RangeEncoder:
    def __init__(self):
        self.low, self.high, self.pending = 0, TOP, 0
        self.bits = []

    def _emit(self, b):
        self.bits.append(b)
        self.bits.extend([1 - b] * self.pending)
        self.pending = 0

    def encode(self, c_lo, c_hi, total):
        rng = self.high - self.low + 1
        self.high = self.low + rng * c_hi // total - 1
        self.low = self.low + rng * c_lo // total
        while True:
            if self.high < HALF:
                self._emit(0)
            elif self.low >= HALF:
                self._emit(1)
                self.low -= HALF
                self.high -= HALF
            elif self.low >= QTR and self.high < HALF + QTR:
                self.pending += 1          # underflow: deferred bit
                self.low -= QTR
                self.high -= QTR
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1

    def finish(self) -> bytes:
        self.pending += 1
        self._emit(0 if self.low < QTR else 1)
        bits = self.bits
        bits = bits + [0] * ((-len(bits)) % 8)
        out = bytearray()
        for i in range(0, len(bits), 8):
            v = 0
            for b in bits[i:i + 8]:
                v = (v << 1) | b
            out.append(v)
        return bytes(out)


class RangeDecoder:
    def __init__(self, data: bytes):
        self.bits = [(byte >> (7 - k)) & 1 for byte in data for k in range(8)]
        self.pos = 0
        self.low, self.high, self.code = 0, TOP, 0
        for _ in range(PREC):
            self.code = (self.code << 1) | self._next()

    def _next(self):
        b = self.bits[self.pos] if self.pos < len(self.bits) else 0
        self.pos += 1
        return b

    def target(self, total):
        rng = self.high - self.low + 1
        return ((self.code - self.low + 1) * total - 1) // rng

    def consume(self, c_lo, c_hi, total):
        rng = self.high - self.low + 1
        self.high = self.low + rng * c_hi // total - 1
        self.low = self.low + rng * c_lo // total
        while True:
            if self.high < HALF:
                pass
            elif self.low >= HALF:
                self.low -= HALF
                self.high -= HALF
                self.code -= HALF
            elif self.low >= QTR and self.high < HALF + QTR:
                self.low -= QTR
                self.high -= QTR
                self.code -= QTR
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1
            self.code = (self.code << 1) | self._next()


def quantize(p: np.ndarray, total: int = 1 << 16) -> np.ndarray:
    """Probabilities -> integer frequencies, each >= 1, summing exactly to total."""
    n = len(p)
    f = 1 + np.floor(np.asarray(p, dtype=np.float64) * (total - n)).astype(np.int64)
    f[int(np.argmax(f))] += total - int(f.sum())
    return f


def encode_with_model(data: bytes, model_factory, total: int = 1 << 16) -> bytes:
    """Multi-symbol coding: one 256-way distribution per byte from `model`."""
    m = model_factory()
    enc = RangeEncoder()
    for byte in data:
        f = quantize(m.probs(), total)
        c = np.concatenate(([0], np.cumsum(f)))
        enc.encode(int(c[byte]), int(c[byte + 1]), total)
        m.update(byte)
    return enc.finish()


def decode_with_model(blob: bytes, n: int, model_factory, total: int = 1 << 16) -> bytes:
    m = model_factory()
    dec = RangeDecoder(blob)
    out = bytearray()
    for _ in range(n):
        f = quantize(m.probs(), total)
        c = np.concatenate(([0], np.cumsum(f)))
        t = dec.target(total)
        s = int(np.searchsorted(c, t, side="right") - 1)
        dec.consume(int(c[s]), int(c[s + 1]), total)
        out.append(s)
        m.update(s)
    return bytes(out)


# --------------------------------------------------------------------- rANS
def rans_encode(symbols, freqs):
    """Unbounded-state rANS (Chapter 4). Encodes in reverse; decoder reads forward."""
    M = sum(freqs)
    cum = [0]
    for f in freqs:
        cum.append(cum[-1] + f)
    x = 0
    for s in reversed(symbols):
        q, r = divmod(x, freqs[s])
        x = M * q + cum[s] + r
    return x


def rans_decode(x, n, freqs):
    M = sum(freqs)
    cum = [0]
    for f in freqs:
        cum.append(cum[-1] + f)
    out = []
    for _ in range(n):
        z = x % M
        s = max(i for i in range(len(freqs)) if cum[i] <= z)
        x = freqs[s] * (x // M) + z - cum[s]
        out.append(s)
    return out
