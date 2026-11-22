"""E06 - Euclidean translations (Case 10).

(a) A backward copy IS a translation of the sequence by -d.  Do translations repeat?
    Compare Elias-delta offsets with a 4-entry repeat-offset (move-to-front) cache.
(b) Text is also a 2D lattice (line, column).  A translation by one row (dy=-1) gives the
    byte ABOVE.  Add (above, left) as an expert; compare with a noise control expert.
"""
import math
from collections import defaultdict
from hcl.ctx import CtxExpert, Ensemble, order_keys, run_bits
from experiments.common import get_data, save

N_A, N_B = 100_000, 30_000


def delta(n):
    l = int(math.log2(n)); return l + 2 * int(math.log2(l + 1)) + 1


def offsets_greedy(d, anchor=5, cand=24, minlen=6):
    idx = defaultdict(list); i = 0; offs = []
    pos = 0
    n = len(d)
    while i < n - anchor:
        best = (0, 0)
        for j in idx[d[i:i + anchor]][-cand:]:
            L = 0
            while i + L < n and L < 320 and d[j + L] == d[i + L]: L += 1
            if L > best[0]: best = (L, i - j)
        step = best[0] if best[0] >= minlen else 1
        for t in range(i, min(i + step, n - anchor)):
            idx[d[t:t + anchor]].append(t)
        if best[0] >= minlen: offs.append(best[1])
        i += step
    return offs


def part_a():
    d = get_data(N_A)
    offs = offsets_greedy(d)
    plain = sum(1 + delta(o) for o in offs)
    cache, hits, cached = [], 0, 0
    for o in offs:
        if o in cache:
            cached += 1 + 2; hits += 1; cache.remove(o)
        else:
            cached += 1 + 1 + delta(o)
        cache.insert(0, o); del cache[4:]
    return dict(copies=len(offs), plain_bits=plain, repeat_cache_bits=cached,
                cache_hit_rate=round(hits / len(offs), 4), gain_bytes=round((plain - cached) / 8))


def part_b():
    d = get_data(N_B)
    base = lambda: CtxExpert("order3", order_keys(3))
    above = lambda: CtxExpert("above_left", lambda st: [b"", bytes([st.above()]), bytes([st.above(), st.left()]),
                                                       bytes([st.above(), st.left(), st.left(2)])])
    noise = lambda: CtxExpert("noise_col_mod7", lambda st: [b"", bytes([len(st.cur) % 7]), bytes([len(st.cur) % 7, st.left()])])
    out = {}
    out["base_order3_bpb"] = round(run_bits(d, Ensemble([base()], mode="base_only")) / len(d), 4)
    out["base_plus_mixer_only_bpb"] = round(run_bits(d, Ensemble([base()])) / len(d), 4)
    out["plus_above_left_bpb"] = round(run_bits(d, Ensemble([base(), above()])) / len(d), 4)
    out["plus_noise_control_bpb"] = round(run_bits(d, Ensemble([base(), noise()])) / len(d), 4)
    return out


def run():
    return save("e06_euclid", dict(a_translation_repeat=part_a(), b_row_translation=part_b()))


if __name__ == "__main__":
    print(run())
