"""E03 - Copy-with-edits as shortest path (Case 7).

Static surrogate cost model (bits), identical for every variant:
  tag 1 bit; literal = order-0 surprisal; copy = Elias-delta(offset) + Elias-gamma(len-MIN+1);
  edit (substitution) = Elias-gamma(gap+1) + order-0 surprisal of replacement byte.
Variants: greedy longest exact copy | DP exact | DP with up to K substitutions.
"""
import math
from collections import Counter, defaultdict
from experiments.common import get_data, save

N, MIN, ANCHOR, CAND, CAP, K = 60_000, 6, 5, 24, 320, 3


def gamma(n): return 2 * int(math.log2(n)) + 1
def delta(n):
    l = int(math.log2(n)); return l + 2 * int(math.log2(l + 1)) + 1


def build(d, allow_edits):
    n = len(d); c = Counter(d)
    lit = [-math.log2(c[b] / n) for b in d]
    idx = defaultdict(list); edges = [[] for _ in range(n)]
    for i in range(n - ANCHOR):
        key = d[i:i + ANCHOR]
        for j in idx[key][-CAND:]:
            # exact
            L = 0
            while i + L < n and L < CAP and d[j + L] == d[i + L]: L += 1
            off = i - j
            if L >= MIN:
                edges[i].append((L, 1 + delta(off) + gamma(L - MIN + 1), 0, ()))
            if allow_edits and L >= MIN:
                p, e, bits, gap, last, sp = L, 0, 0.0, L, L, []
                while e < K and i + p < n and j + p < n and p < CAP:
                    bits += gamma(gap + 1) + lit[i + p]          # substitution at donor offset p
                    sp.append(i + p); e += 1; p += 1
                    q = p
                    while i + q < n and q < CAP and d[j + q] == d[i + q]: q += 1
                    gap = q - p; p = q
                    if p - L >= 2:
                        edges[i].append((p, 1 + delta(off) + gamma(p - MIN + 1) + bits, e, tuple(sp)))
        idx[d[i:i + ANCHOR]].append(i)
    return lit, edges


def solve(d, allow_edits=True, greedy=False):
    n = len(d); lit, edges = build(d, allow_edits)
    if greedy:
        i, bits, ops = 0, 0.0, Counter()
        while i < n:
            ex = [e for e in edges[i] if e[2] == 0]
            if ex:
                L, b, _, _s = max(ex, key=lambda e: e[0]); bits += b; i += L; ops["copy"] += 1
            else:
                bits += 1 + lit[i]; i += 1; ops["lit"] += 1
        return bits, ops
    cost = [0.0] * (n + 1); choice = [None] * n
    for i in range(n - 1, -1, -1):
        best, ch = 1 + lit[i] + cost[i + 1], ("lit", 1, 0, ())
        for L, b, e, sp in edges[i]:
            if i + L <= n and b + cost[i + L] < best:
                best, ch = b + cost[i + L], ("copy" if e == 0 else "edit", L, e, sp)
        cost[i], choice[i] = best, ch
    i, ops, subs, digit = 0, Counter(), 0, 0
    while i < n:
        kind, L, e, sp = choice[i]; ops[kind] += 1; subs += e; i += L
        digit += sum(1 for q in sp if chr(d[q]).isdigit())
    ops["substitutions"] = subs; ops["substituted_bytes_that_are_digits"] = digit
    return cost[0], ops


def run():
    d = get_data(N)
    res = {"bytes": len(d), "unit": "bits under static surrogate cost model (not a byte-exact archive)"}
    gb, go = solve(d, False, greedy=True)
    eb, eo = solve(d, False)
    xb, xo = solve(d, True)
    res["greedy_exact"] = dict(bytes=round(gb / 8), ops=dict(go))
    res["dp_exact"] = dict(bytes=round(eb / 8), ops=dict(eo))
    res["dp_with_edits"] = dict(bytes=round(xb / 8), ops=dict(xo))
    res["edit_gain_bytes"] = round((eb - xb) / 8)
    res["dp_gain_over_greedy_bytes"] = round((gb - eb) / 8)
    # control: same generator with NO near-duplicate revisions -> edits should matter less
    from hcl import corpus
    c = corpus.generate(revision_rate=0.0)[:N]
    ce, _ = solve(c, False); cx, co = solve(c, True)
    res["control_no_revisions"] = dict(dp_exact_bytes=round(ce / 8), dp_edits_bytes=round(cx / 8),
                                       edit_gain_bytes=round((ce - cx) / 8), substitutions=co["substitutions"])
    return save("e03_copy_edits", res)


if __name__ == "__main__":
    print(run())
