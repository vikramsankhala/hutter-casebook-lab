"""E09 - Quantum-topology approaches (Case 12).

(a) Kauffman bracket by STATE SUM (a partition function over 2^c smoothings), verified on the
    trefoil and figure-eight knots against the textbook Laurent polynomials.
(b) At A = exp(2 pi i / 5) the loop weight d is the golden ratio (Fibonacci-anyon point): weights are
    complex AMPLITUDES, not probabilities - only |amplitude|^2 (Born rule) is a distribution.
(c) Planar (Temperley-Lieb / Dyck) structure in a SOURCE: free-group words with nested cancelling pairs.
    Generic order-k context models vs a stack-aware model vs the TRUE source entropy.
"""
import cmath, math, random
from collections import defaultdict
import numpy as np
from hcl import models
from experiments.common import save

TREFOIL = [(1, 4, 2, 5), (3, 6, 4, 1), (5, 2, 6, 3)]
FIG8 = [(4, 2, 5, 1), (8, 6, 1, 5), (6, 3, 7, 4), (2, 7, 3, 8)]


def pmul(p, q):
    r = defaultdict(int)
    for a, x in p.items():
        for b, y in q.items():
            r[a + b] += x * y
    return {k: v for k, v in r.items() if v}


def padd(p, q):
    r = defaultdict(int, p)
    for k, v in q.items(): r[k] += v
    return {k: v for k, v in r.items() if v}


def ppow(p, n):
    r = {0: 1}
    for _ in range(n): r = pmul(r, p)
    return r


def bracket(pd):
    """Laurent polynomial in A (dict exponent -> int). <unknot> = 1, d = -A^2 - A^-2."""
    d = {2: -1, -2: -1}
    total = {}
    c = len(pd)
    labels = sorted({x for cr in pd for x in cr})
    for s in range(1 << c):
        parent = {x: x for x in labels}
        def f(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        a_count = 0
        for i, (a, b, cc, dd) in enumerate(pd):
            if (s >> i) & 1:      # A-smoothing joins (a,d),(b,c)
                a_count += 1; pairs = ((a, dd), (b, cc))
            else:                 # B-smoothing joins (a,b),(c,d)
                pairs = ((a, b), (cc, dd))
            for x, y in pairs: parent[f(x)] = f(y)
        loops = len({f(x) for x in labels})
        term = ppow(d, loops - 1)
        term = {k + (a_count - (c - a_count)): v for k, v in term.items()}
        total = padd(total, term)
    return dict(sorted(total.items()))


def poly_str(p):
    return " + ".join("%d*A^%d" % (v, k) for k, v in sorted(p.items()))


def evaluate(p, A):
    return sum(v * A ** k for k, v in p.items())


def part_a():
    tre, fig = bracket(TREFOIL), bracket(FIG8)
    mirror = lambda p: {-k: v for k, v in p.items()}
    exp_tre = {-7: 1, -3: -1, 5: -1}
    exp_fig = {-8: 1, -4: -1, 0: 1, 4: -1, 8: 1}
    return dict(trefoil=poly_str(tre), trefoil_matches_textbook_or_mirror=(tre == exp_tre or tre == mirror(exp_tre)),
                figure_eight=poly_str(fig), figure_eight_matches_textbook=(fig == exp_fig),
                states_summed=dict(trefoil=1 << len(TREFOIL), figure_eight=1 << len(FIG8)))


def part_b():
    A = cmath.exp(2j * math.pi / 5)
    d = -A ** 2 - A ** -2
    tre = bracket(TREFOIL)
    val = evaluate(tre, A)
    return dict(loop_weight_d=round(d.real, 6), golden_ratio=round((1 + 5 ** .5) / 2, 6),
                trefoil_bracket_at_A=[round(val.real, 6), round(val.imag, 6)],
                born_probability_weight=round(abs(val) ** 2, 6),
                note="state weights are complex amplitudes; only squared magnitudes form a distribution")


# ------------------------------------------------------- planar source (free group)
LET = b"aAbB"; INV = {ord("a"): ord("A"), ord("A"): ord("a"), ord("b"): ord("B"), ord("B"): ord("b")}


def gen_word(n, p_push=0.35, seed=5):
    rnd = random.Random(seed); out = bytearray(); stack = []; bits = 0.0
    while len(out) < n:
        if not stack or rnd.random() < p_push:
            x = LET[rnd.randrange(4)]; out.append(x); stack.append(INV[x])
            bits += (2.0 if not stack[:-1] else (-math.log2(p_push) + 2.0))
        else:
            out.append(stack.pop()); bits += -math.log2(1 - p_push)
    return bytes(out), bits


class StackModel:
    def __init__(self, beta=1.0):
        self.stack = []; self.t = {}; self.beta = beta; self.key = None
    def probs(self):
        top = self.stack[-1] if self.stack else 0
        self.key = (top, min(len(self.stack), 4))
        row = self.t.get(self.key)
        base = np.full(256, 1 / 256)
        if row is None: return base
        return (row[0] + self.beta * base) / (row[1] + self.beta)
    def update(self, b):
        row = self.t.setdefault(self.key, [np.zeros(256), 0.0]); row[0][b] += 1; row[1] += 1
        if self.stack and b == self.stack[-1]: self.stack.pop()
        else: self.stack.append(INV[b])


def part_c():
    d, true_bits = gen_word(30_000)
    out = dict(symbols=len(d), action_level_entropy_bpb=round(true_bits / len(d), 4),
               note="action entropy is an UPPER bound on symbol entropy: a symbol equal to the pending closer can arise from a pop or a push, so a symbol-level model may legitimately beat it")
    for k in (2, 4, 6, 8):
        out["generic_order%d_bpb" % k] = round(models.cost_bits(d, models.Backoff(order=k)) / len(d), 4)
    out["stack_aware_bpb"] = round(models.cost_bits(d, StackModel()) / len(d), 4)
    # normal-form view: free reduction leaves a short word; the nesting structure is the residue
    st = []
    for b in d:
        if st and st[-1] == INV[b]: st.pop()
        else: st.append(b)
    out["freely_reduced_length"] = len(st)
    return out


def run():
    return save("e09_quantum_topology", dict(a_kauffman_state_sum=part_a(), b_amplitudes=part_b(), c_planar_source=part_c()))


if __name__ == "__main__":
    print(run())
