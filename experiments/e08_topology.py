"""E08 - Topological approaches (Case 12).

(a) Planarity of markup: '{{ }}' and '[[ ]]' pairs never cross -> a PUSHDOWN (stack) context.
    Count of planar pairings = Catalan numbers: log2 C_n ~ 2n bits is the *structure* entropy.
    Test: stack-aware expert vs capacity-matched noise control; attribute the gain to closing delimiters.
(b) Link-graph topology: Betti numbers b0, b1 and persistent H0 of the co-link graph;
    a positive control (community-structured links) and the real corpus links.
"""
import math, re, random
from collections import Counter, defaultdict
import numpy as np
from hcl.ctx import CtxExpert, Ensemble, order_keys, State
from experiments.common import get_data, save

N_A = 40_000


def catalan_log2(n):
    return math.log2(math.comb(2 * n, n) / (n + 1))


def per_byte_cost(data, ens):
    c = np.empty(len(data))
    for i, b in enumerate(data):
        p = ens.probs(); c[i] = -math.log2(p[b]); ens.update(b)
    return c


def part_a():
    d = get_data(N_A)
    base = lambda: CtxExpert("order3", order_keys(3))
    top = lambda st: ord(st.stack[-1]) if st.stack else 0
    stack = lambda: CtxExpert("stack", lambda st: [b"", bytes([top(st), min(len(st.stack), 3)]),
                                                  bytes([top(st), min(len(st.stack), 3), st.left()]),
                                                  bytes([top(st), min(len(st.stack), 3), st.left(), st.left(2)])])
    noise = lambda: CtxExpert("noise", lambda st: [b"", bytes([len(st.cur) % 7]), bytes([len(st.cur) % 7, st.left()]),
                                                  bytes([len(st.cur) % 7, st.left(), st.left(2)])])
    out = {}
    cost = {}
    for name, ex in (("mixer_only", [base()]), ("plus_stack", [base(), stack()]), ("plus_noise_control", [base(), noise()])):
        cost[name] = per_byte_cost(d, Ensemble(ex))
        out[name + "_bpb"] = round(float(cost[name].mean()), 4)
    arr = np.frombuffer(d, dtype=np.uint8)
    closer = (arr == ord("}")) | (arr == ord("]"))
    for name in cost:
        out[name + "_bits_on_closing_delimiters"] = round(float(cost[name][closer].sum()), 1)
        out[name + "_bits_on_everything_else"] = round(float(cost[name][~closer].sum()), 1)
    out["closing_delimiter_bytes"] = int(closer.sum())
    # structure entropy of planar pairings (information the stack makes free)
    pairs = d.count(b"{{") + d.count(b"[[")
    out["catalan_log2_of_pairs"] = round(catalan_log2(pairs), 1)
    out["pairs_in_slice"] = pairs
    # check parser: stack ends balanced at article ends in the full corpus
    st = State()
    depth_max = 0
    for b in get_data(None):
        st.update(b); depth_max = max(depth_max, len(st.stack))
    out["max_nesting_depth_observed"] = depth_max
    out["stack_empty_at_end_of_corpus"] = (len(st.stack) == 0)
    return out


# ------------------------------------------------------------------ link graph
class DSU:
    def __init__(self): self.p = {}
    def f(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def u(self, a, b):
        a, b = self.f(a), self.f(b)
        if a == b: return False
        self.p[a] = b; return True


def betti(edges):
    dsu = DSU(); nodes = set(); E = 0
    for a, b in edges:
        nodes.add(a); nodes.add(b); E += 1; dsu.u(a, b)
    comps = len({dsu.f(x) for x in nodes})
    return dict(V=len(nodes), E=E, b0=comps, b1=E - len(nodes) + comps)


def components_at(co, thr):
    dsu = DSU()
    for (a, b), w in co.items():
        dsu.f(a); dsu.f(b)
        if w >= thr: dsu.u(a, b)
    return len({dsu.f(x) for x in list(dsu.p)})


def links_from_corpus(data):
    arts = data.split(b"<page>")
    pat = re.compile(rb"\[\[([^\]|#:]+)(?:\|[^\]]*)?\]\]")
    return [[m.decode() for m in pat.findall(a)] for a in arts if a.strip()]


def structured_links(seed=3, comm=8, size=15, arts=400, per=8, p_in=0.85):
    rnd = random.Random(seed); out = []
    for _ in range(arts):
        c = rnd.randrange(comm); lst = []
        for _ in range(per):
            cc = c if rnd.random() < p_in else rnd.randrange(comm)
            lst.append("p%d_%d" % (cc, rnd.randrange(size)))
        out.append(lst)
    return out


def link_cost(arts):
    """Adaptive coding of link targets: order-0 vs previous-link (bigram, backoff to order-0)."""
    vocab = {t for a in arts for t in a}; V = len(vocab)
    c0 = Counter(); n0 = 0; big = defaultdict(Counter); nb = Counter()
    b0 = b1 = 0.0; beta = 2.0
    for a in arts:
        prev = None
        for t in a:
            p0 = (c0[t] + 1) / (n0 + V)
            b0 -= math.log2(p0)
            if prev is None: p1 = p0
            else: p1 = (big[prev][t] + beta * p0) / (nb[prev] + beta)
            b1 -= math.log2(p1)
            c0[t] += 1; n0 += 1
            if prev is not None: big[prev][t] += 1; nb[prev] += 1
            prev = t
    n = sum(len(a) for a in arts)
    return dict(links=n, vocab=V, uniform_bits_per_link=round(math.log2(V), 3), order0_bits_per_link=round(b0 / n, 3),
                previous_link_bits_per_link=round(b1 / n, 3))


def link_cost_cluster(arts, refresh=40, beta=2.0, oracle=False):
    """Topology-aware predictor: context = persistent-H0 cluster of the previous link.
    Clusters are recomputed from PAST articles only (causal) every `refresh` articles."""
    vocab = sorted({t for a in arts for t in a}); V = len(vocab)
    c0 = Counter(); n0 = 0; co = Counter(); label = {}
    cc = defaultdict(Counter); nc = Counter(); bits = 0.0; n = 0

    def recluster():
        best = None
        ws = sorted({w for w in co.values()})
        for thr in ws:
            dsu = DSU()
            for (a, b), w in co.items():
                dsu.f(a); dsu.f(b)
                if w >= thr: dsu.u(a, b)
            roots = Counter(dsu.f(x) for x in list(dsu.p))
            k, big = len(roots), max(roots.values())
            if 4 <= k <= 30 and big <= 0.5 * len(dsu.p):
                best = {x: dsu.f(x) for x in dsu.p}; break
        return best

    for ai, a in enumerate(arts):
        if oracle:
            label = {t: t.split('_')[0] for t in vocab}
        elif ai and ai % refresh == 0 and co:
            lab = recluster()
            if lab: label = lab
        prev = None
        for t in a:
            p0 = (c0[t] + 1) / (n0 + V)
            if prev is not None and prev in label:
                ctx = label[prev]
                p = (cc[ctx][t] + beta * p0) / (nc[ctx] + beta)
            else:
                p = p0
            bits -= math.log2(p); n += 1
            c0[t] += 1; n0 += 1
            if prev is not None and prev in label:
                cc[label[prev]][t] += 1; nc[label[prev]] += 1
            prev = t
        s = sorted(set(a))
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                co[(s[i], s[j])] += 1
    return round(bits / n, 3)


def analyse(arts):
    co = Counter(); edges = set()
    for a in arts:
        s = sorted(set(a))
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                co[(s[i], s[j])] += 1; edges.add((s[i], s[j]))
    maxw = max(co.values()) if co else 0
    ths = sorted({1, 2, 3, 5, 8, 13, 21, max(1, maxw // 2), maxw})
    return dict(betti=betti(edges), components_by_edge_weight_threshold={str(t): components_at(co, t) for t in ths},
                coding=dict(link_cost(arts), persistent_cluster_context_bits_per_link=link_cost_cluster(arts), oracle_true_cluster_bits_per_link=(link_cost_cluster(arts, oracle=True) if arts and '_' in arts[0][0] else None)))


def part_b():
    real = links_from_corpus(get_data(None))
    return dict(corpus_links=analyse(real), structured_positive_control=analyse(structured_links()))


def run():
    return save("e08_topology", dict(a_planar_markup=part_a(), b_link_graph=part_b()))


if __name__ == "__main__":
    print(run())
