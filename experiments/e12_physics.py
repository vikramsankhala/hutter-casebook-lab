"""E12 - Theoretical-physics approaches (Case 16).

(a) Statistical mechanics: log-loss is a FREE ENERGY; temperature scaling p ~ p^(1/T)/Z.  Fit T on the first half,
    evaluate on the second.
(b) Renormalisation group: coarse-grain the context (word level, character-class blocks) and add as experts,
    with a capacity-matched noise control.
(c) Zipf law: pipeline sanity check (the generator PLANTS exponent 1, so the fit must recover ~1).
(d) Landauer bound for decoding enwik9 versus a stated machine budget (assumptions labelled).
(e) Ergodic theory: Ornstein-Weiss recurrence estimator of the entropy rate vs the model's coding rate.
"""
import math, re
from collections import Counter, defaultdict
import numpy as np
from hcl import models
from hcl.ctx import CtxExpert, Ensemble, order_keys, run_bits
from experiments.common import get_data, save


def part_a(n=30_000):
    d = get_data(n)
    m = models.Backoff(order=3)
    P = np.empty((len(d), 256), dtype=np.float32)
    for i, b in enumerate(d):
        P[i] = m.probs(); m.update(b)
    y = np.frombuffer(d, dtype=np.uint8)

    def cost(T, lo, hi):
        lp = np.log(P[lo:hi].astype(np.float64)) / T
        lz = np.log(np.exp(lp - lp.max(1, keepdims=True)).sum(1)) + lp.max(1)
        return float(-(lp[np.arange(hi - lo), y[lo:hi]] - lz).sum())
    h = len(d) // 2
    grid = [0.7, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.6, 2.0]
    fit = {T: cost(T, 0, h) for T in grid}
    T_star = min(fit, key=fit.get)
    base2 = cost(1.0, h, len(d)) / math.log(2); tuned2 = cost(T_star, h, len(d)) / math.log(2)
    return dict(grid=grid, best_T_on_first_half=T_star, second_half_bpb_T1=round(base2 / (len(d) - h), 4),
                second_half_bpb_tuned=round(tuned2 / (len(d) - h), 4))


def part_b(n=30_000):
    d = get_data(n)
    base = lambda: CtxExpert("order3", order_keys(3))
    word = lambda: CtxExpert("word", lambda st: [b"", st.word, st.pword + b"|" + st.word,
                                                st.ppword + b"|" + st.pword + b"|" + st.word])
    cls = lambda b: 1 if 97 <= b <= 122 else 2 if 65 <= b <= 90 else 3 if 48 <= b <= 57 else 4 if b == 32 else 5
    block = lambda: CtxExpert("char_class_block", lambda st: [bytes(cls(x) for x in st.hist[-3:]),
                                                           bytes(cls(x) for x in st.hist[-6:]),
                                                           bytes(cls(x) for x in st.hist[-6:]) + st.hist[-1:]])
    noise = lambda: CtxExpert("noise", lambda st: [b"", bytes([st.n % 7]), bytes([st.n % 7, st.left()])])
    out = {}
    for name, ex in (("mixer_only", [base()]), ("plus_word_scale", [base(), word()]),
                     ("plus_class_block_scale", [base(), block()]), ("plus_both_scales", [base(), word(), block()]),
                     ("plus_noise_control", [base(), noise()])):
        out[name + "_bpb"] = round(run_bits(d, Ensemble(ex)) / len(d), 4)
    return out


def part_c():
    d = get_data(None).lower()
    words = Counter(re.findall(rb"[a-z]{3,}", d))
    f = np.array(sorted(words.values(), reverse=True)[:150], dtype=float)
    r = np.arange(1, len(f) + 1)
    slope = np.polyfit(np.log(r), np.log(f), 1)[0]
    return dict(fitted_exponent=round(-float(slope), 3), planted_exponent=1.0, vocabulary=len(words))


def part_d():
    k, T = 1.380649e-23, 300.0
    per_bit = k * T * math.log(2)
    bits = 8e9
    energy_floor = per_bit * bits
    machine = 100.0 * 50 * 3600       # ASSUMPTION: 100 W for the 50 h limit shown on the prize page
    return dict(landauer_J_per_bit_at_300K=per_bit, enwik9_bits=bits, landauer_floor_J=energy_floor,
                assumed_machine_J=machine, ratio=machine / energy_floor, assumptions="100 W average draw, 50 h, 300 K")


def ow_estimate(d):
    n = len(d); idx = defaultdict(list); L = []
    for i in range(n):
        best = 0
        if i >= 4:
            for j in idx[d[i:i + 4]][-32:]:
                k = 0
                while i + k < n and k < 400 and d[j + k] == d[i + k]: k += 1
                best = max(best, k)
        if i >= 1000: L.append((best + 1) / math.log2(i))
        if i + 4 <= n: idx[d[i:i + 4]].append(i)
    return 1.0 / float(np.mean(L))


def part_e(n=30_000):
    d = get_data(n); n = len(d)
    rng = np.random.default_rng(0)
    iid = bytes(rng.integers(0, 4, size=n, dtype=np.uint8) + 97)          # H = 2 bits exactly
    from hcl import corpus
    norev = corpus.generate(revision_rate=0.0)[:n]
    validation = dict(iid_4_symbol_true_2_bits=round(ow_estimate(iid), 4),
                      corpus_without_revisions=round(ow_estimate(norev), 4),
                      order4_on_corpus_without_revisions=round(models.cost_bits(norev, models.Backoff(order=4)) / n, 4))
    h_est = ow_estimate(d)
    cm = models.cost_bits(d, models.Backoff(order=4)) / n
    return dict(ornstein_weiss_bits_per_byte=round(h_est, 4), order4_model_bpb=round(cm, 4),
                validation=validation,
                note="the recurrence estimator converges slowly and is biased on short samples; treat as an order-of-magnitude cross-check")


def run():
    return save("e12_physics", dict(a_temperature=part_a(), b_renormalisation=part_b(), c_zipf=part_c(),
                                   d_landauer=part_d(), e_recurrence_entropy_rate=part_e()))


if __name__ == "__main__":
    print(run())
