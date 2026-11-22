"""E10 - Harmonic analysis / Langlands-flavoured approaches (Case 14).

Langlands in one student-sized sentence: objects that look unrelated (arithmetic data, automorphic forms,
Galois representations) share one generator; harmonic analysis on groups (characters, Fourier transforms)
is the dictionary between descriptions.  Two testable consequences for compression:
 (a) PERIODICITY = a character of Z/P.  Detect P from the spectrum, then use column-mod-P context.
 (b) GENERATOR-AWARE coding: a table produced by a short rule (Ramanujan tau from the eta-product, primes)
     costs only the rule, IF the decoder's size is counted - the Hutter accounting makes break-even explicit.
"""
import lzma, math, subprocess, sys, textwrap
import numpy as np
from hcl.ctx import CtxExpert, Ensemble, order_keys, run_bits
from hcl import corpus, models
from experiments.common import get_data, save


def equality_autocorr(d, maxlag=200):
    a = np.frombuffer(d, dtype=np.uint8)
    return {k: float((a[:-k] == a[k:]).mean()) for k in range(2, maxlag)}


def fixed_width_stream(n_rec=900, seed=2):
    rng = np.random.default_rng(seed)
    names = [corpus._word(rng, 2, 3).ljust(9) for _ in range(40)]
    rows = []
    for i in range(n_rec):
        rows.append("%06d|%s|%9.2f|%s" % (1000 + 3 * i, names[int(rng.integers(0, 40))],
                                         50 + i * 1.5 + float(rng.normal(0, 3)), "YN"[int(rng.integers(0, 2))]))
    return "".join(rows).encode()               # no separators: pure fixed-width


def part_a():
    fw = fixed_width_stream()
    ac = equality_autocorr(fw)
    P = max(ac, key=ac.get)
    synth = get_data(60_000)
    ac2 = equality_autocorr(synth)
    P2 = max(ac2, key=ac2.get)
    d = fw[:25_000]
    base = lambda: CtxExpert("order3", order_keys(3))
    per = lambda: CtxExpert("col_mod_P", lambda st: [b"", bytes([st.n % P]), bytes([st.n % P, st.lag(P)]),
                                                  bytes([st.n % P, st.lag(P), st.left()])])
    noise = lambda: CtxExpert("noise", lambda st: [b"", bytes([st.n % 7]), bytes([st.n % 7, st.left()])])
    res = dict(fixed_width_true_period=len(fw) // 900, detected_period=P, peak_value=round(ac[P], 3),
               peak_value_at_2P=round(ac[2 * P], 3) if 2 * P in ac else None,
               river_corpus_peak_lag=P2, river_corpus_peak_value=round(ac2[P2], 3))
    res["fixed_width_mixer_only_bpb"] = round(run_bits(d, Ensemble([base()])) / len(d), 4)
    res["fixed_width_plus_period_bpb"] = round(run_bits(d, Ensemble([base(), per()])) / len(d), 4)
    res["fixed_width_plus_noise_control_bpb"] = round(run_bits(d, Ensemble([base(), noise()])) / len(d), 4)
    return res


GEN = ("N=%d;c=[1]+[0]*N\nfor n in range(1,N+1):\n for _ in range(24):\n  for i in range(N,n-1,-1):c[i]-=c[i-n]\n"
       "print(*c[:N])")


def tau_table(N):
    c = [1] + [0] * N
    for n in range(1, N + 1):
        for _ in range(24):
            for i in range(N, n - 1, -1):
                c[i] -= c[i - n]
    return c[:N]            # c[k] = tau(k+1)


def part_b():
    out = {}
    tau = tau_table(400)
    # Hecke multiplicativity: tau(mn) = tau(m) tau(n) for gcd(m,n)=1
    bad, checked = 0, 0
    for m in range(2, 20):
        for n in range(2, 20):
            if math.gcd(m, n) == 1 and m * n <= 400:
                checked += 1; bad += tau[m * n - 1] != tau[m - 1] * tau[n - 1]
    out["hecke_multiplicativity_checked"] = checked; out["hecke_violations"] = bad
    out["tau_1_to_5"] = tau[:5]
    rows = []
    for N in (50, 100, 200, 400):
        text = "\n".join(str(x) for x in tau_table(N)).encode()
        prog = (GEN % N).encode()
        ran = subprocess.run([sys.executable, "-I", "-c", GEN % N], capture_output=True, text=True, timeout=60).stdout.split()
        assert [int(x) for x in ran] == tau_table(N)           # the generator really reproduces the table
        rows.append(dict(entries=N, table_bytes=len(text), xz_bytes=len(lzma.compress(text, preset=9 | lzma.PRESET_EXTREME)),
                         generator_bytes=len(prog)))
    out["tau_table_vs_generator"] = rows
    # primes page inside the synthetic corpus
    data = corpus.default_corpus()
    start = data.find(b"== List of primes ==")
    end = data.find(b"</text>", start)
    tr = models.cost_trace(data, models.Backoff(order=3))
    page_bits = float(tr[start:end].sum())
    gen = b"p=[];k=2\nwhile len(p)<200:\n if all(k%q for q in p if q*q<=k):p+=k,\n k+=1\nprint(*p)"
    out["primes_page"] = dict(page_bytes=end - start, order3_model_cost_bytes=round(page_bits / 8),
                              generator_bytes=len(gen), share_of_corpus_cost_pct=round(100 * page_bits / tr.sum(), 2),
                              note="a decoder that embeds the generator pays len(generator) once; net gain = model cost - generator - glue")
    return out


def run():
    return save("e10_harmonic", dict(a_period_character=part_a(), b_generator_aware=part_b()))


if __name__ == "__main__":
    print(run())
