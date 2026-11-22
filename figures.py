#!/usr/bin/env python3
"""Generate every figure in the casebook from results/all_results.json (Solinexta navy/teal/amber palette)."""
import json, os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Arc, Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "docs", "figs"); os.makedirs(OUT, exist_ok=True)
R = json.load(open(os.path.join(HERE, "results", "all_results.json")))
for f in ("/usr/share/fonts/truetype/crosextra/Carlito-Regular.ttf", "/usr/share/fonts/truetype/crosextra/Carlito-Bold.ttf"):
    if os.path.exists(f): fm.fontManager.addfont(f)
NAVY, TEAL, AMBER, INK, MUTED, TINT, LINE = "#1F3A5F", "#0E7C7B", "#E8A317", "#1B2533", "#556270", "#C9D7E8", "#D5DCE5"
plt.rcParams.update({"font.family": "Carlito", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": LINE, "grid.linewidth": .6, "figure.dpi": 200})


def save(name):
    plt.tight_layout(); plt.savefig(os.path.join(OUT, name + ".png"), dpi=200); plt.close()


# 1 arithmetic coding intervals
def interval():
    fig, ax = plt.subplots(figsize=(6.4, 2.2)); ax.grid(False)
    steps = [("start", 0, 1), ("A", 0, 3 / 4), ("AB", 9 / 16, 3 / 4), ("ABA", 9 / 16, 45 / 64)]
    for i, (lab, lo, hi) in enumerate(steps):
        y = -i
        ax.plot([0, 1], [y, y], color=LINE, lw=1)
        ax.add_patch(plt.Rectangle((lo, y - .2), hi - lo, .4, color=TEAL if i == 3 else NAVY, alpha=.9))
        ax.text(-0.02, y, lab, ha="right", va="center", color=INK)
        ax.text(hi + .01, y, "[%s, %s)" % (_f(lo), _f(hi)), va="center", color=MUTED, fontsize=9)
    ax.axvline(5 / 8, color=AMBER, lw=1.5); ax.text(5 / 8, .45, "code point 5/8", ha="center", color=INK, fontsize=9)
    ax.set_xlim(-.15, 1.35); ax.set_ylim(-3.6, .8); ax.set_yticks([]); ax.set_xticks([0, .25, .5, .75, 1])
    save("fig_interval")


def _f(x):
    from fractions import Fraction
    return str(Fraction(x).limit_denominator(1000))


# 2 baseline ladder
def ladder():
    e = R["e01_baselines"]
    labs = ["order 0", "order 1", "order 2", "order 3", "order 4"]
    v = [e["adaptive_order%d_bpb" % k] for k in range(5)]
    fig, ax = plt.subplots(figsize=(6.4, 3))
    ax.bar(labs, v, color=NAVY)
    for name, col in (("zlib", e["zlib9_bytes"]), ("bz2", e["bz2_bytes"]), ("xz", e["xz_bytes"])):
        ax.axhline(8 * col / e["bytes"], color=TEAL if name == "xz" else AMBER if name == "zlib" else MUTED, lw=1.2, ls="--")
        ax.text(4.45, 8 * col / e["bytes"], name, va="bottom", ha="right", fontsize=9, color=INK)
    for i, x in enumerate(v): ax.text(i, x + .05, "%.2f" % x, ha="center", fontsize=9)
    ax.set_ylabel("bits per byte (lower is better)"); ax.grid(axis="x", visible=False)
    save("fig_ladder")


# 3 grammar U curve
def grammar():
    c = R["e04_grammar"]["description_length_bytes_by_rules"]
    xs = sorted(int(k) for k in c); ys = [c[str(k)] for k in xs]
    fig, ax = plt.subplots(figsize=(6.4, 3))
    ax.plot(xs, ys, color=NAVY, marker="o", ms=4)
    b = R["e04_grammar"]["best_rule_count"]; ax.axvline(b, color=TEAL, ls="--"); ax.text(b + 30, max(ys) * .9, "best ≈ %d rules" % b, color=TEAL)
    ax.set_xlabel("rules in grammar"); ax.set_ylabel("description length (bytes)")
    save("fig_grammar")


# 4 net-of-control
def net():
    import scoreboard
    rows = scoreboard.rows()
    names = [r[0].replace("E06 ", "").replace("E08 ", "").replace("E10 ", "").replace("E12 ", "") for r in rows]
    cand = [scoreboard.pct(r[3], r[2]) for r in rows]; ctl = [scoreboard.pct(r[4], r[2]) for r in rows]
    y = np.arange(len(rows)); fig, ax = plt.subplots(figsize=(6.6, 3.3))
    ax.barh(y + .18, [-c for c in cand], .34, color=TEAL, label="candidate expert")
    ax.barh(y - .18, [-c for c in ctl], .34, color=AMBER, label="noise control (same capacity)")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8.5); ax.invert_yaxis()
    ax.set_xlabel("bits saved versus mixer-only baseline (%)"); ax.legend(frameon=False, fontsize=8.5, loc="lower right"); ax.grid(axis="y", visible=False)
    save("fig_net")


# 5 simplex geodesics
def simplex():
    fig, ax = plt.subplots(figsize=(4.6, 4.1)); ax.grid(False); ax.axis("off")
    V = np.array([[0, 0], [1, 0], [.5, math.sqrt(3) / 2]])
    ax.add_patch(Polygon(V, fill=False, ec=MUTED))
    def xy(p): return p @ V
    p = np.array([.7, .2, .1]); q = np.array([.1, .3, .6])
    ts = np.linspace(0, 1, 60)
    lin = np.array([(1 - t) * p + t * q for t in ts])
    geo = np.array([(p ** (1 - t)) * (q ** t) / ((p ** (1 - t)) * (q ** t)).sum() for t in ts])
    fr = []
    for t in ts:
        s = np.sqrt(p); u = np.sqrt(q); th = math.acos(float(s @ u))
        w = (math.sin((1 - t) * th) * s + math.sin(t * th) * u) / math.sin(th); fr.append(w ** 2)
    fr = np.array(fr)
    for arr, col, lab in ((lin, NAVY, "m-geodesic (linear mix)"), (geo, TEAL, "e-geodesic (geometric mix)"), (fr, AMBER, "Fisher–Rao geodesic")):
        pts = np.array([xy(a) for a in arr]); ax.plot(pts[:, 0], pts[:, 1], color=col, lw=2, label=lab)
    for pt, lab in ((p, "p"), (q, "q")):
        z = xy(pt); ax.scatter(*z, color=INK, zorder=5); ax.text(z[0] + .02, z[1] + .02, lab, fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0, 1.02))
    save("fig_simplex")


# 6 planar arcs
def arcs():
    fig, ax = plt.subplots(figsize=(6.4, 1.9)); ax.grid(False); ax.axis("off")
    seq = "{{ [[ ]] {{ }} }}".split()
    pos = {i: i for i in range(len(seq))}
    pairs, st = [], []
    for i, s in enumerate(seq):
        if s in ("{{", "[["): st.append(i)
        else: pairs.append((st.pop(), i))
    for i, s in enumerate(seq):
        ax.text(i, 0, s, ha="center", va="center", fontsize=13, color=NAVY, family="DejaVu Sans Mono")
    for a, b in pairs:
        ax.add_patch(Arc(((a + b) / 2, .25), b - a, (b - a) * .6, theta1=0, theta2=180, color=TEAL, lw=2))
    ax.set_xlim(-.7, len(seq) - .3); ax.set_ylim(-.3, 2.0)
    save("fig_arcs")


# 7 autocorrelation spectra
def spectrum():
    import experiments.e10_harmonic as e10
    from experiments.common import get_data
    fw = e10.fixed_width_stream(); a = e10.equality_autocorr(fw, 120); b = e10.equality_autocorr(get_data(60000), 120)
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=True)
    for k, (arr, title, col) in enumerate(((a, "fixed-width records", TEAL), (b, "River Archive II text", NAVY))):
        ax[k].bar(list(arr), list(arr.values()), color=col, width=1.0); ax[k].set_title(title, fontsize=10, color=INK); ax[k].set_xlabel("lag k")
    ax[0].set_ylabel("P(byte[i] = byte[i+k])")
    save("fig_spectrum")


# 8 temperature / sweeps
def sweeps():
    s = R["e07_infogeom"]["learning_rate_sweep_bpb"]
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    for name, col in (("geometric_sgd", TEAL), ("geometric_natural", AMBER)):
        xs = sorted(float(k) for k in s[name]); ys = [s[name][str(k) if str(k) in s[name] else repr(k)] if (str(k) in s[name] or repr(k) in s[name]) else None for k in xs]
        ax.plot(xs, ys, marker="o", color=col, label=name.replace("_", " "))
    ax.axhline(R["e07_infogeom"]["best_single_order3_bpb"], color=NAVY, ls="--", label="best single order-3")
    ax.set_xscale("log"); ax.set_xlabel("learning rate"); ax.set_ylabel("bits per byte"); ax.legend(frameon=False, fontsize=8.5)
    save("fig_sweeps")


if __name__ == "__main__":
    for f in (interval, ladder, grammar, net, simplex, arcs, spectrum, sweeps):
        f(); print("ok", f.__name__)
