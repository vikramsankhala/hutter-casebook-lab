"""E07 - Differential / information geometry of mixing (Case 9).

Space of distributions = probability simplex with Fisher-Rao metric.
  m-geodesic (linear mixture)       q = sum w_k p_k
  e-geodesic (geometric mixture)    q ∝ prod p_k^{w_k}   (= logistic mixing in PAQ-family coders)
Facts verified numerically: linear mix minimises sum_i KL(p_i || q); geometric mix minimises sum_i KL(q || p_i).
Then compared as online mixers over order-0..3 experts, with plain SGD vs NATURAL gradient (Fisher preconditioned).
"""
import math
import numpy as np
from hcl import models
from experiments.common import get_data, save

N = 30_000


def kl(p, q):
    m = p > 0
    return float((p[m] * np.log2(p[m] / q[m])).sum())


def hellinger_angle(p, q):
    return 2 * math.acos(min(1.0, float(np.sqrt(p * q).sum())))     # Fisher-Rao geodesic distance


def verify_projection_facts(seed=0, trials=200):
    rng = np.random.default_rng(seed); ok_m = ok_e = 0
    for _ in range(trials):
        P = rng.dirichlet(np.ones(12) * 0.5, size=3)
        lin = P.mean(0)
        geo = np.exp(np.log(P).mean(0)); geo /= geo.sum()
        for q, tag in ((lin, "m"), (geo, "e")):
            pass
        f_m = lambda q: sum(kl(p, q) for p in P)        # forward KL
        f_e = lambda q: sum(kl(q, p) for p in P)        # reverse KL
        pert = lin * np.exp(rng.normal(0, .05, 12)); pert /= pert.sum()
        ok_m += f_m(lin) <= f_m(pert) and f_m(lin) <= f_m(geo) + 1e-12
        pert = geo * np.exp(rng.normal(0, .05, 12)); pert /= pert.sum()
        ok_e += f_e(geo) <= f_e(pert) and f_e(geo) <= f_e(lin) + 1e-12
    return dict(trials=trials, linear_minimises_forward_KL=ok_m, geometric_minimises_reverse_KL=ok_e)


class Mixer:
    def __init__(self, mode, lr=0.02, orders=(0, 1, 2, 3)):
        self.b = models.Backoff(orders=list(orders)); self.mode = mode; self.K = len(orders)
        self.w = np.zeros(self.K); self.w[-1] = 1.0
        if mode == "linear_bayes":
            self.w = np.full(self.K, 1 / self.K)
        self.lr = lr

    def probs(self):
        self.E = self.b.experts(); P = np.array(self.E)
        if self.mode == "linear_bayes":
            self.p = self.w @ P
        else:
            self.L = np.log(P)
            s = self.w @ self.L; z = np.exp(s - s.max()); self.p = z / z.sum()
            self.p = 0.999999 * self.p + 0.000001 / 256   # numerical floor, same for every geometric variant
        return self.p

    def update(self, byte):
        if self.mode == "linear_bayes":
            lik = np.array([e[byte] for e in self.E]); self.w = self.w * lik; self.w /= self.w.sum()
        else:
            g = -self.L[:, byte] + self.L @ self.p                 # d(-log p_x)/dw
            if self.mode == "geometric_natural":
                m = self.L @ self.p
                F = (self.L * self.p) @ self.L.T - np.outer(m, m)  # Fisher information in w
                g = np.linalg.solve(F + 1e-3 * np.eye(self.K), g)
            self.w -= self.lr * np.clip(g, -5, 5)
        self.b.update(byte)


def run():
    d = get_data(N)
    res = {"bytes": len(d), "projection_facts": verify_projection_facts()}
    res["best_single_order3_bpb"] = round(models.cost_bits(d, models.Backoff(order=3)) / len(d), 4)
    sweep = {}
    for mode, lrs in (("geometric_sgd", (0.001, 0.002, 0.005, 0.02)), ("geometric_natural", (0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.02))):
        sweep[mode] = {}
        for lr in lrs:
            m = Mixer(mode, lr=lr)
            sweep[mode][str(lr)] = round(models.cost_bits(d, m) / len(d), 4)
    res["learning_rate_sweep_bpb"] = sweep
    m = Mixer("linear_bayes"); res["linear_bayes_bpb"] = round(models.cost_bits(d, m) / len(d), 4)
    res["linear_bayes_final_weights"] = [round(float(x), 3) for x in m.w]
    res["geometric_sgd_best_bpb"] = min(sweep["geometric_sgd"].values())
    res["geometric_natural_best_bpb"] = min(sweep["geometric_natural"].values())
    a = np.array([.6, .3, .1]); b = np.array([.2, .2, .6])
    res["fisher_rao_distance_example"] = round(hellinger_angle(a, b), 4)
    return save("e07_infogeom", res)


if __name__ == "__main__":
    print(run())
