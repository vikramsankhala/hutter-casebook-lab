"""E05 - Energy-based models (Case 9).

Energy E(x|ctx) = -sum_k W_k[h_k(ctx), x];  p(x|ctx) = exp(-E)/Z  (Boltzmann / log-linear).
Learned online by AdaGrad on log-loss; compared with count-based backoff, and with a
Born-rule (amplitude) parameterisation p = a^2 / sum a^2 as a 'quantum-inspired' control.
"""
import math
import numpy as np
from hcl import models
from experiments.common import get_data, save

N = 40_000


class LogLinear:
    def __init__(self, orders=(0, 1, 2), bits=15, lr=0.6, born=False):
        self.orders, self.mask, self.lr, self.born = orders, (1 << bits) - 1, lr, born
        init = 1.0 if born else 0.0
        self.W = {k: np.full((1 << bits, 256), init / len(orders) if born else 0.0) for k in orders}
        self.G = {k: np.full(1 << bits, 1e-3) for k in orders}
        self.hist = b""; self.rows = None; self.p = None; self.a = None

    def _rows(self):
        return {k: ((int.from_bytes(self.hist[-k:] if k else b"", 'big') * 2654435761 + 97 * k) >> 3) & self.mask for k in self.orders}

    def probs(self):
        self.rows = self._rows()
        s = sum(self.W[k][r] for k, r in self.rows.items())
        if self.born:
            self.a = s + 1e-3
            p = self.a ** 2; p /= p.sum()
        else:
            z = np.exp(s - s.max()); p = z / z.sum()
        self.p = p
        return p

    def update(self, b):
        p = self.p
        if self.born:
            a = self.a; S = (a * a).sum()
            g = 2 * a / S; g[b] -= 2 / a[b]          # d(-log p_b)/da
        else:
            g = p.copy(); g[b] -= 1                  # d(-log p_b)/ds
        for k, r in self.rows.items():
            self.G[k][r] += float(g @ g)
            self.W[k][r] -= self.lr * g / math.sqrt(self.G[k][r])
        self.hist = (self.hist + bytes([b]))[-8:]


def run():
    d = get_data(N)
    res = {"bytes": len(d)}
    ref = {}
    for k in (1, 2, 3):
        ref["count_backoff_order%d" % k] = round(models.cost_bits(d, models.Backoff(order=k)) / len(d), 4)
    res["reference_bpb"] = ref
    sweep = {}
    for name, kw, lrs in [("boltzmann_orders012", dict(orders=(0, 1, 2)), (1.2, 2.5, 5.0, 10.0)),
                          ("boltzmann_orders0123", dict(orders=(0, 1, 2, 3)), (1.2, 2.5, 5.0, 10.0)),
                          ("born_amplitude_orders012", dict(orders=(0, 1, 2), born=True), (1.0, 2.0, 4.0, 8.0))]:
        sweep[name] = {str(lr): round(models.cost_bits(d, LogLinear(lr=lr, **kw)) / len(d), 4) for lr in lrs}
        res[name + "_best_bpb"] = min(sweep[name].values())
    res["learning_rate_sweep_bpb"] = sweep
    return save("e05_energy", res)


if __name__ == "__main__":
    print(run())
