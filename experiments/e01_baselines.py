"""E01 - What does an honest baseline look like? (Case 1-2)"""
import zlib, bz2, lzma
from hcl import models, coders
from experiments.common import get_data, save, order0_bits

N = 100_000


def run():
    d = get_data(N)
    res = {"bytes": len(d), "corpus": "synthetic River Archive II unless HCL_ENWIK is set"}
    res["zlib9_bytes"] = len(zlib.compress(d, 9))
    res["bz2_bytes"] = len(bz2.compress(d, 9))
    res["xz_bytes"] = len(lzma.compress(d, preset=9 | lzma.PRESET_EXTREME))
    res["order0_empirical_bpb"] = round(order0_bits(d) / len(d), 4)
    for k in range(0, 5):
        b = models.cost_bits(d, models.Backoff(order=k))
        res["adaptive_order%d_bpb" % k] = round(b / len(d), 4)
        res["adaptive_order%d_bytes" % k] = round(b / 8)
    # real coder vs ideal at order 3 on a 20 KB slice
    s = d[:20000]
    mk = lambda: models.Backoff(order=3)
    ideal = models.cost_bits(s, mk()) / 8
    real = {}
    for T in (12, 16, 20):
        blob = coders.encode_with_model(s, mk, 1 << T)
        real[str(T)] = len(blob)
    res["coder_check_20KB"] = dict(ideal_bytes=round(ideal, 1), real_bytes_by_total_bits=real)
    return save("e01_baselines", res)


if __name__ == "__main__":
    print(run())
