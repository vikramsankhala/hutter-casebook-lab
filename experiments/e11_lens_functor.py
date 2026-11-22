"""E11 - Functorial approaches (Case 14): preprocessing as lenses, compression as an approximate monoid map.

(a) CaseLens and WordDictLens; PutGet / GetPut checked on the corpus and on random inputs; composition checked.
    Cost accounting: downstream(view) + cost(complement) [+ dictionary]. A lowercase 'view' WITHOUT its
    complement is shown to be lossy.
(b) C(x) + C(y) - C(xy) is the *defect* from additivity: a compressor is an approximate monoid homomorphism
    (strings, concat) -> (bits, +).  The defect measures shared structure (mutual information proxy).
"""
import lzma, os, random
from hcl import models, corpus
from hcl.lenses import CaseLens, WordDictLens, Compose, check_laws
from experiments.common import get_data, save

N = 60_000


def xz(b): return len(lzma.compress(b, preset=9 | lzma.PRESET_EXTREME))


def comp_bytes(c):
    """Serialise a complement (tuples/ints/bytes) deterministically and xz it."""
    def ser(x):
        if isinstance(x, bytes): return x + b"\x00"
        if isinstance(x, int): return x.to_bytes(3, "big")
        return b"".join(ser(y) for y in x)
    return xz(ser(c))


def part_a():
    d = get_data(N)
    case, wd = CaseLens(), WordDictLens(k=100)
    pipe = Compose(case, wd)
    rnd = random.Random(0)
    rand = [bytes(rnd.randrange(128) for _ in range(60)) for _ in range(200)]
    laws = dict(case=check_laws(case, [d[:5000]] + rand), worddict=check_laws(wd, [d[:5000].lower()] + [r.lower() for r in rand]),
                composed=check_laws(pipe, [d[:5000]] + rand))
    lossy = case.get(d)[0]
    naive_lossless = (lossy == d)
    v, c = pipe.get(d)
    assert pipe.put(v, c) == d
    cc, cw = c
    res = dict(bytes=len(d), laws=laws, naive_lowercase_without_complement_is_lossless=naive_lossless,
               view_bytes=len(v), upper_case_letters=len(cc), dictionary_words=len(cw))
    res["xz"] = dict(raw=xz(d), pipeline_view=xz(v), complement_case=comp_bytes(cc), complement_dict=comp_bytes(cw))
    res["xz"]["pipeline_total"] = res["xz"]["pipeline_view"] + res["xz"]["complement_case"] + res["xz"]["complement_dict"]
    m = lambda x: round(models.cost_bits(x, models.Backoff(order=3)) / 8)
    raw3, view3 = m(d), m(v)
    res["order3_cm"] = dict(raw=raw3, pipeline_view=view3)
    res["order3_cm"]["pipeline_total_with_xz_complements"] = view3 + res["xz"]["complement_case"] + res["xz"]["complement_dict"]
    return res


def part_b():
    d = get_data(None)
    half = len(d) // 2
    pages = d.split(b"  <page>\n")
    titles = {}
    pair = None
    for p in pages:
        t = p.split(b"</title>")[0]
        if t in titles: pair = (titles[t], p); break
        titles[t] = p
    rr = random.Random(1)
    rb = lambda n: bytes(rr.randrange(256) for _ in range(n))
    cases = {"two halves of corpus": (d[:half][:40000], d[half:][:40000]),
             "article and its revision": pair,
             "two independent random strings": (rb(5000), rb(5000))}
    out = {}
    for name, (x, y) in cases.items():
        cx, cy, cxy = xz(x), xz(y), xz(x + y)
        out[name] = dict(Cx=cx, Cy=cy, Cxy=cxy, defect_bytes=cx + cy - cxy,
                         NCD=round((cxy - min(cx, cy)) / max(cx, cy), 4))
    return out


def run():
    return save("e11_lens_functor", dict(a_lenses=part_a(), b_subadditivity_defect=part_b()))


if __name__ == "__main__":
    print(run())
