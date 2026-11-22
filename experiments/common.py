"""Shared helpers for experiments."""
import json, os, math, time, hashlib
from collections import Counter
from hcl import corpus

RESULTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)


def get_data(n=None, enwik=None, offset=0):
    """Synthetic corpus by default; set HCL_ENWIK=/path/enwik8 to use a real slice."""
    path = enwik or os.environ.get("HCL_ENWIK")
    if path:
        return corpus.load_enwik(path, offset, n or (1 << 17))
    d = corpus.default_corpus()
    return d[:n] if n else d


def order0_bits(data: bytes) -> float:
    c = Counter(data); n = len(data)
    return -sum(v * math.log2(v / n) for v in c.values())


def save(name: str, result: dict):
    result = dict(result)
    result["_experiment"] = name
    with open(os.path.join(RESULTS, name + ".json"), "w") as f:
        json.dump(result, f, indent=2, default=float)
    return result


def timed(fn, *a, **k):
    t = time.time(); r = fn(*a, **k); return r, round(time.time() - t, 2)
