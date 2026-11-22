"""Deterministic synthetic Wikipedia-style corpus (the "River Archive II").

Why synthetic?  The official benchmark files (enwik8/enwik9) are not bundled.
This generator produces wiki-markup with the *structures* the Hutter Prize case
cares about: infoboxes with aligned columns, wikilinks, nested templates,
citations with drifting fields, numeric tables, and near-duplicate revisions.
Numbers measured here are properties of THIS corpus, never of enwik9.

Use `load_enwik(path, offset, length)` to run any experiment on a real slice.
"""
from __future__ import annotations
import hashlib
import numpy as np

FUNC = ["the", "of", "and", "in", "was", "is", "a", "to", "by", "for", "with",
        "as", "on", "from", "that", "its", "which", "at", "an", "are"]
SYL = ["ka", "re", "mo", "li", "ve", "tan", "bel", "dor", "sin", "ar", "um",
       "pa", "ne", "ro", "ta", "shi", "gal", "wen", "or", "ith", "lu", "man",
       "dar", "est", "oun", "ber", "fal", "gri", "hov", "jin"]
KINDS = ["town", "village", "river", "hamlet", "district", "valley", "lake"]
FIELDS_CITE = ["url", "title", "year", "publisher"]


def _word(rng, lo=1, hi=3) -> str:
    return "".join(rng.choice(SYL) for _ in range(int(rng.integers(lo, hi + 1))))


def _vocab(rng, n):
    seen, out = set(), []
    while len(out) < n:
        w = _word(rng, 2, 4)
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


class _Gen:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.vocab = _vocab(self.rng, 1500)
        ranks = 1.0 / np.arange(1, len(self.vocab) + 1)
        self.zipf = ranks / ranks.sum()
        self.places = [w.capitalize() for w in self.vocab[:120]]

    def word(self):
        return self.vocab[int(self.rng.choice(len(self.vocab), p=self.zipf))]

    def link(self):
        p = self.places[int(self.rng.integers(0, len(self.places)))]
        if self.rng.random() < 0.25:
            return "[[%s|%s]]" % (p, self.word())
        return "[[%s]]" % p

    def cite(self, year):
        rng = self.rng
        name = "Cite web" if rng.random() < 0.3 else "cite web"
        host = self.word()
        return ("<ref>{{%s |url=http://www.%s.org/%s |title=%s %s |year=%d |publisher=%s}}</ref>"
                % (name, host, self.word(), self.word().capitalize(), self.word(), year,
                   self.word().capitalize()))

    def sentence(self, year):
        rng = self.rng
        toks = []
        for _ in range(int(rng.integers(8, 19))):
            r = rng.random()
            if r < 0.34:
                toks.append(FUNC[int(rng.integers(0, len(FUNC)))])
            elif r < 0.40:
                toks.append(self.link())
            else:
                toks.append(self.word())
        s = " ".join(toks)
        s = s[0].upper() + s[1:] + "."
        if rng.random() < 0.15:
            s += self.cite(year)
        if rng.random() < 0.10:
            s += " The area is {{convert|%d|km|mi}}." % int(rng.integers(3, 900))
        return s

    def article(self, idx):
        rng = self.rng
        title = self.places[idx % len(self.places)] + " " + self.word().capitalize()
        pop = int(rng.integers(900, 90000))
        est = int(rng.integers(1650, 1950))
        area = round(float(rng.uniform(2, 400)), 1)
        info = ("{{Infobox settlement\n"
                "| name              = %s\n"
                "| population        = %d\n"
                "| area_km2          = %s\n"
                "| established       = %d\n"
                "| region            = %s\n"
                "}}\n" % (title, pop, area, est, self.places[int(rng.integers(0, 30))]))
        lead = "'''%s''' is a %s in %s, %s. " % (
            title, KINDS[int(rng.integers(0, len(KINDS)))], self.link(), self.link())
        body = []
        for _ in range(int(rng.integers(3, 7))):
            body.append(" ".join(self.sentence(int(rng.integers(1990, 2012)))
                                 for _ in range(int(rng.integers(3, 6)))))
        # numeric table: arithmetic-ish growth with noise
        yrs = list(range(1950, 2011, 10))
        base, g = pop // 3, int(rng.integers(100, 2500))
        rows = ["|-\n| %d || %s" % (y, format(base + g * k + int(rng.integers(-40, 40)), ","))
                for k, y in enumerate(yrs)]
        table = "{| class=\"wikitable\"\n! Year !! Population\n" + "\n".join(rows) + "\n|}\n"
        cats = "".join("[[Category:%s]]\n" % self.places[int(rng.integers(0, 12))] for _ in range(2))
        text = info + lead + "\n\n".join(body) + "\n\n== Demographics ==\n" + table + "\n" + cats
        return title, text, dict(pop=pop, est=est)

    def revise(self, text: str) -> str:
        """Near-duplicate revision: digit substitutions plus an inserted sentence."""
        rng = self.rng
        out = list(text)
        digits = [i for i, c in enumerate(out) if c.isdigit()]
        for i in rng.choice(digits, size=min(4, len(digits)), replace=False):
            out[int(i)] = str((int(out[int(i)]) + int(rng.integers(1, 9))) % 10)
        text = "".join(out)
        cut = text.find("\n\n")
        if cut > 0:
            text = text[:cut] + " " + self.sentence(2013) + text[cut:]
        return text


def primes_article(n=200) -> str:
    ps, k = [], 2
    while len(ps) < n:
        if all(k % p for p in ps if p * p <= k):
            ps.append(k)
        k += 1
    rows = "\n".join("* %d" % p for p in ps)
    return "== List of primes ==\n" + rows + "\n"


def generate(seed: int = 7, n_articles: int = 70, revision_rate: float = 0.30) -> bytes:
    g = _Gen(seed)
    pages, texts = [], []
    for i in range(n_articles):
        title, text, _ = g.article(i)
        texts.append((title, text))
        pages.append((title, text))
        if i > 4 and g.rng.random() < revision_rate:
            j = int(g.rng.integers(0, len(texts) - 1))
            t2, x2 = texts[j]
            pages.append((t2, g.revise(x2)))
    pages.insert(len(pages) // 2, ("List of prime numbers", primes_article()))
    out = []
    for title, text in pages:
        out.append("  <page>\n    <title>%s</title>\n    <text xml:space=\"preserve\">%s</text>\n  </page>\n"
                   % (title, text))
    return "".join(out).encode("utf-8")


def load_enwik(path: str, offset: int = 0, length: int = 1 << 18) -> bytes:
    """Load a slice of a real enwik8/enwik9 file for the same experiments."""
    with open(path, "rb") as f:
        f.seek(offset)
        return f.read(length)


def manifest(data: bytes, label: str) -> dict:
    return dict(label=label, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


_cache = {}


def default_corpus() -> bytes:
    if "d" not in _cache:
        _cache["d"] = generate()
    return _cache["d"]
