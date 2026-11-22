import unittest
from fractions import Fraction
import numpy as np

from hcl import corpus, coders, models


class TestCorpus(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(corpus.generate(), corpus.generate())

    def test_different_seed(self):
        self.assertNotEqual(corpus.generate(seed=7), corpus.generate(seed=8))


class TestCoders(unittest.TestCase):
    def test_exact_abA(self):
        ac = coders.ExactArithmetic({"A": Fraction(3, 4), "B": Fraction(1, 4)})
        lo, hi = ac.encode("ABA")
        self.assertEqual((lo, hi), (Fraction(9, 16), Fraction(45, 64)))
        self.assertEqual(ac.decode(Fraction(5, 8), 3), list("ABA"))

    def test_range_roundtrip_binary_and_multisymbol(self):
        rng = np.random.default_rng(1)
        for k in (2, 5, 256):
            p = rng.dirichlet(np.ones(k))
            f = coders.quantize(p, 1 << 16)
            self.assertEqual(int(f.sum()), 1 << 16)
            c = np.concatenate(([0], np.cumsum(f)))
            syms = rng.choice(k, size=3000, p=p)
            enc = coders.RangeEncoder()
            for s in syms:
                enc.encode(int(c[s]), int(c[s + 1]), 1 << 16)
            blob = enc.finish()
            dec = coders.RangeDecoder(blob)
            out = []
            for _ in syms:
                t = dec.target(1 << 16)
                s = int(np.searchsorted(c, t, side="right") - 1)
                dec.consume(int(c[s]), int(c[s + 1]), 1 << 16)
                out.append(s)
            self.assertEqual(out, list(syms))

    def test_model_roundtrip_and_cost_matches_ideal(self):
        data = corpus.default_corpus()[:6000]
        mk = lambda: models.Backoff(order=2)
        blob = coders.encode_with_model(data, mk)
        self.assertEqual(coders.decode_with_model(blob, len(data), mk), data)
        ideal = models.cost_bits(data, mk()) / 8
        self.assertLess(abs(len(blob) - ideal), 0.05 * ideal + 8)  # quantisation sharpens an under-confident model (see E02)

    def test_rans_roundtrip(self):
        fr = [4, 2, 1, 1]
        s = [0, 1, 3, 2, 0, 0, 1, 2, 3, 3]
        x = coders.rans_encode(s, fr)
        self.assertEqual(coders.rans_decode(x, len(s), fr), s)

    def test_rans_worked_example(self):
        # Appendix D of the participant textbook: freq (4,2,1,1), symbol 1, x=13 -> 53
        fr, cum = [4, 2, 1, 1], [0, 4, 6, 7, 8]
        q, r = divmod(13, fr[1])
        self.assertEqual(8 * q + cum[1] + r, 53)


class TestEntropy(unittest.TestCase):
    def test_entropy_dyadic(self):
        self.assertAlmostEqual(coders.entropy([.5, .25, .125, .125]), 1.75)


if __name__ == "__main__":
    unittest.main()
