import math, unittest
import numpy as np
from hcl import corpus, lenses, models
from experiments import e03_copy_edits as e03, e09_quantum_topology as e09, e10_harmonic as e10, e08_topology as e08


class TestApproaches(unittest.TestCase):
    def test_edit_dp_never_worse_than_exact_dp(self):
        d = corpus.default_corpus()[:6000]
        exact, _ = e03.solve(d, False)
        edits, _ = e03.solve(d, True)
        self.assertLessEqual(edits, exact + 1e-9)

    def test_dp_never_worse_than_greedy(self):
        d = corpus.default_corpus()[:6000]
        g, _ = e03.solve(d, False, greedy=True)
        o, _ = e03.solve(d, False)
        self.assertLessEqual(o, g + 1e-9)

    def test_lens_laws(self):
        d = corpus.default_corpus()[:4000]
        for lens, s in ((lenses.CaseLens(), d), (lenses.WordDictLens(k=30), d.lower()),
                        (lenses.Compose(lenses.CaseLens(), lenses.WordDictLens(k=30)), d)):
            r = lenses.check_laws(lens, [s, b"", b"AbC dEf"] if not isinstance(lens, lenses.WordDictLens) else [s, b""])
            self.assertEqual(r["put_get"], r["samples"]); self.assertEqual(r["get_put"], r["samples"])

    def test_naive_lowercase_is_lossy(self):
        d = b"Hello World"
        self.assertNotEqual(lenses.CaseLens().get(d)[0], d)

    def test_kauffman_bracket_trefoil_and_fig8(self):
        t = e09.bracket(e09.TREFOIL)
        self.assertIn(t, ({-7: 1, -3: -1, 5: -1}, {7: 1, 3: -1, -5: -1}))
        self.assertEqual(e09.bracket(e09.FIG8), {-8: 1, -4: -1, 0: 1, 4: -1, 8: 1})

    def test_loop_weight_is_golden_ratio_at_fifth_root(self):
        r = e09.part_b()
        self.assertAlmostEqual(r["loop_weight_d"], r["golden_ratio"], places=5)

    def test_stack_model_beats_generic_on_planar_source(self):
        d, _ = e09.gen_word(6000)
        generic = models.cost_bits(d, models.Backoff(order=4))
        stack = models.cost_bits(d, e09.StackModel())
        self.assertLess(stack, generic)

    def test_hecke_and_tau(self):
        tau = e10.tau_table(60)
        self.assertEqual(tau[:6], [1, -24, 252, -1472, 4830, -6048])
        self.assertEqual(tau[3 * 5 - 1], tau[2] * tau[4])        # tau(15) = tau(3) tau(5)

    def test_betti_of_triangle(self):
        self.assertEqual(e08.betti([(1, 2), (2, 3), (1, 3)]), dict(V=3, E=3, b0=1, b1=1))

    def test_catalan(self):
        self.assertAlmostEqual(e08.catalan_log2(3), math.log2(5))


if __name__ == "__main__":
    unittest.main()
