"""Checks that the scripts reproduce the workbook's numbers.  Run:  python -m unittest test_examples

Every expected value below was read from the recalculated Excel workbook (../Backpropagation_Explained.xlsx).
Only the Python standard library is needed.
"""
import unittest

import s01_the_learning_loop as s01
import s03_derivatives as s03
import s04_gradient_descent as s04
import s05_chain_rule as s05
import s06_backprop_one_neuron as s06
import s07_softmax_ce_gradient as s07
import s08_e2e_backprop_2layer as s08
import s09_e2e_training_loop as s09
import s11_optimizers as s11
from common import numeric_derivative


class TestFoundations(unittest.TestCase):
    def test_learning_loop_walks_to_2(self):
        *_, last = s01.train()
        self.assertAlmostEqual(last[5], 1.990181, places=6)

    def test_every_derivative_rule_matches_numeric(self):
        for name, f, d in s03.RULES:
            self.assertAlmostEqual(d(0.7), numeric_derivative(f, 0.7), places=6, msg=name)

    def test_line_fit(self):
        k, w, b, *_ = s04.fit_line()[-1]
        self.assertAlmostEqual(w, 2.010121, places=6)
        self.assertAlmostEqual(b, 0.978391, places=6)

    def test_divergence_with_big_learning_rate(self):
        ws = s04.descend_bowl(1.1)
        self.assertGreater(abs(ws[-1] - 3), abs(ws[1] - 3))

    def test_graph_gradients(self):
        g = s05.graph_forward_backward(2, -3, 10)
        self.assertEqual((g["dL/da"], g["dL/db"], g["dL/dc"]), (-24, 16, 8))


class TestBackprop(unittest.TestCase):
    def test_one_neuron(self):
        *_, gw, gb = s06.backward(0.8, -0.5)
        self.assertAlmostEqual(gw, -0.220701, places=6)
        self.assertAlmostEqual(s06.forward(0.8 - gw, -0.5 - gb)[2], 0.055403, places=6)

    def test_softmax_cross_entropy_gradient(self):
        analytic = s07.ce_gradient([2, 1, 0.1], 0)
        for a, n in zip(analytic, s07.numeric_gradient([2, 1, 0.1], 0)):
            self.assertAlmostEqual(a, n, places=7)

    def test_two_layer_gradients_and_update(self):
        f, _, _, grads = s08.backward(s08.PARAMS)
        expected = [-0.06051, 0.038601, -0.030255, 0.019301, -0.06051, 0.038601, -0.260516, -0.214372, -0.389884]
        for got, exp in zip(grads.values(), expected):
            self.assertAlmostEqual(got, exp, places=6)
        numeric = s08.numeric_gradients(s08.PARAMS)
        for k in grads:
            self.assertAlmostEqual(grads[k], numeric[k], places=7, msg=k)
        self.assertAlmostEqual(f["loss"], 0.494107, places=6)
        self.assertAlmostEqual(s08.forward(s08.step(s08.PARAMS, grads))["loss"], 0.369785, places=6)


class TestTraining(unittest.TestCase):
    def test_or_learned_xor_not(self):
        or_rows, xor_rows = s09.train(s09.TASKS["OR"]), s09.train(s09.TASKS["XOR"])
        self.assertAlmostEqual(or_rows[-1][5], 0.106777, places=6)
        self.assertEqual(or_rows[-1][6], 1.0)
        self.assertAlmostEqual(xor_rows[-1][5], 0.693147, places=6)
        self.assertLessEqual(max(r[6] for r in xor_rows), 0.5)

    def test_optimizers_at_step_20(self):
        self.assertAlmostEqual(s11.loss(s11.sgd()[20]), 0.236494, places=6)
        self.assertAlmostEqual(s11.loss(s11.momentum()[20]), 0.0, places=4)
        self.assertAlmostEqual(s11.loss(s11.adam()[20]), 0.311097, places=6)


if __name__ == "__main__":
    unittest.main()
