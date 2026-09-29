"""Checks that the scripts reproduce the workbook's numbers.  Run:  python -m unittest test_examples

Expected values were read from the recalculated Excel workbook (../Neural_Network_Basics_Explained.xlsx).
Only the Python standard library is needed.
"""
import unittest

from common import linear, softmax, sigmoid, relu, gelu
from s01_what_is_a_neural_net import count_parameters
from s04_the_neuron import neuron
from s05_linear_layer import collapse
from s07_softmax import stable_softmax
from s08_why_layers_xor import xor_network, best_single_neuron
import s09_e2e_fruit_classifier as fruit
import s10_e2e_exam_predictor as exam
import s11_e2e_bridge_to_attention as bridge
from common import step


class Close(unittest.TestCase):
    def assertListAlmostEqual(self, got, expected, places=6):
        self.assertEqual(len(got), len(expected))
        for g, e in zip(got, expected):
            self.assertAlmostEqual(g, e, places=places)


class TestBuildingBlocks(Close):
    def test_parameter_count(self):
        self.assertEqual(sum(count_parameters([3, 4, 2])), 26)

    def test_walk_neuron(self):
        self.assertEqual(neuron([1, 1, 0], [2, 1, -3], -1.5, step), (1.5, 1.0))

    def test_linear_layer_and_collapse(self):
        x, W, b = [1, 2, 3], [[0.5, -1], [1, 0], [0, 2]], [0.5, -1]
        y = linear(x, W, b)
        self.assertListAlmostEqual(y, [3, 4])
        W2, b2 = [[1, 2], [-1, 0.5]], [0, 1]
        Wc, bc = collapse(W, b, W2, b2)
        self.assertListAlmostEqual(linear(y, W2, b2), linear(x, Wc, bc))

    def test_activations(self):
        self.assertAlmostEqual(sigmoid(-2), 0.119203, places=6)
        self.assertEqual(relu(-2), 0)
        self.assertAlmostEqual(gelu(3), 2.996363, places=6)

    def test_softmax(self):
        self.assertListAlmostEqual(softmax([2, 1, 0.1]), [0.659001, 0.242433, 0.098566])
        self.assertListAlmostEqual(stable_softmax([1000, 999, 998]), [0.665241, 0.244728, 0.090031])

    def test_xor(self):
        self.assertEqual([xor_network(a, b)[4] for a in (0, 1) for b in (0, 1)], [0, 1, 1, 0])
        self.assertEqual(best_single_neuron(), 3)


class TestEndToEnd(Close):
    def test_fruit(self):
        a = fruit.forward([0.8, 0.7, 0.2])
        self.assertListAlmostEqual(a["z1"], [0.8, -0.2, -0.4])
        self.assertListAlmostEqual(a["p"], [0.924621, 0.037690, 0.037690])
        self.assertEqual(a["prediction"], "apple")
        self.assertEqual(fruit.forward([0.9, 0.1, 0.9])["prediction"], "banana")
        self.assertEqual(fruit.forward([0.2, 0.9, 0.8])["prediction"], "carrot")

    def test_exam(self):
        a, b = exam.forward([8, 7, 3]), exam.forward([2, 5, 0])
        self.assertAlmostEqual(a["p"], 0.906964, places=6)
        self.assertEqual((a["decision"], b["decision"]), ("PASS", "FAIL"))
        self.assertEqual(exam.forward([5, 8, 1], threshold=0.4)["decision"], "PASS")

    def test_bridge_attention(self):
        *_, weights, output = bridge.attention(bridge.X)
        for row in weights:
            self.assertAlmostEqual(sum(row), 1.0)
        self.assertListAlmostEqual(output[2], [1.254631, 1.418210])


if __name__ == "__main__":
    unittest.main()
