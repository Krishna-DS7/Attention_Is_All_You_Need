"""Checks that the scripts reproduce the workbook's numbers.  Run:  python -m unittest test_examples

Expected values were read from the recalculated Excel workbook (Attention_Is_All_You_Need_Explained.xlsx).
Only the Python standard library is needed.
"""
import math
import unittest

from common import softmax, cosine
from s03_embeddings import lookup
from s04_positional_encoding import encoder_input, positional_encoding
from s05_query_key_value import project_qkv, soft_lookup
from s06_scaled_attention import attention
from s07_masking import causal
from s08_multi_head import multi_head
from s11_feed_forward import encoder_layer
from s12_output_and_decoding import greedy, beam_search
from s13_loss_and_smoothing import perplexity
from s14_optimizer_and_lr import learning_rate
from s17_model_anatomy import count_params
from s18_bleu import bleu
from s19_results import training_flops


class MatrixCase(unittest.TestCase):
    def assertMatrixAlmostEqual(self, got, expected, places=6):
        for rg, re_ in zip(got, expected):
            for g, e in zip(rg, re_):
                self.assertAlmostEqual(g, e, places=places)


class TestToolkit(MatrixCase):
    def test_softmax(self):
        self.assertMatrixAlmostEqual([softmax([2, 1, 0.1])], [[0.659001, 0.242433, 0.098566]])

    def test_cat_katze_similar(self):
        self.assertGreater(cosine(lookup("cat"), lookup("Katze")), 0.95)


class TestRunningExample(MatrixCase):
    """The 'the cat sat' thread through sheets 03 → 11."""

    def setUp(self):
        self.X = encoder_input()

    def test_encoder_input(self):
        self.assertMatrixAlmostEqual(self.X, [[1.0, 0.6, 0.2, 1.0],
                                              [0.241471, 2.140302, 0.81, 0.79995],
                                              [1.309297, -0.216147, -1.180001, 1.9998]])

    def test_single_head_weights(self):
        weights, _ = attention(*project_qkv(self.X))
        self.assertMatrixAlmostEqual(weights, [[0.192719, 0.09151, 0.715771],
                                               [0.223861, 0.637769, 0.13837],
                                               [0.078179, 0.011631, 0.91019]])
        for row in weights:
            self.assertAlmostEqual(sum(row), 1.0)

    def test_multi_head(self):
        self.assertMatrixAlmostEqual(multi_head(self.X)[1], [[1.828765, -0.015246, 0.39794, 1.377686],
                                                             [1.026093, 1.685141, 1.107662, 1.019722],
                                                             [1.9154, -0.191566, 0.449971, 1.37756]])

    def test_encoder_layer_output(self):
        self.assertMatrixAlmostEqual(encoder_layer(self.X)[2], [[1.555322, -0.833474, -0.920281, 0.198432],
                                                                [-1.046442, 1.576819, -0.637982, 0.107604],
                                                                [1.433397, -0.86582, -1.003043, 0.435466]])


class TestAttentionDetails(MatrixCase):
    def test_fruit_shop(self):
        self.assertAlmostEqual(soft_lookup([1, 0], [[2, 0], [0, 2], [1, 1]], [30, 10, 20])[4], 25.752104, places=6)

    def test_mask_blocks_future(self):
        Q = [[1, 0], [0, 1], [1, 1], [2, 0]]
        K = [[1, 1], [1, 0], [0, 2], [1, -1]]
        V = [[1, 0], [0, 1], [2, 2], [5, -5]]
        w, out = attention(Q, K, V, mask=causal)
        self.assertEqual(w[0][1:], [0.0, 0.0, 0.0])
        _, out2 = attention(Q, K, V[:3] + [[100, 100]], mask=causal)
        self.assertEqual(out[:3], out2[:3])          # changing the last word never affects earlier rows

    def test_positional_rotation(self):
        p, k = 3, 2
        self.assertAlmostEqual(math.sin(p + k), math.sin(p) * math.cos(k) + math.cos(p) * math.sin(k))
        self.assertEqual(positional_encoding(0, 4), [0.0, 1.0, 0.0, 1.0])


class TestPaperNumbers(unittest.TestCase):
    def test_peak_learning_rate(self):
        self.assertAlmostEqual(learning_rate(4000), 0.000698771, places=9)

    def test_parameter_counts(self):
        self.assertEqual(count_params(512, 2048)["TOTAL"], 63_082_496)      # paper: 65M
        self.assertEqual(count_params(1024, 4096)["TOTAL"], 214_245_376)    # paper: 213M

    def test_flops(self):
        self.assertAlmostEqual(training_flops(12) / 1e18, 3.2832, places=4)   # paper: 3.3e18
        self.assertAlmostEqual(training_flops(84) / 1e19, 2.29824, places=5)  # paper: 2.3e19

    def test_beam_beats_greedy(self):
        self.assertEqual(greedy()[0], "Die Katze")
        self.assertEqual(beam_search(2)[0][0], "Eine Katze")

    def test_bleu(self):
        self.assertAlmostEqual(bleu("the cat sat on mat".split(), "the cat sat on the mat".split())[0], 0.578930, places=6)

    def test_perplexity(self):
        self.assertAlmostEqual(perplexity([0.6, 0.3, 0.9, 0.8]), 5 / 3, places=9)


if __name__ == "__main__":
    unittest.main()
