"""Sheet 11 · End-to-end example 3 — the same building blocks, rearranged, ARE attention.

    Q = X·W_Q   K = X·W_K   V = X·W_V   weights = softmax(Q·Kᵀ / √d)   output = weights · V

Continue with ../../../python/s05_query_key_value.py and s06_scaled_attention.py.
"""
import math
from common import title, section, work, show_matrix, dot, dot_work, matmul, softmax, vec, show_stages, fmt

WORDS = ["I", "love", "pizza"]
X = [[1, 0, 1], [0, 2, 0], [1, 1, 1]]
W_Q = [[1, 0], [0, 1], [1, 0]]
W_K = [[0, 1], [1, 0], [1, 1]]
W_V = [[1, 0], [0, 1], [0.5, 0.5]]


def attention(X):
    Q, K, V = matmul(X, W_Q), matmul(X, W_K), matmul(X, W_V)       # stage 1: three linear layers
    d = len(Q[0])
    scores = [[dot(q, k) / math.sqrt(d) for k in K] for q in Q]      # stage 2: dot products, scaled
    weights = [softmax(row) for row in scores]                       # stage 3: softmax per row
    output = matmul(weights, V)                                      # stage 4: weighted sum of values
    return Q, K, V, scores, weights, output


def main():
    title("11 · End-to-End Example 3 — Bridge to Attention")
    Q, K, V, S, A, O = attention(X)
    section("Stage 1 · three linear layers")
    show_matrix("Q = X·W_Q", Q, WORDS, ["a", "b"])
    show_matrix("K = X·W_K", K, WORDS, ["a", "b"])
    show_matrix("V = X·W_V", V, WORDS, ["a", "b"])
    section("Stage 2 · dot products ÷ √d")
    for j, w in enumerate(WORDS):
        work(f"pizza → {w}", f"{dot_work(Q[2], K[j])}   ÷ √2  =  {fmt(S[2][j])}")
    section("Stage 3 · softmax on each row")
    show_matrix("attention weights", A, WORDS, WORDS)
    section("Stage 4 · weighted sum of values")
    show_matrix("output = weights · V", O, WORDS, ["a", "b"])
    show_stages([("0 · input", "word vector x", "1 × 3", vec(X[2])),
                 ("1 · linear ×3", "q = x·W_Q (k, v alike)", "1 × 2", vec(Q[2])),
                 ("2 · dot products", "q · every key ÷ √d", "1 × 3", vec(S[2])),
                 ("3 · softmax", "weights sum to 1", "1 × 3", vec(A[2])),
                 ("4 · weighted sum", "Σ weight × value", "1 × 2", vec(O[2]))])


if __name__ == "__main__":
    main()
