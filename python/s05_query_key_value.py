"""Sheet 05 · Queries, Keys and Values (paper §3.2, §3.2.2).

BIG IDEA: attention is a soft dictionary lookup. A QUERY is compared with every
KEY; the matches become weights; the answer is the weighted blend of VALUES.
In self-attention all three come from the same words: Q = X·W_Q, K = X·W_K, V = X·W_V.
"""
from common import title, section, work, show_matrix, dot, dot_work, matmul, transpose, softmax_steps, fmt
from s03_embeddings import SENTENCE
from s04_positional_encoding import encoder_input

# Made-up but fixed projection matrices (learned in a real model). Shared with scripts 06 and 08.
W_Q = [[1, 0, 1, 0], [0, 1, 0, -1], [1, 0, -1, 0], [0, -1, 0, 1]]
W_K = [[1, 0, 0, 1], [0, 1, 1, 0], [0, 1, -1, 0], [1, 0, 0, -1]]
W_V = [[0.5, 0, 0.5, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 0.5]]


def soft_lookup(query, keys, values):
    """Score = q·k for each key → softmax → weighted sum of values."""
    scores = [dot(query, k) for k in keys]
    exps, total, weights = softmax_steps(scores)
    answer = sum(w * v for w, v in zip(weights, values))
    return scores, exps, total, weights, answer


def project_qkv(X):
    """Three learned 'lenses' applied to the same words."""
    return matmul(X, W_Q), matmul(X, W_K), matmul(X, W_V)


def main():
    title("05 · Queries, Keys and Values")
    section("Example 1 (simplest): a fruit shop — query 'sweet, please' = [1, 0]")
    names = ["mango", "carrot", "apple"]
    keys = [[2, 0], [0, 2], [1, 1]]            # [sweet, crunchy]
    prices = [30, 10, 20]                      # values
    q = [1, 0]
    scores, exps, total, weights, answer = soft_lookup(q, keys, prices)
    for n, k in zip(names, keys):
        work(f"score {n}", "q·k = " + dot_work(q, k))
    for n, e, w in zip(names, exps, weights):
        work(f"weight {n}", f"{fmt(e)} ÷ {fmt(total)} = {fmt(w)}")
    work("soft answer", " + ".join(f"{fmt(w)}×{p}" for w, p in zip(weights, prices)) + f"  =  {fmt(answer)}")
    work("hard lookup", f"{prices[scores.index(max(scores))]} (an ordinary dictionary returns only the best match)")

    section("Example 2 (moderate): Q, K, V for 'the cat sat' (single head, d_k = 4)")
    X = encoder_input()
    Q, K, V = project_qkv(X)
    show_matrix("X (from script 04)", X, SENTENCE)
    show_matrix("Q = X·W_Q", Q, SENTENCE, ["q1", "q2", "q3", "q4"])
    show_matrix("K = X·W_K", K, SENTENCE, ["k1", "k2", "k3", "k4"])
    show_matrix("V = X·W_V", V, SENTENCE, ["v1", "v2", "v3", "v4"])
    for j, col in enumerate(transpose(W_Q)):
        work(f"Q[cat, q{j+1}]", dot_work(X[1], col))


if __name__ == "__main__":
    main()
