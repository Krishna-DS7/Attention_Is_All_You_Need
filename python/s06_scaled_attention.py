"""Sheet 06 · Scaled Dot-Product Attention — Equation (1), paper §3.2.1.

    Attention(Q, K, V) = softmax( Q·Kᵀ / √d_k ) · V

Four moves: score every query against every key, divide by √d_k,
softmax each row, blend the values.
"""
import math
from common import title, section, work, show_matrix, dot, dot_work, matmul, transpose, softmax_steps, fmt
from s03_embeddings import SENTENCE
from s04_positional_encoding import encoder_input
from s05_query_key_value import project_qkv


def attention(Q, K, V, mask=None, verbose=False, labels=None):
    """Scaled dot-product attention. `mask(i, j)` returns True where row i may NOT look at column j.
    Returns (weights, output)."""
    d_k = len(K[0])
    n = len(Q)
    labels = labels or [str(i + 1) for i in range(n)]
    scores = [[dot(q, k) / math.sqrt(d_k) for k in K] for q in Q]                    # steps 1 & 2
    if mask:
        scores = [[-math.inf if mask(i, j) else s for j, s in enumerate(row)] for i, row in enumerate(scores)]
    steps = [softmax_steps(row) for row in scores]                                      # step 3
    weights = [w for _, _, w in steps]
    output = matmul(weights, V)                                                        # step 4
    if verbose:
        show_matrix(f"S ÷ √d_k  (d_k = {d_k}, √d_k = {fmt(math.sqrt(d_k))})", scores, labels, labels)
        show_matrix("Attention weights (each row sums to 1)", weights, labels, labels)
        show_matrix("Output = weights · V", output, labels)
    return weights, output


def main():
    title("06 · Scaled Dot-Product Attention")
    section("Example 1 (simplest): 3 words, d_k = 2, Q/K/V given directly")
    toks = ["I", "love", "pizza"]
    Q = [[1, 0], [1, 1], [0, 2]]
    K = [[2, 0], [0, 1], [1, 1]]
    V = [[1, 0], [0, 1], [1, 1]]
    for i in range(3):
        for j in range(3):
            work(f"S[{toks[i]}→{toks[j]}]", dot_work(Q[i], K[j]))
    weights, out = attention(Q, K, V, verbose=True, labels=toks)
    exps, total, w = softmax_steps([dot(Q[0], k) / math.sqrt(2) for k in K])
    work("row 'I'", f"({', '.join(fmt(e) for e in exps)}) ÷ {fmt(total)} = ({', '.join(fmt(x) for x in w)})")
    for i in range(3):
        work(f"Out[{toks[i]}, d1]", dot_work(weights[i], transpose(V)[0]))

    section("Example 2 (moderate): the real Q, K, V of 'the cat sat' from script 05 (d_k = 4)")
    Q, K, V = project_qkv(encoder_input())
    weights, out = attention(Q, K, V, verbose=True, labels=SENTENCE)
    for j, name in enumerate(SENTENCE):
        work(f"S[cat→{name}]", dot_work(Q[1], K[j]))

    section("Why divide by √d_k? (footnote 4) — keys at +1 std, 0, −1 std")
    for d_k in [1, 4, 16, 64, 512]:
        sd = math.sqrt(d_k)
        p_raw = math.exp(sd) / (math.exp(sd) + 1 + math.exp(-sd))
        p_scaled = math.exp(1) / (math.exp(1) + 1 + math.exp(-1))
        work(f"d_k = {d_k}", f"score size {sd:6.2f} | top weight unscaled {p_raw:.5f}, gradient {p_raw*(1-p_raw):.5f}"
                             f" | scaled {p_scaled:.3f}, gradient {p_scaled*(1-p_scaled):.3f}")


if __name__ == "__main__":
    main()
