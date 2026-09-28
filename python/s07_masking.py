"""Sheet 07 · Masking — the decoder must not peek at future words (paper §3.1, §3.2.3).

    masked score[i][j] = score[i][j] if j <= i, else −∞      (then softmax as usual)

In Python we can use a real −∞: math.exp(-math.inf) is exactly 0.
"""
import math
from common import title, section, work, show_matrix, softmax_steps, dot_work, transpose, fmt
from s06_scaled_attention import attention


def causal(i, j):
    """True where position i may NOT look at position j (j is in the future)."""
    return j > i


def main():
    title("07 · Masking")
    section("Shifted right: what each decoder position sees and predicts ('Die Katze saß')")
    inputs, targets = ["<s>", "Die", "Katze", "saß"], ["Die", "Katze", "saß", "</s>"]
    for i, (x, y) in enumerate(zip(inputs, targets)):
        work(f"position {i+1}", f"input {x:<6} may see {', '.join(inputs[:i+1]):<22} must predict {y}")

    section("Example 1 (simplest): 3×3 scores, apply the mask, softmax")
    toks = ["<s>", "Die", "Katze"]
    S = [[2, 1, 3], [1, 2, 0], [0, 1, 2]]
    masked = [[-math.inf if causal(i, j) else S[i][j] for j in range(3)] for i in range(3)]
    show_matrix("Masked scores", masked, toks, toks)
    rows = [softmax_steps(r) for r in masked]
    show_matrix("Masked attention weights (upper triangle = 0)", [w for _, _, w in rows], toks, toks)
    exps, total, w = rows[1]
    work("row Die", f"e^1, e^2, e^-inf → {fmt(exps[0])}, {fmt(exps[1])}, 0 ÷ {fmt(total)} → "
                    f"{fmt(w[0])}, {fmt(w[1])}, 0")

    section("Example 2 (moderate): full masked attention for '<s> Die Katze saß' (d_k = 2)")
    Q = [[1, 0], [0, 1], [1, 1], [2, 0]]
    K = [[1, 1], [1, 0], [0, 2], [1, -1]]
    V = [[1, 0], [0, 1], [2, 2], [5, -5]]
    weights, out = attention(Q, K, V, mask=causal, verbose=True, labels=inputs)
    work("Out[Katze, d1]", dot_work(weights[2], transpose(V)[0]))

    print("\n  TRY IT: change V for 'saß' — only the last output row changes:")
    V2 = V[:3] + [[100, 100]]
    _, out2 = attention(Q, K, V2, mask=causal)
    for i, name in enumerate(inputs):
        work(name, f"before {[fmt(v) for v in out[i]]}   after {[fmt(v) for v in out2[i]]}")


if __name__ == "__main__":
    main()
