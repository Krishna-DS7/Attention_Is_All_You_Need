"""Sheet 11 · Position-wise Feed-Forward Network — Equation (2), paper §3.3.

    FFN(x) = max(0, x·W1 + b1) · W2 + b2

Expand each word's vector (512 → 2048), switch off negatives (ReLU), compress back.
The same weights are applied to every position.
"""
from common import title, section, work, show_matrix, show_vector, dot, dot_work, transpose, relu, fmt
from s03_embeddings import SENTENCE
from s04_positional_encoding import encoder_input
from s08_multi_head import multi_head
from s10_add_and_norm import add_and_norm

# Example 2 weights (d_model = 4, d_ff = 8), same as the workbook
W1 = [[0.5, -0.5, 1, 0, 0.5, -1, 0, 0.5], [1, 0.5, 0, -0.5, 0, 0.5, -1, 0],
      [0, 1, -0.5, 1, -0.5, 0, 0.5, 1], [-0.5, 0, 0.5, 0.5, 1, 0, 0.5, -0.5]]
B1 = [0, 0.1, 0, -0.1, 0, 0.1, 0, 0]
W2 = [[0.5, 0, 0, 0.5], [0, 0.5, 0.5, 0], [0.5, 0.5, 0, 0], [0, 0, 0.5, 0.5], [0.5, 0, 0.5, 0],
      [0, 0.5, 0, 0.5], [-0.5, 0, 0, 0.5], [0, -0.5, 0.5, 0]]
B2 = [0, 0, 0, 0]


def ffn(x, W1, b1, W2, b2):
    """One word: expand, ReLU, compress. Returns (pre-activation, hidden, output)."""
    pre = [dot(x, col) + b for col, b in zip(transpose(W1), b1)]
    hidden = [relu(p) for p in pre]
    out = [dot(hidden, col) + b for col, b in zip(transpose(W2), b2)]
    return pre, hidden, out


def encoder_layer(X):
    """A complete encoder layer: multi-head attention → Add & Norm → FFN → Add & Norm."""
    _, MH = multi_head(X)
    LN1 = add_and_norm(X, MH)
    F = [ffn(row, W1, B1, W2, B2)[2] for row in LN1]      # same weights for every word
    return LN1, F, add_and_norm(LN1, F)


def main():
    title("11 · Feed-Forward Network")
    section("Example 1 (simplest): d_model = 2, d_ff = 4")
    x = [1, 2]
    w1 = [[1, -1, 0.5, 2], [0, 1, -1, -0.5]]
    b1 = [0, 0.5, 0, -1]
    w2 = [[1, 0], [0, 1], [3, 3], [-1, 2]]
    b2 = [0.1, -0.1]
    pre, hidden, out = ffn(x, w1, b1, w2, b2)
    for j, col in enumerate(transpose(w1)):
        work(f"h{j+1}", f"{dot_work(x, col)}, + b1 {fmt(b1[j])} = {fmt(pre[j])}")
    work("ReLU", f"{[fmt(p) for p in pre]} → {[fmt(h) for h in hidden]}")
    for j, col in enumerate(transpose(w2)):
        work(f"o{j+1}", f"{dot_work(hidden, col)}, + b2 {fmt(b2[j])} = {fmt(out[j])}")

    section("Example 2 (moderate): FFN on all 3 words, then Add & Norm = ENCODER LAYER 1 OUTPUT")
    LN1, F, out = encoder_layer(encoder_input())
    show_matrix("Input (from script 10)", LN1, SENTENCE)
    show_matrix("FFN output", F, SENTENCE)
    show_matrix("ENCODER LAYER 1 OUTPUT", out, SENTENCE)
    print("\n  That is one full encoder layer: embed → +position → Q,K,V → multi-head → Add&Norm → FFN → Add&Norm.")

    section("Example 3 (moderate): FFN weights in the base model")
    d, dff = 512, 2048
    per = 2 * d * dff + dff + d
    work("per FFN", f"2×{d}×{dff} + {dff} + {d} = {per:,}")
    work("× 12 FFNs", f"{per * 12:,}")


if __name__ == "__main__":
    main()
