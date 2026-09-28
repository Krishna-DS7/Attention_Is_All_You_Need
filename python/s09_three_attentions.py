"""Sheet 09 · The Three Uses of Attention (paper §3.2.3).

1. Encoder self-attention:  Q, K, V all from the encoder.            No mask.
2. Masked decoder self-attention: Q, K, V all from the decoder.      Causal mask.
3. Encoder–decoder (cross) attention: Q from the decoder, K and V from the ENCODER OUTPUT.
"""
import math
from common import title, section, work, dot, dot_work, softmax_steps, fmt


def cross_attention(query, memory):
    """Simplified: the encoder memory is used directly as keys AND values (real model projects each)."""
    d_k = len(query)
    scores = [dot(query, m) / math.sqrt(d_k) for m in memory]
    exps, total, weights = softmax_steps(scores)
    context = [sum(w * m[j] for w, m in zip(weights, memory)) for j in range(d_k)]
    return scores, weights, context


def main():
    title("09 · The Three Kinds of Attention")
    section("Example 1 (simplest): cross-attention — the decoder is about to write 'Katze'")
    words = ["the", "cat", "sat"]
    memory = [[0, 1], [2, 0], [1, 2]]
    query = [2, 0]
    scores, weights, context = cross_attention(query, memory)
    for w, m, s, a in zip(words, memory, scores, weights):
        work(f"score {w}", f"({dot_work(query, m)}) ÷ √2 = {fmt(s)}   weight {fmt(a)}")
    work("context vector", f"{[fmt(c) for c in context]}  — mostly 'cat'")

    section("Example 2 (moderate): shapes and counts in the base model")
    n, m, N, h = 5, 4, 6, 8
    for name, rows, cols in [("encoder self-attn", n, n), ("masked decoder self-attn", m, m),
                             ("encoder–decoder attn", m, n)]:
        work(name, f"score matrix {rows} × {cols} = {rows * cols} entries per head; {N} blocks, {N * h} heads")
    work("TOTAL", f"{3 * N} attention blocks, {3 * N * h} heads")


if __name__ == "__main__":
    main()
