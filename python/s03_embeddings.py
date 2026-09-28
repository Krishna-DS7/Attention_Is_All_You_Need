"""Sheet 03 · Tokens & Embeddings (paper §3.4, §5.1).

BIG IDEA: each token is replaced by a learned row of numbers from a lookup
table, then multiplied by √d_model. One table is shared by the encoder input,
the decoder input and the output layer.
"""
import math
from common import title, section, work, show_matrix, cosine, fmt

D_MODEL = 4
VOCAB = ["<s>", "</s>", "the", "cat", "sat", "die", "Katze", "saß"]
# Made-up embeddings; 'cat'≈'Katze', 'the'≈'die', 'sat'≈'saß' on purpose.
E = [[0.1, -0.1, 0.0, 0.2],
     [-0.1, 0.1, 0.2, 0.0],
     [0.5, -0.2, 0.1, 0.0],
     [-0.3, 0.8, 0.4, -0.1],
     [0.2, 0.1, -0.6, 0.5],
     [0.4, -0.3, 0.2, 0.1],
     [-0.2, 0.7, 0.5, -0.2],
     [0.3, 0.0, -0.5, 0.6]]
SENTENCE = ["the", "cat", "sat"]     # the running example used by sheets/scripts 04–11


def lookup(token):
    """Step 1: find the token's row in the table."""
    return E[VOCAB.index(token)]


def embed(tokens, d_model=D_MODEL):
    """Step 2: look up each token and multiply by √d_model."""
    return [[v * math.sqrt(d_model) for v in lookup(t)] for t in tokens]


def main():
    title("03 · Tokens & Embeddings")
    show_matrix("Embedding table E (8 tokens × d_model 4)", E, VOCAB)

    section("Example 1 (simplest): embed 'the cat sat'")
    show_matrix("Step 1 — look up rows", [lookup(t) for t in SENTENCE], SENTENCE)
    scaled = embed(SENTENCE)
    show_matrix("Step 2 — × √d_model = × 2", scaled, SENTENCE)
    row = VOCAB.index("cat") + 1
    work("show the work", f"'cat' is row {row}; dim 2 = {fmt(lookup('cat')[1])} × √4 = {fmt(scaled[1][1])}")

    section("Example 2 (moderate): which tokens are alike? (cosine similarity)")
    for a, b in [("cat", "Katze"), ("the", "die"), ("sat", "saß"), ("cat", "sat"), ("the", "Katze")]:
        c = cosine(lookup(a), lookup(b))
        verdict = "very similar" if c > 0.8 else ("somewhat similar" if c > 0.3 else "different")
        work(f"{a} vs {b}", f"{fmt(c)}  {verdict}")

    section("Example 3 (moderate): the real table and what weight sharing saves")
    V, d = 37000, 512
    one = V * d
    work("params in one table", f"{V:,} × {d} = {one:,}")
    work("memory (4-byte floats)", f"{one * 4 / 1024**2:,.1f} MB")
    work("saved by sharing", f"2 × {one:,} = {2 * one:,} (one table used 3 times)")
    work("share of 65M model", f"{one / 65e6:.1%}")


if __name__ == "__main__":
    main()
