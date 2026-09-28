"""Sheet 04 · Positional Encoding (paper §3.5).

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

BIG IDEA: attention can't see word order, so a unique pattern of waves is added
to each position — like clock hands turning at different speeds.
"""
import math
from common import title, section, work, show_matrix, add, dot, fmt
from s03_embeddings import embed, SENTENCE, D_MODEL


def positional_encoding(pos, d_model, base=10000):
    """The paper's formula for one position: a list of d_model numbers."""
    pe = []
    for dim in range(d_model):
        i = dim // 2                              # which sin/cos pair
        angle = pos / base ** (2 * i / d_model)
        pe.append(math.sin(angle) if dim % 2 == 0 else math.cos(angle))
    return pe


def encoder_input(tokens=SENTENCE, d_model=D_MODEL):
    """X = √d_model · E[token] + PE[pos]  — the input to the first encoder layer."""
    return add(embed(tokens, d_model), [positional_encoding(p, d_model) for p in range(len(tokens))])


def main():
    title("04 · Positional Encoding")
    section("Example 1 (simplest): d_model = 4, positions 0–5")
    d = 4
    work("divisor pair i=0", f"10000^(0/4) = {fmt(10000 ** 0)}")
    work("divisor pair i=1", f"10000^(2/4) = {fmt(10000 ** 0.5)}")
    table = [positional_encoding(p, d) for p in range(6)]
    show_matrix("PE (rows = positions)", table, [f"pos {p}" for p in range(6)])
    work("pos 2, dim 1", f"sin(2 / 1) = sin(2 radians) = {fmt(table[2][0])}")
    work("pos 2, dim 3", f"sin(2 / 100) = sin(0.02 radians) = {fmt(table[2][2])}")

    section("Example 2 (moderate): X = √d·E + PE for 'the cat sat' (feeds scripts 05, 08, 10, 11)")
    E = embed(SENTENCE)
    X = encoder_input()
    show_matrix("X (encoder input)", X, SENTENCE)
    work("cat, dim 2", f"{fmt(E[1][1])} (meaning) + {fmt(positional_encoding(1, 4)[1])} (position 1) = {fmt(X[1][1])}")

    section("Example 3 (moderate): d_model = 8 — wavelengths (positions per full cycle)")
    for i in range(4):
        div = 10000 ** (2 * i / 8)
        work(f"pair i={i}", f"divisor {fmt(div)}, wavelength 2π × divisor = {2 * math.pi * div:,.1f}")

    section("Example 4 (moderate): shifting by k positions is a rotation (depends on k only)")
    p, k, w = 3, 2, 1.0
    direct = (math.sin(w * (p + k)), math.cos(w * (p + k)))
    sp, cp, sk, ck = math.sin(w * p), math.cos(w * p), math.sin(w * k), math.cos(w * k)
    rotated = (sp * ck + cp * sk, cp * ck - sp * sk)
    work("sin component", f"direct {direct[0]:.6f}   via rotation {rotated[0]:.6f}")
    work("cos component", f"direct {direct[1]:.6f}   via rotation {rotated[1]:.6f}")

    print("\n  Bonus: PE(p)·PE(p+k) depends only on the distance k (d_model = 8):")
    for k in range(1, 6):
        a = dot(positional_encoding(0, 8), positional_encoding(k, 8))
        b = dot(positional_encoding(5, 8), positional_encoding(5 + k, 8))
        work(f"k = {k}", f"PE(0)·PE({k}) = {a:.4f}   PE(5)·PE({5+k}) = {b:.4f}")


if __name__ == "__main__":
    main()
