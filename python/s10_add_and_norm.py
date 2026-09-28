"""Sheet 10 · Add & Norm (paper §3.1).

    output = LayerNorm( x + Sublayer(x) )
    LayerNorm(v) = γ · (v − mean) / √(variance + ε) + β        (per word, across its own numbers)
"""
import math
from common import title, section, work, show_matrix, show_vector, add, mean, variance, fmt
from s03_embeddings import SENTENCE
from s04_positional_encoding import encoder_input
from s08_multi_head import multi_head

EPS = 1e-6       # the paper does not state ε; 1e-6 is a common choice


def layer_norm(v, gamma=None, beta=None, eps=EPS):
    gamma = gamma or [1.0] * len(v)
    beta = beta or [0.0] * len(v)
    mu, var = mean(v), variance(v)
    sd = math.sqrt(var + eps)
    return [g * (x - mu) / sd + b for x, g, b in zip(v, gamma, beta)]


def add_and_norm(x_rows, sub_rows):
    """Residual add, then LayerNorm each word (row) separately."""
    return [layer_norm(row) for row in add(x_rows, sub_rows)]


def main():
    title("10 · Add & Norm")
    section("Example 1 (simplest): one word, d_model = 4")
    x, sub = [1, 2, 3, 4], [0.5, -1, 0, 2.5]
    v = add(x, sub)
    mu, var = mean(v), variance(v)
    sd = math.sqrt(var + EPS)
    work("ADD", f"x + Sublayer(x) = {[fmt(a) for a in v]}")
    work("mean", f"({' + '.join(fmt(a) for a in v)}) ÷ 4 = {fmt(mu)}")
    work("variance", f"{fmt(var)};  √(σ² + ε) = {fmt(sd)}")
    out = layer_norm(v)
    work("dim 4", f"({fmt(v[3])} − {fmt(mu)}) ÷ {fmt(sd)} = {fmt(out[3])}")
    show_vector("LayerNorm output (γ = 1, β = 0)", out)
    work("check", f"mean {fmt(mean(out))}, variance {fmt(variance(out))}")

    section("Example 2 (moderate): Add & Norm after multi-head attention for 'the cat sat'")
    X = encoder_input()
    _, MH = multi_head(X)
    LN1 = add_and_norm(X, MH)
    show_matrix("LayerNorm(X + MultiHead(X))   (feeds script 11)", LN1, SENTENCE)

    section("Example 3 (moderate, illustration): why residual connections help deep stacks")
    for L in [1, 2, 6, 12, 24]:
        work(f"{L} sub-layers", f"plain stack 0.5^{L} = {0.5 ** L:.6f}   residual highway = 1.0")


if __name__ == "__main__":
    main()
