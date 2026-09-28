"""Sheet 08 · Multi-Head Attention (paper §3.2.2).

    MultiHead(Q, K, V) = Concat(head_1, …, head_h) · W_O
    head_i = Attention(X·W_Q_i, X·W_K_i, X·W_V_i)

BIG IDEA: h smaller attentions run in parallel, each with its own 'lens'; their
outputs are placed side by side and mixed by W_O.
"""
from common import title, section, work, show_matrix, matmul, dot_work, transpose, fmt
from s03_embeddings import SENTENCE
from s04_positional_encoding import encoder_input
from s05_query_key_value import W_Q, W_K, W_V
from s06_scaled_attention import attention

W_O = [[1, 0, 0, 0.5], [0, 1, 0.5, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]]


def columns(M, start, stop):
    """Take columns start..stop-1 of a matrix (head i's slice of the weights)."""
    return [row[start:stop] for row in M]


def multi_head(X, h=2, verbose=False):
    """Split the projections into h heads of size d_k = d_model / h, attend, concat, mix."""
    d_model = len(X[0])
    d_k = d_model // h
    heads = []
    for i in range(h):
        s, e = i * d_k, (i + 1) * d_k
        Qi = matmul(X, columns(W_Q, s, e))
        Ki = matmul(X, columns(W_K, s, e))
        Vi = matmul(X, columns(W_V, s, e))
        weights, out = attention(Qi, Ki, Vi)
        heads.append(out)
        if verbose:
            show_matrix(f"Head {i+1} weights", weights, SENTENCE, SENTENCE)
            show_matrix(f"head{i+1} = A·V{i+1}", out, SENTENCE, ["a", "b"])
            work(f"head{i+1}[cat, a]", dot_work(weights[1], transpose(Vi)[0]))
    concat = [sum((head[r] for head in heads), []) for r in range(len(X))]
    return concat, matmul(concat, W_O)


def main():
    title("08 · Multi-Head Attention")
    section("Example 1 (simplest real case): 'the cat sat', d_model = 4, h = 2, d_k = 2")
    X = encoder_input()
    concat, out = multi_head(X, h=2, verbose=True)
    show_matrix("Concat(head1, head2)", concat, SENTENCE, ["h1.a", "h1.b", "h2.a", "h2.b"])
    show_matrix("MultiHead output = Concat · W_O   (feeds script 10)", out, SENTENCE)
    for j, col in enumerate(transpose(W_O)):
        work(f"Out[cat, dim {j+1}]", dot_work(concat[1], col))

    section("Example 2 (moderate): the base model — parameters and cost")
    d_model, h, n = 512, 8, 30
    d_k = d_model // h
    qkv = 3 * h * d_model * d_k
    wo = h * d_k * d_model
    work("d_k", f"{d_model} / {h} = {d_k}")
    work("W_Q+W_K+W_V", f"3 × {h} × {d_model} × {d_k} = {qkv:,}")
    work("W_O", f"{h} × {d_k} × {d_model} = {wo:,}")
    work("total", f"{qkv + wo:,} = 4 × d_model²")
    work("score mults, 8 heads", f"{h} × {n}² × {d_k} = {h * n * n * d_k:,}")
    work("score mults, 1 head", f"{n}² × {d_model} = {n * n * d_model:,}  (same cost)")

    section("Table 3 rows (A)/(B) from the paper — EN→DE dev set")
    for name, hh, dk, dv, ppl, bleu in [("base", 8, 64, 64, 4.92, 25.8), ("(A)", 1, 512, 512, 5.29, 24.9),
                                         ("(A)", 4, 128, 128, 5.00, 25.5), ("(A)", 16, 32, 32, 4.91, 25.8),
                                         ("(A)", 32, 16, 16, 5.01, 25.4), ("(B)", 8, 16, 64, 5.16, 25.1),
                                         ("(B)", 8, 32, 64, 5.01, 25.4)]:
        work(name, f"h={hh:<3} d_k={dk:<4} d_v={dv:<4} PPL {ppl:.2f}  BLEU {bleu}")


if __name__ == "__main__":
    main()
