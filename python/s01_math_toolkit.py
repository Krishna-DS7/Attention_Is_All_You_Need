"""Sheet 01 · The Math Toolkit — every piece of math the paper uses, from zero.

BIG IDEA: the whole Transformer is built from a handful of operations:
multiply-and-add (dot products), tables of numbers (matrices), turning scores
into percentages (softmax) and a few averages.
"""
import math
from common import (title, section, work, show_matrix, show_vector, dot, dot_work, length, cosine,
                    transpose, matmul, softmax_steps, softmax, mean, variance, relu, fmt)


def tool_dot_product():
    section("Tool 2 · Dot product — 'how much do two vectors agree?'   a·b = a1×b1 + a2×b2 + …")
    a, b = [1, 2], [3, 4]
    work("simplest", "a·b = " + dot_work(a, b))

    words = {"king": [0.9, 0.8, 0.1, 0.1], "queen": [0.85, 0.75, 0.1, 0.9], "apple": [0.0, 0.1, 0.95, 0.2]}
    print("\n  Moderate: cosine similarity = dot ÷ (length × length), from -1 (opposite) to +1 (same direction)")
    for x, y in [("king", "queen"), ("king", "apple"), ("queen", "apple")]:
        c = cosine(words[x], words[y])
        work(f"{x}·{y}", f"dot {fmt(dot(words[x], words[y]))}, lengths {fmt(length(words[x]))} & "
                         f"{fmt(length(words[y]))} → cosine {fmt(c)}  ({'similar' if c > 0.5 else 'not similar'})")


def tool_matrices():
    section("Tools 3 & 4 · Transpose and matrix multiplication")
    K = [[1, 2, 3], [4, 5, 6]]
    show_matrix("K (2×3)", K)
    show_matrix("Kᵀ (3×2) — rows become columns", transpose(K))

    A, B = [[1, 2], [3, 4]], [[5, 6], [7, 8]]
    C = matmul(A, B)
    show_matrix("A · B", C)
    Bt = transpose(B)
    for i in range(2):
        for j in range(2):
            work(f"cell ({i+1},{j+1})", dot_work(A[i], Bt[j]))

    print("\n  Moderate: 3 words × 4 features, times a 4×2 'lens' → 3 words × 2 new features (the shape of Q = X·W_Q)")
    X = [[1, 0, 2, 1], [0, 1, 1, 0], [2, 1, 0, 1]]
    W = [[0.5, 1], [1, 0], [0, 0.5], [1, -1]]
    XW = matmul(X, W)
    show_matrix("X · W", XW, ["word 1", "word 2", "word 3"], ["new 1", "new 2"])
    for i in range(3):
        for j in range(2):
            work(f"word {i+1}, new {j+1}", dot_work(X[i], transpose(W)[j]))
    return XW


def tool_softmax():
    section("Tools 5 & 6 · e^x and softmax — scores → percentages that add up to 1")
    for x in [-2, -1, 0, 1, 2, 5, 10]:
        work(f"e^{x}", fmt(math.exp(x)))
    scores = [2, 1, 0.1]
    exps, total, weights = softmax_steps(scores)
    print()
    for k in range(3):
        work(f"item {k+1}", f"e^{fmt(scores[k])} ÷ ({' + '.join(fmt(e) for e in exps)}) = "
                            f"{fmt(exps[k])} ÷ {fmt(total)} = {fmt(weights[k])}")
    work("check: sum", fmt(sum(weights)))

    print("\n  Moderate: three experiments")
    work("original", [fmt(w) for w in weights])
    work("(a) + 100", [fmt(w) for w in softmax([s + 100 for s in scores])] + ["identical: only differences matter"])
    work("(b) × 10", [fmt(w) for w in softmax([s * 10 for s in scores])] + ["too peaky"])
    work("(c) ÷ 10", [fmt(w) for w in softmax([s / 10 for s in scores])] + ["too flat"])
    return weights


def tool_statistics():
    section("Tool 7 · Mean, variance, standard deviation — and why √ appears")
    data = [2, 4, 6, 8]
    m, var = mean(data), variance(data)
    sq = [(x - m) ** 2 for x in data]
    work("mean", f"({' + '.join(map(str, data))}) ÷ 4 = {fmt(m)}")
    work("variance", f"({' + '.join(fmt(s) for s in sq)}) ÷ 4 = {fmt(var)};  std = √{fmt(var)} = {fmt(math.sqrt(var))}")
    z = [(x - m) / math.sqrt(var) for x in data]
    work("z-scores", f"{[fmt(v) for v in z]}  → mean {fmt(mean(z))}, variance {fmt(variance(z))}")
    print("\n  A sum of d independent numbers of variance 1 has typical size √d:")
    for d in [1, 4, 64, 512]:
        work(f"d = {d}", f"√d = {fmt(math.sqrt(d))}")


def tool_relu_and_log():
    section("Tools 8 & 9 · ReLU = max(0, x)  and  surprise = −ln(p)")
    for x in [-2, -0.5, 0, 1.5, 3]:
        work(f"ReLU({x})", fmt(relu(x)))
    for p in [1, 0.9, 0.5, 0.1, 0.01]:
        work(f"−ln({p})", fmt(-math.log(p)))


def main():
    title("01 · The Math Toolkit")
    tool_dot_product()
    tool_matrices()
    tool_softmax()
    tool_statistics()
    tool_relu_and_log()


if __name__ == "__main__":
    main()
