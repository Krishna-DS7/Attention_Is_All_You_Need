"""Sheet 13 · Cross-entropy loss and label smoothing (paper §5.4, Table 3).

    loss = − Σ q(token) · ln p(token)
    label smoothing: q(right) = (1 − ε) + ε/K,   q(other) = ε/K       (ε = 0.1)
    perplexity = e^(average loss)
"""
import math
from common import title, section, work, fmt


def smoothed_target(correct, K, eps=0.1):
    return [(1 - eps) + eps / K if k == correct else eps / K for k in range(K)]


def cross_entropy(q, p):
    return -sum(qi * math.log(pi) for qi, pi in zip(q, p) if qi > 0)


def perplexity(p_correct):
    """e^(average −ln p) — the geometric mean of 1/p."""
    return math.exp(sum(-math.log(p) for p in p_correct) / len(p_correct))


def main():
    title("13 · Loss and Label Smoothing")
    section("Example 1 (simplest): vocabulary [die, Katze, Hund, saß], correct = Katze")
    K, correct = 4, 1
    hard = [1.0 if k == correct else 0.0 for k in range(K)]
    soft = smoothed_target(correct, K)
    work("smoothed target", [fmt(v) for v in soft])
    models = {"A over-confident, right": [0.001, 0.997, 0.001, 0.001],
              "B calibrated, right": [0.03, 0.91, 0.03, 0.03],
              "C over-confident, WRONG": [0.001, 0.001, 0.997, 0.001]}
    for name, p in models.items():
        work(name, f"hard loss {cross_entropy(hard, p):.4f}   smoothed loss {cross_entropy(soft, p):.4f}")
    p = models["B calibrated, right"]
    work("B, smoothed", "−" + " − ".join(f"{fmt(q)}×ln({fmt(pi)})" for q, pi in zip(soft, p))
         + f" = {fmt(cross_entropy(soft, p))}")
    work("best possible", f"{cross_entropy(soft, soft):.4f} (only when p = q)")

    section("Example 2 (moderate): perplexity over a sentence")
    pc = [0.6, 0.3, 0.9, 0.8]
    losses = [-math.log(x) for x in pc]
    work("losses", [fmt(l) for l in losses])
    work("average", fmt(sum(losses) / len(losses)))
    work("perplexity", fmt(perplexity(pc)))
    for name, ppl in [("base", 4.92), ("big", 4.33), ("(A) 1 head", 5.29), ("(D) no dropout", 5.77)]:
        work(f"paper {name}", f"PPL {ppl} → avg loss {math.log(ppl):.3f} → geometric p(correct) {1 / ppl:.1%}")
    K = 37000
    work("real smoothing", f"right token {1 - 0.1 + 0.1 / K:.7f}, each other {0.1 / K:.2e}")


if __name__ == "__main__":
    main()
