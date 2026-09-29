"""Sheet 07 · Softmax — a list of scores → probabilities that sum to 1.

    softmax(z_i) = e^(z_i) / Σ_j e^(z_j)
"""
import math
from common import title, section, work, softmax_steps, sigmoid, fmt


def stable_softmax(z):
    """Subtract the max first: same answer, no overflow."""
    m = max(z)
    return softmax_steps([v - m for v in z])[2]


def softmax_with_temperature(z, T):
    return stable_softmax([v / T for v in z])


def main():
    title("07 · Softmax")
    section("Example 1 (simplest): scores [2, 1, 0.1] for cat / dog / bird")
    labels, z = ["cat", "dog", "bird"], [2, 1, 0.1]
    exps, total, p = softmax_steps(z)
    for lab, zi, e, pi in zip(labels, z, exps, p):
        work(lab, f"e^{zi} ÷ {fmt(total)} = {fmt(e)} ÷ {fmt(total)} = {fmt(pi)}")
    work("prediction", labels[p.index(max(p))])

    section("Example 2 (moderate): three properties")
    big = [1000, 999, 998]
    try:
        softmax_steps(big)
        naive = "worked"
    except OverflowError:
        naive = "overflow!"
    work("(a) naive", naive)
    work("(a) subtract max", [f"{v:.1%}" for v in stable_softmax(big)])
    for T in (0.5, 1, 2, 10):
        work(f"(b) T = {T}", [f"{v:.1%}" for v in softmax_with_temperature(z, T)])
    a, b = 2, 0.5
    work("(c) 2-class", f"softmax P(A) = {math.exp(a) / (math.exp(a) + math.exp(b)):.6f}   "
                        f"sigmoid(A − B) = {sigmoid(a - b):.6f}")


if __name__ == "__main__":
    main()
