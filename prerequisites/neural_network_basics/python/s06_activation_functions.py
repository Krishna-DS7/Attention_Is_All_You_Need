"""Sheet 06 · Activation functions — the 'bend' that gives networks their power.

    sigmoid' = s(1 − s)      tanh' = 1 − t²      ReLU' = 1 if z > 0 else 0
"""
import math
from common import title, section, work, ACTIVATIONS, sigmoid, fmt


def slopes(z):
    s, t = sigmoid(z), math.tanh(z)
    return s * (1 - s), 1 - t * t, 1.0 if z > 0 else 0.0


def main():
    title("06 · Activation Functions")
    section("Example 1 (simplest): scores −2, 0, 3 through every activation")
    for name, f in ACTIVATIONS.items():
        work(name, "   ".join(f"f({z}) = {f(z):.3f}" for z in (-2, 0, 3)))
    work("sigmoid(−2)", f"1 ÷ (1 + e^2) = 1 ÷ (1 + {fmt(math.exp(2))}) = {fmt(sigmoid(-2))}")

    section("Example 2 (moderate): the curves and their slopes (≈0 slope = saturated, stops learning)")
    print(f"  {'z':>5} " + "".join(f"{n:>9}" for n in ACTIVATIONS) + f"{'σ slope':>9}{'tanh sl.':>9}{'ReLU sl.':>9}")
    for k in range(-8, 9):
        z = k / 2
        print(f"  {z:>5} " + "".join(f"{f(z):>9.3f}" for f in ACTIVATIONS.values())
              + "".join(f"{v:>9.3f}" for v in slopes(z)))


if __name__ == "__main__":
    main()
