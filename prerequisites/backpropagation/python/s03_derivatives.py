"""Sheet 03 · Derivatives — slopes by nudging and by rules.

f'(x) ≈ (f(x + h) − f(x − h)) / 2h     (numeric check)
"""
import math
from common import title, section, work, numeric_derivative, sigmoid, relu, fmt

# (name, f, exact derivative)
RULES = [
    ("x²", lambda x: x ** 2, lambda x: 2 * x),
    ("x³", lambda x: x ** 3, lambda x: 3 * x ** 2),
    ("eˣ", math.exp, math.exp),
    ("ln x", math.log, lambda x: 1 / x),
    ("sigmoid", sigmoid, lambda x: sigmoid(x) * (1 - sigmoid(x))),
    ("tanh", math.tanh, lambda x: 1 - math.tanh(x) ** 2),
    ("ReLU", relu, lambda x: 1.0 if x > 0 else 0.0),
]


def partials(w, b):
    """f(w, b) = (2w + b − 5)²  →  (∂f/∂w, ∂f/∂b) by the chain rule."""
    inner = 2 * w + b - 5
    return 2 * inner * 2, 2 * inner * 1


def main():
    title("03 · Derivatives and Slopes")
    section("Example 1 (simplest): f(x) = x² at x = 3 — shrink the nudge, the slope settles at 6")
    for h in [1, 0.1, 0.01, 0.001]:
        work(f"h = {h}", f"(({3 + h})² − 9) ÷ {h} = {((3 + h) ** 2 - 9) / h:.6f}")
    work("exact", "2x = 6")

    section("Example 2 (moderate): each derivative rule checked numerically at x = 0.7")
    for name, f, d in RULES:
        exact, num = d(0.7), numeric_derivative(f, 0.7)
        work(name, f"exact {exact:.6f}   numeric {num:.6f}   {'✔' if abs(exact - num) < 1e-6 else '✘'}")

    section("Example 3 (moderate): partial derivatives and the gradient of (2w + b − 5)² at w = b = 1")
    gw, gb = partials(1, 1)
    f = lambda w, b: (2 * w + b - 5) ** 2
    work("∂f/∂w", f"{fmt(gw)}   numeric {numeric_derivative(lambda w: f(w, 1), 1):.6f}")
    work("∂f/∂b", f"{fmt(gb)}   numeric {numeric_derivative(lambda b: f(1, b), 1):.6f}")
    work("gradient", f"[{fmt(gw)}, {fmt(gb)}] → step direction [{fmt(-gw)}, {fmt(-gb)}]")


if __name__ == "__main__":
    main()
