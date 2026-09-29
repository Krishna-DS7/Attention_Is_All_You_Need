"""Sheet 10 · Vanishing and exploding gradients.

The gradient reaching layer 1 ≈ product over layers of (weight × activation slope).
"""
import math
from common import title, section, work


def gradient_after(layers, factor):
    return factor ** layers


def main():
    title("10 · Vanishing Gradients")
    w, w_big, z = 1.0, 1.5, 1.0
    factors = {"sigmoid (≤ 0.25)": w * 0.25, "tanh at z = 1": w * (1 - math.tanh(z) ** 2),
               "ReLU active": w * 1.0, "ReLU, w = 1.5": w_big * 1.0, "residual path": 1.0}
    section("Gradient left after L layers")
    for L in [1, 2, 3, 5, 10, 15, 20, 30]:
        work(f"{L} layers", "   ".join(f"{k}: {gradient_after(L, f):.2e}" for k, f in factors.items()))


if __name__ == "__main__":
    main()
