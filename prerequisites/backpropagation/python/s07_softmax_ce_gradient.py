"""Sheet 07 · Softmax + cross-entropy gives the gradient  dL/dz = p − y.

The same shortcut holds for sigmoid + binary cross-entropy.
"""
import math
from common import title, section, work, softmax, sigmoid, bce, fmt


def ce_loss(z, correct):
    return -math.log(softmax(z)[correct])


def ce_gradient(z, correct):
    """Analytic gradient: p − one_hot(correct)."""
    return [p - (1 if i == correct else 0) for i, p in enumerate(softmax(z))]


def numeric_gradient(z, correct, h=1e-4):
    grads = []
    for i in range(len(z)):
        up = [v + h if j == i else v for j, v in enumerate(z)]
        down = [v - h if j == i else v for j, v in enumerate(z)]
        grads.append((ce_loss(up, correct) - ce_loss(down, correct)) / (2 * h))
    return grads


def main():
    title("07 · Softmax + Cross-Entropy Gradient")
    section("Example 1 (simplest): scores [2, 1, 0.1], correct class = cat")
    z, labels = [2, 1, 0.1], ["cat ✔", "dog", "bird"]
    p = softmax(z)
    work("p = softmax(z)", [fmt(v) for v in p])
    work("loss", f"−ln {fmt(p[0])} = {fmt(ce_loss(z, 0))}")
    for lab, g, n in zip(labels, ce_gradient(z, 0), numeric_gradient(z, 0)):
        work(lab, f"p − y = {g:.6f}   numeric {n:.6f}")

    section("Example 2 (moderate): the binary case, sigmoid + BCE at z = 0.3, y = 1")
    zz, y, h = 0.3, 1, 1e-4
    num = (bce(sigmoid(zz + h), y) - bce(sigmoid(zz - h), y)) / (2 * h)
    work("p − y", f"{sigmoid(zz) - y:.6f}   numeric {num:.6f}")


if __name__ == "__main__":
    main()
