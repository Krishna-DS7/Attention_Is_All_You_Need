"""Sheet 08 · Why layers and activations? — the XOR problem.

One neuron draws one straight line, so it can't compute XOR. Two hidden ReLU
neurons plus one output neuron can:
    h1 = ReLU(x1 + x2)      h2 = ReLU(x1 + x2 − 1)      y = h1 − 2·h2
"""
from common import title, section, work, relu, step

CASES = [(0, 0), (0, 1), (1, 0), (1, 1)]


def xor(x1, x2):
    return x1 ^ x2


def single_neuron_score(w1, w2, b):
    """How many of the 4 XOR cases one step-neuron gets right (never 4)."""
    return sum(step(w1 * x1 + w2 * x2 + b) == xor(x1, x2) for x1, x2 in CASES)


def best_single_neuron(grid=(-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2)):
    return max(single_neuron_score(w1, w2, b) for w1 in grid for w2 in grid for b in grid)


def xor_network(x1, x2):
    """Returns (z1, h1, z2, h2, y)."""
    z1 = x1 + x2
    z2 = x1 + x2 - 1
    h1, h2 = relu(z1), relu(z2)
    return z1, h1, z2, h2, h1 - 2 * h2


def main():
    title("08 · Why Layers (XOR)")
    section("Example 1 (simplest): one neuron can't do it")
    work("w=(1,1), b=−0.5", f"{single_neuron_score(1, 1, -0.5)} / 4 correct")
    work("best of 729 tries", f"{best_single_neuron()} / 4 correct — never 4")

    section("Example 2 (moderate): a 2 → 2 → 1 network solves it")
    for x1, x2 in CASES:
        z1, h1, z2, h2, y = xor_network(x1, x2)
        work(f"input ({x1}, {x2})", f"z1 = {z1} → h1 = {h1};  z2 = {z2} → h2 = {h2};  y = {h1} − 2×{h2} = {y}"
                                   f"   target {xor(x1, x2)} {'✔' if y == xor(x1, x2) else '✘'}")


if __name__ == "__main__":
    main()
