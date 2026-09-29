"""Sheet 05 · The linear layer — many neurons at once:  y = x·W + b.

Also shows why stacking linear layers WITHOUT activations collapses into one layer:
(x·W1 + b1)·W2 + b2 = x·(W1·W2) + (b1·W2 + b2).
"""
from common import title, section, work, show_vector, linear, matmul, transpose, wsum_work, fmt


def collapse(W1, b1, W2, b2):
    """The single linear layer equivalent to two stacked linear layers."""
    return matmul(W1, W2), linear(b1, W2, b2)


def main():
    title("05 · The Linear Layer")
    section("Example 1 (simplest): 3 inputs → 2 neurons")
    x = [1, 2, 3]
    W = [[0.5, -1], [1, 0], [0, 2]]
    b = [0.5, -1]
    y = linear(x, W, b)
    for j, col in enumerate(transpose(W)):
        work(f"neuron {j+1}", wsum_work(x, col, b[j]))
    show_vector("y = x·W + b", y)

    section("Example 2 (moderate): two linear layers collapse into one")
    W2, b2 = [[1, 2], [-1, 0.5]], [0, 1]
    two_layers = linear(y, W2, b2)
    Wc, bc = collapse(W, b, W2, b2)
    one_layer = linear(x, Wc, bc)
    work("two layers", [fmt(v) for v in two_layers])
    work("one layer", [fmt(v) for v in one_layer])
    work("identical?", str(all(abs(a - c) < 1e-12 for a, c in zip(two_layers, one_layer))))

    section("Example 3: parameters of a 512 → 2048 layer")
    work("weights + biases", f"512 × 2048 + 2048 = {512 * 2048 + 2048:,}")


if __name__ == "__main__":
    main()
