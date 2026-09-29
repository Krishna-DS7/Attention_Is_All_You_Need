"""Sheet 01 · What is a neural network? — a big adjustable formula.

BIG IDEA: numbers in → numbers out, built from one repeated unit (the neuron):
multiply inputs by weights, add them up, add a bias, bend with an activation.
Learning = finding weight values that make the outputs right.
"""
from common import title, section, work, fmt


def one_neuron_no_activation(x, w, b):
    """y = w·x + b — the straight line from school."""
    return w * x + b


def count_parameters(layer_sizes):
    """Each layer: (inputs × outputs) weights + (outputs) biases."""
    return [n_in * n_out + n_out for n_in, n_out in zip(layer_sizes, layer_sizes[1:])]


def main():
    title("01 · What Is a Neural Network?")
    section("Example 1 (simplest): one neuron, no activation — y = w·x + b with w = 2, b = 1")
    for x in range(5):
        work(f"x = {x}", f"2 × {x} + 1 = {fmt(one_neuron_no_activation(x, 2, 1))}")

    section("Example 2 (moderate): a 3 → 4 → 2 network — how many numbers does it learn?")
    per_layer = count_parameters([3, 4, 2])
    work("layer 1", f"3×4 + 4 = {per_layer[0]}")
    work("layer 2", f"4×2 + 2 = {per_layer[1]}")
    work("total", f"{sum(per_layer)}")
    work("for scale", f"one Transformer FFN (512 → 2048 → 512) = {sum(count_parameters([512, 2048, 512])):,}")


if __name__ == "__main__":
    main()
