"""Sheet 04 · The neuron.

    z = w · x + b          output = f(z)
"""
from common import title, section, work, dot, step, sigmoid, wsum_work, fmt


def neuron(x, w, b, activation):
    z = dot(x, w) + b
    return z, activation(z)


def main():
    title("04 · The Neuron")
    section("Example 1 (simplest): 'Should I go for a walk?'  inputs [sunny, warm, busy]")
    w, b = [2, 1, -3], -1.5
    for x in ([1, 1, 0], [1, 1, 1]):
        z, fired = neuron(x, w, b, step)
        work(f"x = {x}", f"z = {wsum_work(x, w, b)}  →  step {fmt(fired)}, sigmoid {sigmoid(z):.0%}")

    section("Example 2 (moderate): one neuron as a logic gate (w1 = w2 = 1, step activation)")
    for name, bias in [("AND", -1.5), ("OR", -0.5)]:
        outs = [int(neuron([x1, x2], [1, 1], bias, step)[1]) for x1 in (0, 1) for x2 in (0, 1)]
        work(f"{name} (b = {bias})", f"inputs 00, 01, 10, 11 → {outs};  boundary x2 = {fmt(-bias)} − x1")


if __name__ == "__main__":
    main()
