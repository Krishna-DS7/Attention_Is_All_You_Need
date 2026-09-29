"""Sheet 09 · End-to-end example 2 — a training loop: one neuron learns OR but not XOR.

p = σ(w1·x1 + w2·x2 + b), mean BCE over the 4 inputs, full-batch gradient descent:
∂L/∂w1 = mean((p − y)·x1),  ∂L/∂w2 = mean((p − y)·x2),  ∂L/∂b = mean(p − y)
"""
from common import title, section, work, sigmoid, bce

INPUTS = [(0, 0), (0, 1), (1, 0), (1, 1)]
TASKS = {"OR": [0, 1, 1, 1], "XOR": [0, 1, 1, 0]}


def train(targets, w1=0.5, w2=-0.3, b=0.1, lr=2.0, epochs=40):
    """Returns rows (epoch, w1, w2, b, probabilities, loss, accuracy) for epochs 0..epochs."""
    rows = []
    for e in range(epochs + 1):
        p = [sigmoid(w1 * x1 + w2 * x2 + b) for x1, x2 in INPUTS]
        loss = sum(bce(pi, y) for pi, y in zip(p, targets)) / 4
        acc = sum((pi >= 0.5) == bool(y) for pi, y in zip(p, targets)) / 4
        rows.append((e, w1, w2, b, p, loss, acc))
        err = [pi - y for pi, y in zip(p, targets)]
        g1 = sum(d * x1 for d, (x1, _) in zip(err, INPUTS)) / 4
        g2 = sum(d * x2 for d, (_, x2) in zip(err, INPUTS)) / 4
        gb = sum(err) / 4
        w1, w2, b = w1 - lr * g1, w2 - lr * g2, b - lr * gb
    return rows


def main():
    title("09 · End-to-End Training Loop")
    for name, targets in TASKS.items():
        section(f"{name}: targets {targets}")
        rows = train(targets)
        for e, w1, w2, b, p, loss, acc in rows:
            if e in (0, 1, 2, 5, 10, 20, 30, 40):
                work(f"epoch {e}", f"w1 {w1:7.4f}  w2 {w2:7.4f}  b {b:7.4f}  loss {loss:.4f}  accuracy {acc:.0%}")
        first = next((e for e, *_, acc in rows if acc == 1), None)
        work("result", f"100% accuracy first reached at epoch {first}" if first is not None
             else f"never above {max(r[6] for r in rows):.0%}; loss stuck near ln 2 = 0.6931")


if __name__ == "__main__":
    main()
