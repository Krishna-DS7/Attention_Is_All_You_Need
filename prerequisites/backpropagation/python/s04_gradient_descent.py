"""Sheet 04 · Gradient descent:  w ← w − η · dL/dw."""
from common import title, section, work


def descend_bowl(lr, w=0.0, steps=13):
    """Minimise L(w) = (w − 3)²; slope 2(w − 3). Returns the list of w values."""
    ws = []
    for _ in range(steps):
        ws.append(w)
        w = w - lr * 2 * (w - 3)
    return ws


def fit_line(lr=0.05, iterations=30):
    """Fit y = w·x + b to points on y = 2x + 1. Returns rows (iter, w, b, loss, dL/dw, dL/db)."""
    X, Y = [0, 1, 2, 3], [1, 3, 5, 7]
    w = b = 0.0
    rows = []
    for k in range(iterations):
        errs = [w * x + b - y for x, y in zip(X, Y)]
        loss = sum(e * e for e in errs) / 4
        gw = 2 * sum(e * x for e, x in zip(errs, X)) / 4
        gb = 2 * sum(errs) / 4
        rows.append((k, w, b, loss, gw, gb))
        w, b = w - lr * gw, b - lr * gb
    return rows


def main():
    title("04 · Gradient Descent")
    section("Example 1 (simplest): L(w) = (w − 3)² with three learning rates")
    runs = {lr: descend_bowl(lr) for lr in (0.1, 0.4, 1.1)}
    for k in range(13):
        work(f"step {k}", "   ".join(f"η {lr}: w {ws[k]:9.4f} loss {(ws[k] - 3) ** 2:10.4f}" for lr, ws in runs.items()))

    section("Example 2 (moderate): fitting y = w·x + b to points on y = 2x + 1")
    for k, w, b, L, gw, gb in fit_line():
        if k in (0, 1, 2, 5, 10, 20, 29):
            work(f"iter {k}", f"w {w:.4f}  b {b:.4f}  loss {L:.4f}  ∂L/∂w {gw:.4f}  ∂L/∂b {gb:.4f}")


if __name__ == "__main__":
    main()
