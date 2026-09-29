"""Sheet 01 · The learning loop: forward → loss → gradient → update → repeat.

Model y = w·x on data that follow y = 2x. The slope of the MSE loss is
dL/dw = mean( 2·(w·x − y)·x ).
"""
from common import title, section, work, fmt

X = [1, 2, 3]
Y = [2, 4, 6]


def loss(w):
    return sum((w * x - y) ** 2 for x, y in zip(X, Y)) / len(X)


def slope(w):
    return sum(2 * (w * x - y) * x for x, y in zip(X, Y)) / len(X)


def train(w=0.5, lr=0.05, iterations=8):
    """Yields (iteration, w, predictions, loss, slope, new_w)."""
    for k in range(iterations):
        g = slope(w)
        new_w = w - lr * g
        yield k, w, [w * x for x in X], loss(w), g, new_w
        w = new_w


def main():
    title("01 · The Learning Loop")
    section("Example 1 (simplest): the loss landscape — minimum at w = 2")
    for k in range(9):
        w = k * 0.5
        work(f"w = {w}", f"loss = {loss(w):.3f}  " + "#" * int(loss(w) * 2))

    section("Example 2 (moderate): run the loop from w = 0.5 with η = 0.05")
    for k, w, preds, L, g, new_w in train():
        work(f"iteration {k}", f"w {w:.4f}  preds {[round(p, 3) for p in preds]}  loss {L:.4f}  "
                               f"slope {g:.4f}  → new w {new_w:.4f}")
    work("iteration 0", f"slope = 2×((0.5−2)×1 + (1−4)×2 + (1.5−6)×3) ÷ 3 = {fmt(slope(0.5))}")


if __name__ == "__main__":
    main()
