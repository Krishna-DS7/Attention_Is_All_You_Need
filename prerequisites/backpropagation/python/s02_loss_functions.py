"""Sheet 02 · Loss functions — one number that says 'how wrong'.

MSE = mean((pred − target)²)     MAE = mean(|pred − target|)
BCE = −[y ln p + (1 − y) ln(1 − p)]     CE = −ln p(correct class)
"""
import math
from common import title, section, work, bce


def mse(preds, targets):
    return sum((p - t) ** 2 for p, t in zip(preds, targets)) / len(preds)


def mae(preds, targets):
    return sum(abs(p - t) for p, t in zip(preds, targets)) / len(preds)


def cross_entropy(probs, correct):
    return -math.log(probs[correct])


def main():
    title("02 · Loss Functions")
    section("Example 1 (simplest): house prices in $1000s")
    actual, predicted = [300, 250, 400], [310, 240, 340]
    for i, (a, p) in enumerate(zip(actual, predicted), 1):
        work(f"house {i}", f"error {p - a:+}, squared {(p - a) ** 2}, |error| {abs(p - a)}")
    work("MSE", f"{mse(predicted, actual):,.1f}")
    work("MAE", f"{mae(predicted, actual):,.1f}")

    section("Example 2 (moderate): true y = 1 — MSE vs cross-entropy as p gets worse")
    for p in [0.99, 0.9, 0.5, 0.1, 0.01]:
        work(f"p = {p}", f"MSE (1 − p)² = {(1 - p) ** 2:.4f}    BCE −ln p = {bce(p, 1):.4f}")
    work("3-class CE", f"p = [0.7, 0.2, 0.1], correct = cat → −ln 0.7 = {cross_entropy([0.7, 0.2, 0.1], 0):.4f}")


if __name__ == "__main__":
    main()
