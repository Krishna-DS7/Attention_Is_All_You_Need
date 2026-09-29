"""Sheet 11 · Optimizers in a narrow valley: L = w1² + 10·w2², start (−4, 1).

SGD:       w ← w − η·g
Momentum:  m ← β·m + g;  w ← w − η·m
Adam:      m ← β1·m + (1−β1)·g;  v ← β2·v + (1−β2)·g²;  w ← w − η·m̂/(√v̂ + ε)
"""
import math
from common import title, section, work


def loss(w):
    return w[0] ** 2 + 10 * w[1] ** 2


def grad(w):
    return [2 * w[0], 20 * w[1]]


def sgd(lr=0.05, steps=30):
    w, out = [-4.0, 1.0], []
    for _ in range(steps + 1):
        out.append(list(w))
        w = [wi - lr * gi for wi, gi in zip(w, grad(w))]
    return out


def momentum(lr=0.05, beta=0.5, steps=30):
    w, m, out = [-4.0, 1.0], [0.0, 0.0], []
    for _ in range(steps + 1):
        out.append(list(w))
        m = [beta * mi + gi for mi, gi in zip(m, grad(w))]
        w = [wi - lr * mi for wi, mi in zip(w, m)]
    return out


def adam(lr=0.2, b1=0.9, b2=0.999, eps=1e-8, steps=30):
    w, m, v, out = [-4.0, 1.0], [0.0, 0.0], [0.0, 0.0], [[-4.0, 1.0]]
    for t in range(1, steps + 1):
        g = grad(w)
        m = [b1 * mi + (1 - b1) * gi for mi, gi in zip(m, g)]
        v = [b2 * vi + (1 - b2) * gi * gi for vi, gi in zip(v, g)]
        w = [wi - lr * (mi / (1 - b1 ** t)) / (math.sqrt(vi / (1 - b2 ** t)) + eps) for wi, mi, vi in zip(w, m, v)]
        out.append(list(w))
    return out


def main():
    title("11 · Optimizers")
    runs = {"SGD": sgd(), "Momentum": momentum(), "Adam": adam()}
    section("Loss per step")
    for k in (0, 1, 2, 5, 10, 20, 30):
        work(f"step {k}", "   ".join(f"{n}: {loss(r[k]):.4f}" for n, r in runs.items()))
    a = runs["Adam"]
    work("Adam step 1 sizes", f"w1 moved {abs(a[1][0] - a[0][0]):.4f}, w2 moved {abs(a[1][1] - a[0][1]):.4f} (both ≈ η = 0.2)")


if __name__ == "__main__":
    main()
