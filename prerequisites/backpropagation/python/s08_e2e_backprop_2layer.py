"""Sheet 08 · End-to-end example 1 — one full training step of a 2-layer network.

x (2) → [x·W1 + b1] → sigmoid → a (2) → [a·w2 + b2] → sigmoid → p → loss = BCE(p, y)

Backward:  δ2 = p − y;   dL/dw2 = δ2·a;   dL/db2 = δ2
           δ1 = (δ2·w2) ⊙ a(1 − a);   dL/dW1[i][j] = x_i·δ1_j;   dL/db1 = δ1
"""
from common import title, section, work, sigmoid, bce, vec, show_stages, fmt

X = [1, 0.5]
Y = 1
PARAMS = {"W1[x1→h1]": 0.5, "W1[x1→h2]": -0.3, "W1[x2→h1]": 0.2, "W1[x2→h2]": 0.8,
          "b1[h1]": 0.1, "b1[h2]": 0.1, "w2[h1]": 0.7, "w2[h2]": -0.4, "b2": 0.2}


def unpack(P):
    W1 = [[P["W1[x1→h1]"], P["W1[x1→h2]"]], [P["W1[x2→h1]"], P["W1[x2→h2]"]]]
    return W1, [P["b1[h1]"], P["b1[h2]"]], [P["w2[h1]"], P["w2[h2]"]], P["b2"]


def forward(P, x=X, y=Y):
    W1, b1, w2, b2 = unpack(P)
    z1 = [x[0] * W1[0][j] + x[1] * W1[1][j] + b1[j] for j in range(2)]
    a = [sigmoid(v) for v in z1]
    z2 = a[0] * w2[0] + a[1] * w2[1] + b2
    p = sigmoid(z2)
    return {"z1": z1, "a": a, "z2": z2, "p": p, "loss": bce(p, y)}


def backward(P, x=X, y=Y):
    f = forward(P, x, y)
    _, _, w2, _ = unpack(P)
    a, p = f["a"], f["p"]
    d2 = p - y
    d1 = [d2 * w2[j] * a[j] * (1 - a[j]) for j in range(2)]
    grads = {"W1[x1→h1]": x[0] * d1[0], "W1[x1→h2]": x[0] * d1[1], "W1[x2→h1]": x[1] * d1[0],
             "W1[x2→h2]": x[1] * d1[1], "b1[h1]": d1[0], "b1[h2]": d1[1],
             "w2[h1]": d2 * a[0], "w2[h2]": d2 * a[1], "b2": d2}
    return f, d2, d1, grads


def numeric_gradients(P, h=1e-4):
    out = {}
    for k in P:
        up, down = dict(P), dict(P)
        up[k] += h
        down[k] -= h
        out[k] = (forward(up)["loss"] - forward(down)["loss"]) / (2 * h)
    return out


def step(P, grads, lr=0.5):
    return {k: P[k] - lr * grads[k] for k in P}


def main():
    title("08 · End-to-End Backprop Through a 2-Layer Network")
    f, d2, d1, grads = backward(PARAMS)
    section("Forward pass")
    work("z1", vec(f["z1"]))
    work("a = σ(z1)", vec(f["a"]))
    work("z2", fmt(f["z2"]))
    work("p, loss", f"{fmt(f['p'])}, −ln p = {fmt(f['loss'])}")
    section("Backward pass")
    work("δ2 = p − y", fmt(d2))
    work("δ1", f"(δ2 × w2) ⊙ a(1 − a) = {vec(d1)}")
    section("Gradient check — all nine parameters")
    num = numeric_gradients(PARAMS)
    for k in PARAMS:
        work(k, f"backprop {grads[k]:+.6f}   numeric {num[k]:+.6f}   {'✔' if abs(grads[k] - num[k]) < 1e-6 else '✘'}")
    section("Update (η = 0.5)")
    new = step(PARAMS, grads)
    after = forward(new)["loss"]
    work("loss", f"{fmt(f['loss'])} → {fmt(after)}  (change {after - f['loss']:+.4f})")
    show_stages([("F1 · linear", "z1 = x·W1 + b1", "2", vec(f["z1"])), ("F2 · sigmoid", "a = σ(z1)", "2", vec(f["a"])),
                 ("F3 · linear", "z2 = a·w2 + b2", "1", fmt(f["z2"])), ("F4 · output", "p, L = −ln p", "1+1",
                 f"{fmt(f['p'])} → L {fmt(f['loss'])}"), ("B1 · output err", "δ2 = p − y", "1", fmt(d2)),
                 ("B3 · hidden err", "δ1", "2", vec(d1)), ("U · update", "θ ← θ − η∇θ", "9",
                 f"loss {fmt(f['loss'])} → {fmt(after)}")])


if __name__ == "__main__":
    main()
