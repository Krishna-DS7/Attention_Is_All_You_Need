"""Sheet 06 · Backpropagation through one neuron.

z = w·x + b,  p = σ(z),  L = (p − y)²
dL/dw = 2(p − y) × p(1 − p) × x        dL/db = 2(p − y) × p(1 − p)
"""
from common import title, section, work, sigmoid, numeric_derivative, fmt

X, Y = 1.5, 1.0


def forward(w, b, x=X, y=Y):
    z = w * x + b
    p = sigmoid(z)
    return z, p, (p - y) ** 2


def backward(w, b, x=X, y=Y):
    """Returns (dL/dp, dp/dz, δ = dL/dz, dL/dw, dL/db)."""
    _, p, _ = forward(w, b, x, y)
    dL_dp = 2 * (p - y)
    dp_dz = p * (1 - p)
    delta = dL_dp * dp_dz
    return dL_dp, dp_dz, delta, delta * x, delta


def train(w=0.8, b=-0.5, lr=1.0, steps=15):
    """Yields (step, w, b, p, loss, dL/dw, dL/db)."""
    for k in range(steps):
        _, p, L = forward(w, b)
        *_, gw, gb = backward(w, b)
        yield k, w, b, p, L, gw, gb
        w, b = w - lr * gw, b - lr * gb


def main():
    title("06 · Backprop Through One Neuron")
    w, b, lr = 0.8, -0.5, 1.0
    section("Example 1 (simplest): one full step (x = 1.5, y = 1, w = 0.8, b = −0.5)")
    z, p, L = forward(w, b)
    work("forward", f"z = {fmt(z)};  p = σ(z) = {fmt(p)};  L = ({fmt(p)} − 1)² = {fmt(L)}")
    dL_dp, dp_dz, delta, gw, gb = backward(w, b)
    work("backward", f"dL/dp {fmt(dL_dp)} × dp/dz {fmt(dp_dz)} = δ {fmt(delta)};  dL/dw = δ×x = {fmt(gw)};  dL/db = {fmt(gb)}")
    nw = numeric_derivative(lambda w_: forward(w_, b)[2], w, 1e-4)
    nb = numeric_derivative(lambda b_: forward(w, b_)[2], b, 1e-4)
    work("gradient check", f"numeric {nw:.6f}, {nb:.6f}  {'✔' if abs(nw - gw) < 1e-6 and abs(nb - gb) < 1e-6 else '✘'}")
    w1, b1 = w - lr * gw, b - lr * gb
    work("update", f"w {fmt(w1)}, b {fmt(b1)} → new loss {fmt(forward(w1, b1)[2])}")

    section("Example 2 (moderate): 15 training steps")
    for k, w_, b_, p_, L_, gw_, gb_ in train():
        work(f"step {k}", f"w {w_:.4f}  b {b_:.4f}  p {p_:.4f}  loss {L_:.4f}")


if __name__ == "__main__":
    main()
