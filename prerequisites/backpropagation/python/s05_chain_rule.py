"""Sheet 05 · The chain rule:  dy/dx = dy/du × du/dx.

Example 2 is a computational graph for L = (a·b + c)²: forward left to right,
then gradients right to left, each = local slope × upstream gradient.
"""
from common import title, section, work, numeric_derivative, fmt


def graph_forward_backward(a, b, c):
    q = a * b                 # multiply node
    r = q + c                 # add node
    L = r ** 2                # square node
    dL_dr = 2 * r * 1         # local slope 2r, upstream dL/dL = 1
    dL_dq = 1 * dL_dr         # add node passes the gradient through
    dL_dc = 1 * dL_dr
    dL_da = b * dL_dq         # multiply node sends each input the OTHER input
    dL_db = a * dL_dq
    return {"q": q, "r": r, "L": L, "dL/dr": dL_dr, "dL/dq": dL_dq, "dL/dc": dL_dc, "dL/da": dL_da, "dL/db": dL_db}


def main():
    title("05 · Chain Rule")
    section("Example 1 (simplest): y = (3x + 1)² at x = 2")
    x = 2
    u = 3 * x + 1
    work("forward", f"u = 3×2 + 1 = {u};  y = u² = {u * u}")
    work("dy/dx", f"dy/du × du/dx = 2u × 3 = 2×{u} × 3 = {2 * u * 3}")
    work("numeric", f"{numeric_derivative(lambda x: (3 * x + 1) ** 2, x):.6f}")

    section("Example 2 (moderate): computational graph for L = (a·b + c)², a = 2, b = −3, c = 10")
    g = graph_forward_backward(2, -3, 10)
    for k in ("q", "r", "L"):
        work(f"forward {k}", fmt(g[k]))
    for k in ("dL/dr", "dL/dq", "dL/dc", "dL/da", "dL/db"):
        work(k, fmt(g[k]))
    work("numeric dL/da", f"{numeric_derivative(lambda a: (a * -3 + 10) ** 2, 2):.6f}")
    work("numeric dL/db", f"{numeric_derivative(lambda b: (2 * b + 10) ** 2, -3):.6f}")


if __name__ == "__main__":
    main()
