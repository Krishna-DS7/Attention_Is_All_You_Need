"""Sheet 14 · Adam and the warm-up learning rate — Equation (3), paper §5.2–5.3.

    lrate = d_model^(−0.5) · min(step^(−0.5), step · warmup^(−1.5))
"""
import math
from common import title, section, work, fmt


def learning_rate(step, d_model=512, warmup=4000):
    return d_model ** -0.5 * min(step ** -0.5, step * warmup ** -1.5)


def adam_steps(theta, grads, lr=0.001, b1=0.9, b2=0.98, eps=1e-9):
    """Adam for one weight. Yields (t, g, m, v, m_hat, v_hat, step, theta) per step."""
    m = v = 0.0
    for t, g in enumerate(grads, start=1):
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        m_hat, v_hat = m / (1 - b1 ** t), v / (1 - b2 ** t)
        step = lr * m_hat / (math.sqrt(v_hat) + eps)
        theta -= step
        yield t, g, m, v, m_hat, v_hat, step, theta


def main():
    title("14 · Optimizer and Learning Rate")
    section("Example 1 (simplest): d_model = 4, warmup = 4")
    for step in range(1, 13):
        a, b = step * 4 ** -1.5, step ** -0.5
        phase = "warming up" if a < b else ("peak" if a == b else "decaying")
        work(f"step {step}", f"min({a:.4f}, {b:.4f}) × 4^-0.5 = {learning_rate(step, 4, 4):.4f}  {phase}")

    section("Example 2 (moderate): the paper's schedule (d_model = 512, warmup = 4000)")
    peak = learning_rate(4000)
    work("peak", f"{peak:.6f} at step 4000")
    for step in [1, 100, 1000, 2000, 4000, 8000, 16000, 50000, 100000, 300000]:
        work(f"step {step:,}", f"{learning_rate(step):.6f}  ({learning_rate(step) / peak:.0%} of peak)")

    section("Example 3 (moderate): Adam, one weight, three steps (β1 = 0.9, β2 = 0.98, ε = 1e-9)")
    for t, g, m, v, mh, vh, st, th in adam_steps(0.5, [0.5, 0.3, -0.2]):
        work(f"t = {t}", f"g {g:5}  m {m:.5f}  v {v:.5f}  m̂ {mh:.4f}  v̂ {vh:.4f}  step {st:.6f}  θ {th:.6f}")

    section("Example 4: checking the paper's training times")
    work("base", f"100,000 × 0.4 s = {100000 * 0.4 / 3600:.1f} hours (paper: 12 hours)")
    work("big", f"300,000 × 1.0 s = {300000 / 86400:.2f} days (paper: 3.5 days)")


if __name__ == "__main__":
    main()
