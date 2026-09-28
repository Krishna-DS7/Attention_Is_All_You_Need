"""Sheet 15 · Regularisation: dropout, label smoothing, checkpoint averaging (paper §5.4, §6.1).

Dropout zeroes a fraction p of numbers during training and scales survivors by
1/(1 − p) ('inverted dropout', as TensorFlow does). The random draws are fixed
here so the output is repeatable.
"""
from common import title, section, work, fmt


def dropout(values, draws, p):
    """Drop a value when its random draw is below p; scale survivors by 1/(1 − p)."""
    return [0.0 if r < p else x / (1 - p) for x, r in zip(values, draws)]


def main():
    title("15 · Regularization")
    section("Example 1 (simplest): dropout on 6 numbers, p = 0.3")
    x = [0.8, -1.2, 2.0, 0.5, -0.3, 1.1]
    draws = [0.62, 0.15, 0.88, 0.41, 0.07, 0.73]
    out = dropout(x, draws, 0.3)
    for i, (a, r, o) in enumerate(zip(x, draws, out), 1):
        work(f"x{i}", f"draw {r} {'< 0.3 → dropped' if r < 0.3 else '≥ 0.3 → kept, ÷ 0.7'} → {fmt(o)}")

    section("Example 2 (moderate): scaling keeps the average the same")
    draws10 = [0.62, 0.15, 0.88, 0.41, 0.07, 0.73, 0.29, 0.95, 0.52, 0.34]
    passed = dropout([2.0] * 10, draws10, 0.3)
    work("10 steps", [fmt(v) for v in passed])
    work("average", f"{fmt(sum(passed) / 10)}  (= the original 2.0)")

    section("Checkpoint averaging: the last 5 checkpoints of one weight")
    ck = [0.512, 0.498, 0.505, 0.521, 0.494]
    work("average", f"{fmt(sum(ck) / len(ck))}")

    section("Table 3 rows (D) from the paper")
    for name, pd, ls, ppl, bleu in [("base", 0.1, 0.1, 4.92, 25.8), ("no dropout", 0.0, 0.1, 5.77, 24.6),
                                     ("more dropout", 0.2, 0.1, 4.95, 25.5), ("no smoothing", 0.1, 0.0, 4.67, 25.3),
                                     ("more smoothing", 0.1, 0.2, 5.47, 25.7)]:
        work(name, f"P_drop {pd}  ε_ls {ls}  PPL {ppl}  BLEU {bleu}")


if __name__ == "__main__":
    main()
