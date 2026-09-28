"""Sheet 16 · Why Self-Attention? — Table 1, paper §4.

Layer type                 Work per layer   Sequential ops   Max path length
Self-attention             n²·d             1                1
Recurrent                  n·d²             n                n
Convolutional              k·n·d²           1                log_k(n)
Restricted self-attention  r·n·d            1                n/r
"""
import math
from common import title, section, work


def table1(n, d, k, r):
    return {
        "self-attention": (n * n * d, 1, 1),
        "recurrent": (n * d * d, n, n),
        "convolutional": (k * n * d * d, 1, math.log(n) / math.log(k)),
        "restricted self-attn": (r * n * d, 1, n / r),
    }


def main():
    title("16 · Why Self-Attention?")
    section("Example 1 (simplest): n = 5, d = 8, k = 3, r = 2")
    for name, (work_, seq, path) in table1(5, 8, 3, 2).items():
        work(name, f"work {work_:>5,}   sequential {seq}   path {path:.2f}")

    section("Example 2 (moderate): d = 512 — self-attention is cheaper exactly when n < d")
    d = 512
    for n in [10, 50, 100, 256, 512, 1000, 4096]:
        sa, rnn = n * n * d, n * d * d
        verdict = "self-attention cheaper" if sa < rnn else ("equal" if sa == rnn else "RNN cheaper per layer")
        work(f"n = {n}", f"n²d {sa:>13,}   nd² {rnn:>13,}   ratio {sa / rnn:.2f}  {verdict}")


if __name__ == "__main__":
    main()
