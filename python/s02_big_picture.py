"""Sheet 02 · The Big Picture — the problem the paper solves (Abstract, §1–3).

BIG IDEA: RNNs read a sentence one word at a time (h_t = f(h_{t-1}, x_t)), so they
cannot run in parallel and distant words are many steps apart. Self-attention
lets every word look at every other word directly, all at once.
"""
from common import title, section, work, fmt


def compare_rnn_vs_attention(n):
    """The three counts from Example 1 for a sentence of n words."""
    return {
        "sequential steps": (n, 1),
        "steps for word 1 to reach word n": (n - 1, 1),
        "word-pair connections per layer": (n - 1, n * n),
    }


def telephone_signal(keep_per_hop, distance):
    """ILLUSTRATION only (not from the paper): signal left after `distance` RNN hops."""
    return keep_per_hop ** distance


def main():
    title("02 · The Big Picture")
    section("Example 1 (simplest): 'the cat sat', n = 3")
    for name, (rnn, attn) in compare_rnn_vs_attention(3).items():
        work(name, f"RNN {rnn:>3}   self-attention {attn:>3}")

    section("Example 2 (moderate): toy 'telephone game', keep 90% per hop")
    for dist in [1, 5, 10, 20, 50, 100]:
        s = telephone_signal(0.9, dist)
        verdict = "lost" if s < 0.01 else ("badly faded" if s < 0.5 else "OK")
        work(f"distance {dist}", f"RNN 0.9^{dist} = {s:.4f} ({verdict});  attention 0.9 (always one hop)")


if __name__ == "__main__":
    main()
