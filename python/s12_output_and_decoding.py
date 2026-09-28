"""Sheet 12 · From vectors back to words (paper §3.4, §6.1).

    P(next token) = softmax( h · Eᵀ )      — E is the shared embedding table
Then pick words with greedy search or beam search (beam 4, length penalty α = 0.6).
"""
import math
from common import title, section, work, dot, dot_work, softmax_steps, fmt
from s03_embeddings import VOCAB, E


def next_token_probs(h):
    logits = [dot(h, row) for row in E]
    _, _, probs = softmax_steps(logits)
    return logits, probs


# Toy probabilities for the beam-search example
FIRST = {"Die": 0.5, "Eine": 0.4, "Der": 0.1}
SECOND = {"Die": {"Katze": 0.4, "Kater": 0.35, "Hund": 0.25},
          "Eine": {"Katze": 0.9, "Kater": 0.05, "Hund": 0.05},
          "Der": {"Kater": 0.8, "Hund": 0.2}}


def greedy():
    w1 = max(FIRST, key=FIRST.get)
    w2 = max(SECOND[w1], key=SECOND[w1].get)
    return f"{w1} {w2}", FIRST[w1] * SECOND[w1][w2]


def beam_search(beam_size):
    kept = sorted(FIRST, key=FIRST.get, reverse=True)[:beam_size]          # prune after step 1
    candidates = [(f"{a} {b}", FIRST[a] * p) for a in kept for b, p in SECOND[a].items()]
    return max(candidates, key=lambda c: c[1]), kept


def length_penalty(length, alpha=0.6):
    """From Wu et al. (GNMT, the paper's ref [38]): lp(Y) = ((5 + |Y|) / 6)^α."""
    return ((5 + length) / 6) ** alpha


def main():
    title("12 · Output and Decoding")
    section("Example 1 (simplest): score the 8-token vocabulary")
    h = [1, 2.4, 3, -2]
    logits, probs = next_token_probs(h)
    for tok, lg, p in zip(VOCAB, logits, probs):
        work(tok, f"logit {lg:7.3f}   probability {p:6.1%}")
    work("logit Katze", "h·E[Katze] = " + dot_work(h, E[VOCAB.index("Katze")]))
    work("most likely", VOCAB[probs.index(max(probs))])

    section("Example 2 (moderate): greedy vs beam search (beam size 2)")
    g, gp = greedy()
    (b, bp), kept = beam_search(2)
    work("greedy", f"{g}  P = {fmt(gp)}")
    work("beam keeps", ", ".join(kept))
    work("beam", f"{b}  P = {fmt(bp)}   log P = {fmt(math.log(bp))}")

    section("Length penalty α = 0.6: score = log P / lp(|Y|)")
    for name, n, logp in [("short translation", 2, -1.4), ("complete translation", 5, -1.5)]:
        work(name, f"|Y| = {n}, log P = {logp}, lp = {length_penalty(n):.4f}, score = {logp / length_penalty(n):.4f}")


if __name__ == "__main__":
    main()
