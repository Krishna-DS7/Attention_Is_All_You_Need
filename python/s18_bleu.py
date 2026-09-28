"""Sheet 18 · BLEU score (Papineni et al., 2002 — the metric used in all the paper's tables).

    BLEU = BP × exp( (ln p1 + ln p2 + ln p3 + ln p4) / 4 ),   BP = 1 if c > r else e^(1 − r/c)
    p_n = clipped n-gram matches ÷ n-grams in the candidate
"""
import math
from collections import Counter
from common import title, section, work, fmt


def ngrams(words, n):
    return [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]


def clipped_precision(candidate, reference, n):
    """Each candidate n-gram is credited at most as often as it appears in the reference."""
    cand, ref = Counter(ngrams(candidate, n)), Counter(ngrams(reference, n))
    matches = sum(min(count, ref[g]) for g, count in cand.items())
    return matches, sum(cand.values())


def bleu(candidate, reference, max_n=4):
    precisions = []
    for n in range(1, max_n + 1):
        m, total = clipped_precision(candidate, reference, n)
        precisions.append(m / total if total else 0.0)
    c, r = len(candidate), len(reference)
    bp = 1.0 if c > r else math.exp(1 - r / c)
    if min(precisions) == 0:
        return 0.0, bp, precisions
    return bp * math.exp(sum(math.log(p) for p in precisions) / max_n), bp, precisions


def main():
    title("18 · BLEU Score")
    section("Example 1 (simplest): why clipping — 'the the the the' vs 'the cat is on the mat'")
    ref = "the cat is on the mat".split()
    cand = "the the the the".split()
    naive = sum(1 for w in cand if w in ref) / len(cand)
    m, total = clipped_precision(cand, ref, 1)
    work("naive precision", f"{naive:.0%}")
    work("clipped", f"min(4, {ref.count('the')}) ÷ 4 = {m / total:.0%}")

    section("Example 2 (moderate): 'the cat sat on mat' vs 'the cat sat on the mat'")
    cand, ref = "the cat sat on mat".split(), "the cat sat on the mat".split()
    score, bp, ps = bleu(cand, ref)
    for n, p in enumerate(ps, 1):
        m, total = clipped_precision(cand, ref, n)
        work(f"{n}-grams", f"{m}/{total} = {p:.3f}   ln = {math.log(p):.3f}")
    work("brevity penalty", f"c = {len(cand)}, r = {len(ref)} → e^(1 − {len(ref)}/{len(cand)}) = {bp:.4f}")
    work("BLEU", f"{fmt(bp)} × exp(({' + '.join(fmt(math.log(p)) for p in ps)}) ÷ 4) = {score:.4f}  (×100 = {score*100:.1f})")


if __name__ == "__main__":
    main()
