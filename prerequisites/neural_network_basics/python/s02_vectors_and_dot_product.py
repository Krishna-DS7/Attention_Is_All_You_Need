"""Sheet 02 · Vectors and the dot product.

    a · b = a1·b1 + a2·b2 + … = |a| × |b| × cos(angle)

A neuron's weighted sum and attention's query·key score are both dot products.
"""
import math
from common import title, section, work, dot, dot_work, length, cosine, fmt


def angle_degrees(a, b):
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine(a, b)))))


def main():
    title("02 · Vectors and the Dot Product")
    section("Example 1 (simplest): same direction, perpendicular, opposite")
    a = [3, 4]
    work("|a|", f"√(3² + 4²) = {fmt(length(a))}")
    for name, v, reading in [("a · b", [6, 8], "same way → big positive"), ("a · c", [-4, 3], "right angle → 0"),
                             ("a · d", [-3, -4], "opposite → big negative")]:
        work(name, f"{dot_work(a, v)}   cosine {cosine(a, v):.2f}, angle {angle_degrees(a, v):.0f}°  ({reading})")

    section("Example 2 (moderate): a movie recommender — the dot product as weighted voting")
    taste = [0.9, -0.5, 0.6]              # [action, romance, comedy] — the 'weights'
    movies = {"Car Chase 7": [0.9, 0.1, 0.3], "Rainy Paris": [0.1, 0.9, 0.4],
              "Heist Comedy": [0.5, 0.2, 0.9], "Slow Romance": [0.0, 0.8, 0.0]}
    scores = {m: dot(taste, v) for m, v in movies.items()}
    for rank, m in enumerate(sorted(scores, key=scores.get, reverse=True), 1):
        work(f"#{rank} {m}", dot_work(taste, movies[m]))


if __name__ == "__main__":
    main()
