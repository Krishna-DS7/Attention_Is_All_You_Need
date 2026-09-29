"""Sheet 09 · End-to-end example 1 — a fruit classifier.

    x (3 features) → [× W1 + b1] → ReLU → [× W2 + b2] → softmax → probabilities → prediction

Weights are hand-set so each hidden neuron has a readable job (apple-, banana-, carrot-detector).
"""
from common import (title, section, work, show_matrix, linear, relu, softmax_steps, transpose, wsum_work, vec,
                    show_stages, fmt)

FEATURES = ["sweet", "crunchy", "long"]
HIDDEN = ["h1 apple-ish", "h2 banana-ish", "h3 carrot-ish"]
CLASSES = ["apple", "banana", "carrot"]
W1 = [[1, 1, -1], [1, -1, 1], [-1, 1, 1]]     # rows = features, columns = hidden neurons
B1 = [-0.5, -0.5, -0.5]
W2 = [[3, -1, -1], [-1, 3, -1], [-1, -1, 3]]  # rows = hidden neurons, columns = classes
B2 = [0, 0, 0]


def forward(x):
    """The whole network. Returns every intermediate stage."""
    z1 = linear(x, W1, B1)
    h = [relu(v) for v in z1]
    z2 = linear(h, W2, B2)
    exps, total, p = softmax_steps(z2)
    return {"x": x, "z1": z1, "h": h, "z2": z2, "exps": exps, "total": total, "p": p,
            "prediction": CLASSES[p.index(max(p))]}


def walkthrough(name, x):
    section(f"{name}: x = {x}  ({', '.join(FEATURES)})")
    s = forward(x)
    print("  Stage 1 · linear layer 1: z1 = x·W1 + b1")
    for j, col in enumerate(transpose(W1)):
        work(HIDDEN[j], wsum_work(x, col, B1[j]))
    print("  Stage 2 · ReLU: h = max(0, z1)")
    work("ReLU", ";   ".join(f"max(0, {fmt(z)}) = {fmt(h)}" for z, h in zip(s["z1"], s["h"])))
    print("  Stage 3 · linear layer 2: z2 = h·W2 + b2")
    for j, col in enumerate(transpose(W2)):
        work(CLASSES[j], wsum_work(s["h"], col, B2[j]))
    print("  Stage 4 · softmax")
    work("softmax", f"{vec(s['exps'])} ÷ {fmt(s['total'])} = {vec(s['p'])}")
    work("STAGE 5 · prediction", f"{s['prediction']} ({max(s['p']):.0%})")
    show_stages([("0 · input", "raw features", "1 × 3", vec(s["x"])),
                 ("1 · linear", "z1 = x·W1 + b1", "1 × 3", vec(s["z1"])),
                 ("2 · ReLU", "h = max(0, z1)", "1 × 3", vec(s["h"])),
                 ("3 · linear", "z2 = h·W2 + b2", "1 × 3", vec(s["z2"])),
                 ("4 · softmax", "p = e^z2 / Σe^z2", "1 × 3", vec(s["p"])),
                 ("5 · decision", "argmax(p)", "label", s["prediction"])])


def main():
    title("09 · End-to-End Example 1 — Fruit Classifier")
    walkthrough("Example 1 (simplest): Fruit A — sweet, crunchy, short", [0.8, 0.7, 0.2])
    walkthrough("Example 2: Fruit B — very sweet, soft, long", [0.9, 0.1, 0.9])
    section("Example 3 (moderate): a batch of four fruits")
    batch = {"Fruit A": [0.8, 0.7, 0.2], "Fruit B": [0.9, 0.1, 0.9], "Fruit C": [0.2, 0.9, 0.8],
             "Fruit D (?)": [0.6, 0.5, 0.5]}
    results = {n: forward(x) for n, x in batch.items()}
    show_matrix("P = softmax(each row)", [r["p"] for r in results.values()], list(batch), CLASSES)
    for n, r in results.items():
        work(n, r["prediction"])


if __name__ == "__main__":
    main()
