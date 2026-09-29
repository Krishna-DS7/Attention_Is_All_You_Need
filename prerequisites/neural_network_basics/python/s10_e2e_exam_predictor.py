"""Sheet 10 · End-to-end example 2 — will a student pass?

    raw x → standardise (x − μ)/σ → [× W1 + b1] → tanh → [· w2 + b2] → sigmoid → p(pass) → decision
"""
import math
from common import title, section, work, dot, linear, sigmoid, transpose, wsum_work, vec, show_stages, fmt

INPUTS = ["hours studied", "hours slept", "practice tests"]
HIDDEN = ["h1 preparation", "h2 rest"]
MEAN = [5, 6, 1.5]            # assumed statistics of the training data
STD = [2.5, 1.5, 1.2]
W1 = [[0.8, 0], [0, 1.0], [0.6, 0]]
B1 = [0, -0.2]
W2 = [2.5, 1.0]
B2 = -0.5
THRESHOLD = 0.5


def forward(raw, threshold=THRESHOLD):
    xs = [(x - m) / s for x, m, s in zip(raw, MEAN, STD)]
    z1 = linear(xs, W1, B1)
    h = [math.tanh(v) for v in z1]
    z2 = dot(h, W2) + B2
    p = sigmoid(z2)
    return {"raw": raw, "xs": xs, "z1": z1, "h": h, "z2": z2, "p": p,
            "decision": "PASS" if p >= threshold else "FAIL"}


def walkthrough(name, raw):
    section(f"{name}: {dict(zip(INPUTS, raw))}")
    s = forward(raw)
    print("  Stage 1 · standardise: (x − mean) ÷ std")
    for n, x, m, sd, v in zip(INPUTS, raw, MEAN, STD, s["xs"]):
        work(n, f"({fmt(x)} − {fmt(m)}) ÷ {fmt(sd)} = {fmt(v)}")
    print("  Stage 2 · linear layer 1: z1 = x̃·W1 + b1")
    for j, col in enumerate(transpose(W1)):
        work(HIDDEN[j], wsum_work(s["xs"], col, B1[j]))
    print("  Stage 3 · tanh")
    work("tanh", ";   ".join(f"tanh({fmt(z)}) = {fmt(h)}" for z, h in zip(s["z1"], s["h"])))
    print("  Stage 4 · output neuron: z2 = h·w2 + b2")
    work("z2", wsum_work(s["h"], W2, B2))
    print("  Stage 5 · sigmoid")
    work("p(pass)", f"1 ÷ (1 + e^({fmt(-s['z2'])})) = {fmt(s['p'])}")
    work("STAGE 6 · decision", f"{s['decision']} (threshold {THRESHOLD})")
    show_stages([("0 · raw input", "hours, hours, count", "1 × 3", vec(s["raw"])),
                 ("1 · standardise", "(x − μ) ÷ σ", "1 × 3", vec(s["xs"])),
                 ("2 · linear", "z1 = x̃·W1 + b1", "1 × 2", vec(s["z1"])),
                 ("3 · tanh", "h = tanh(z1)", "1 × 2", vec(s["h"])),
                 ("4 · linear", "z2 = h·w2 + b2", "1 × 1", fmt(s["z2"])),
                 ("5 · sigmoid", "p = 1/(1+e^−z2)", "1 × 1", f"{s['p']:.1%}"),
                 ("6 · decision", "p ≥ threshold?", "label", s["decision"])])


def main():
    title("10 · End-to-End Example 2 — Exam Predictor")
    walkthrough("Example 1 (simplest): Student A", [8, 7, 3])
    walkthrough("Example 2: Student B", [2, 5, 0])
    section("Example 3 (moderate): the whole class, and the effect of the threshold")
    for name, raw in [("Student A", [8, 7, 3]), ("Student B", [2, 5, 0]), ("Student C", [5, 8, 1])]:
        s = forward(raw)
        work(name, f"p = {s['p']:.1%} → {s['decision']} at 0.5, {forward(raw, 0.4)['decision']} at 0.4")


if __name__ == "__main__":
    main()
