"""Shared helpers for the Neural Network Basics scripts (sheets 02–07 of the workbook).

Everything is plain Python lists and loops so each step is visible.
A vector is a list of numbers; a matrix is a list of rows (each row a list).
"""
import math


# ---------------------------------------------------------------- formatting
def fmt(x):
    """Format like the workbook's 'show the work' rows: 3 decimals, negatives in brackets."""
    r = round(x, 3)
    if r == 0:
        r = 0.0                      # avoid printing -0
    s = f"{r:g}"
    return f"({s})" if r < 0 else s


def title(text):
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def section(text):
    print(f"\n--- {text} ---")


def work(label, text):
    """One 'show the work' line. `text` may be a string or a list of items."""
    if isinstance(text, (list, tuple)):
        text = ", ".join(str(t) for t in text)
    print(f"  {label:<18} {text}")


def show_matrix(name, M, row_labels=None, col_labels=None, width=9):
    print(f"\n  {name}")
    row_labels = row_labels or [f"row {i+1}" for i in range(len(M))]
    col_labels = col_labels or [f"d{j+1}" for j in range(len(M[0]))]
    print("  " + " " * 10 + "".join(f"{c:>{width}}" for c in col_labels))
    for lab, row in zip(row_labels, M):
        cells = "".join(f"{('-inf' if v == -math.inf else f'{v:.3f}'):>{width}}" for v in row)
        print(f"  {lab:<10}{cells}")


def show_vector(name, v, labels=None):
    show_matrix(name, [v], [""], labels)


# ---------------------------------------------------------------- vectors & matrices
def dot(a, b):
    """a · b = a1*b1 + a2*b2 + ... (Tool 2)."""
    return sum(x * y for x, y in zip(a, b))


def dot_work(a, b):
    """The dot product written out: '1×3 + 2×4  =  11'."""
    return " + ".join(f"{fmt(x)}×{fmt(y)}" for x, y in zip(a, b)) + f"  =  {fmt(dot(a, b))}"


def length(v):
    return math.sqrt(sum(x * x for x in v))


def cosine(a, b):
    return dot(a, b) / (length(a) * length(b))


def transpose(M):
    """Rows become columns (Tool 3)."""
    return [list(col) for col in zip(*M)]


def matmul(A, B):
    """(A·B)[i][j] = row i of A · column j of B (Tool 4)."""
    cols = transpose(B)
    return [[dot(row, col) for col in cols] for row in A]


def add(A, B):
    """Element-wise sum of two matrices (or two vectors)."""
    if isinstance(A[0], list):
        return [[a + b for a, b in zip(ra, rb)] for ra, rb in zip(A, B)]
    return [a + b for a, b in zip(A, B)]


def scale(M, c):
    if isinstance(M[0], list):
        return [[v * c for v in row] for row in M]
    return [v * c for v in M]


# ---------------------------------------------------------------- softmax & statistics
def softmax_steps(scores):
    """Softmax the way the sheets do it: e^score, total, divide. Returns (exps, total, weights).
    A score of -inf gives e^-inf = 0, i.e. zero weight (used for masking)."""
    exps = [math.exp(s) for s in scores]
    total = sum(exps)
    return exps, total, [e / total for e in exps]


def softmax(scores):
    return softmax_steps(scores)[2]


def mean(v):
    return sum(v) / len(v)


def variance(v):
    """Population variance: average squared distance from the mean (what LayerNorm uses)."""
    m = mean(v)
    return sum((x - m) ** 2 for x in v) / len(v)


def relu(x):
    return max(0.0, x)


# ---------------------------------------------------------------- activation functions (sheet 06)
def step(z):
    return 1.0 if z > 0 else 0.0


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def tanh(z):
    return math.tanh(z)


def gelu(z):
    """The tanh approximation used by BERT and GPT models."""
    return 0.5 * z * (1 + math.tanh(math.sqrt(2 / math.pi) * (z + 0.044715 * z ** 3)))


ACTIVATIONS = {"step": step, "sigmoid": sigmoid, "tanh": tanh, "ReLU": relu, "GELU": gelu}


# ---------------------------------------------------------------- one layer and the stage tracker
def linear(x, W, b):
    """y = x·W + b for one input vector: each column of W is one neuron."""
    return [dot(x, col) + bj for col, bj in zip(transpose(W), b)]


def wsum_work(x, w, b):
    """'x1×w1 + x2×w2 + (b)  =  z' — a neuron's weighted sum written out."""
    z = dot(x, w) + b
    return " + ".join(f"{fmt(a)}×{fmt(c)}" for a, c in zip(x, w)) + f" + {fmt(b)}  =  {fmt(z)}"


def vec(v):
    return "[" + ", ".join(fmt(x) for x in v) + "]"


def show_stages(rows):
    """Print the 'how the numbers transformed' table: (stage, operation, shape, values)."""
    print("\n  How the numbers transformed, stage by stage:")
    for stage, op, shape, values in rows:
        print(f"    {stage:<16} {op:<22} {shape:<8} {values}")
