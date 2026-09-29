"""Sheet 03 · Matrices and matrix multiplication — many dot products at once.

    (A·B)[i][j] = row i of A · column j of B          (m × p) · (p × q) → (m × q)
"""
from common import title, section, work, show_matrix, matmul, transpose, dot_work


def shape_check(m, p, p2, q):
    return f"OK → result is {m} × {q}" if p == p2 else f"✘ mismatch: A has {p} columns but B has {p2} rows"


def main():
    title("03 · Matrices")
    section("Shape checker")
    work("(3×4)·(4×2)", shape_check(3, 4, 4, 2))
    work("(4×2)·(4×2)", shape_check(4, 2, 4, 2))

    section("Example 1 (simplest): 2×2 times 2×2")
    A, B = [[1, 2], [3, 4]], [[5, 6], [7, 8]]
    show_matrix("A · B", matmul(A, B))
    for i in range(2):
        for j in range(2):
            work(f"({i+1},{j+1})", dot_work(A[i], transpose(B)[j]))

    section("Example 2 (moderate): a batch of 3 examples through 2 neurons")
    X = [[1, 0, 2], [0, 1, 1], [2, 1, 0]]          # rows = examples
    W = [[0.5, -1], [1, 0], [0, 2]]                # columns = neurons
    XW = matmul(X, W)
    show_matrix("X · W  (rows = examples, columns = neurons)", XW, ["example 1", "example 2", "example 3"],
                ["neuron 1", "neuron 2"])
    for i in range(3):
        for j in range(2):
            work(f"ex {i+1}, neuron {j+1}", dot_work(X[i], transpose(W)[j]))
    show_matrix("Wᵀ (rows ↔ columns)", transpose(W), ["neuron 1", "neuron 2"], ["f1", "f2", "f3"])
    identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    work("X · I == X ?", str(matmul(X, identity) == X))


if __name__ == "__main__":
    main()
