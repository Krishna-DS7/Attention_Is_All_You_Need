# Attention Is All You Need, explained from zero

A study companion for the Transformer paper by Vaswani et al. (2017), [arXiv:1706.03762](https://arxiv.org/abs/1706.03762). It has two parts that cover the same material:

1. **`Attention_Is_All_You_Need_Explained.xlsx`**: a 21-sheet Excel course. Every calculation is a live formula.
2. **`python/`**: the same examples in plain Python, one script per sheet, standard library only.

## The Excel course

Open the workbook and start with **00 Start Here**, which links to every sheet. Each concept gets its own sheet with the same structure:

- **Big idea** and a plain-English explanation with an everyday analogy
- **The paper's formula**, read symbol by symbol, with the values the paper uses
- **Example 1 (simplest)**: tiny numbers you can check by hand, every step written out ("1×2 + 0×1 = 2")
- **Example 2 and up (moderate)**: realistic sizes, often reproducing a number printed in the paper
- **Check yourself** questions

| # | Sheet | Paper |
|---|---|---|
| 01 | Math toolkit: vectors, dot products, matrices, softmax, variance | prerequisite |
| 02 | Big picture: RNN limits and the one bold idea | §1–3 |
| 03 | Tokens and embeddings | §3.4, §5.1 |
| 04 | Positional encoding | §3.5 |
| 05 | Queries, keys, values | §3.2 |
| 06 | Scaled dot-product attention, and why divide by √d_k | §3.2.1 |
| 07 | Masking | §3.1, §3.2.3 |
| 08 | Multi-head attention | §3.2.2 |
| 09 | The three uses of attention | §3.2.3 |
| 10 | Add & Norm | §3.1 |
| 11 | Feed-forward network | §3.3 |
| 12 | Output layer, greedy and beam search | §3.4, §6.1 |
| 13 | Loss, label smoothing, perplexity | §5.4 |
| 14 | Adam and the warm-up learning rate | §5.2–5.3 |
| 15 | Regularisation | §5.4, §6.1 |
| 16 | Why self-attention (Table 1) | §4 |
| 17 | Model anatomy: shapes and parameter counts | §3, Table 3 |
| 18 | BLEU score | Tables 2–3 |
| 19 | Results (Tables 2 and 4, re-derived) | §6 |
| 20 | Glossary | |

The sentence "the cat sat" runs through sheets 03 → 11, so you compute one complete encoder layer across them. Blue-on-yellow cells are inputs: change one and everything downstream recalculates.

## The Python version

```bash
cd python
python run_all.py                  # every sheet, in order
python run_all.py 6                # just sheet 06
python -m unittest test_examples   # confirms the scripts match the workbook
```

Requires Python 3.8 or newer, with no packages to install. See [`python/README.md`](python/README.md) for the script-to-sheet map.

## Accuracy

- The scripts' results match the workbook at 32 checked points, and the workbook's matrices match an independent NumPy implementation.
- Every number and quote attributed to the paper was checked against the v7 PDF. The paper's own inconsistency (41.0 BLEU in the §6.1 text vs 41.8 in Table 2 for English→French) is flagged in sheet 19.
- Example weights, embeddings and probabilities are small made-up values chosen for readability. Details the paper leaves out, such as the LayerNorm ε and the dropout scaling, are labelled where they appear.
