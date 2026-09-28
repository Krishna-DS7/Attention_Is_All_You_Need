# Attention Is All You Need: the workbook in plain Python

These scripts recreate the Excel workbook (`../Attention_Is_All_You_Need_Explained.xlsx`, in the repo root) one sheet at a time. They use the same examples and numbers, and print the same step-by-step working. They use plain Python lists and loops, with only the standard `math` module, so every dot product and softmax is written out where you can read it.

## Run

```bash
cd python
python run_all.py          # every sheet, in order
python run_all.py 6        # just sheet 06 (scaled attention)
python s08_multi_head.py   # or any single script
python -m unittest test_examples   # confirms the scripts match the workbook's numbers
```

Requires Python 3.8 or newer. No packages to install.

## Map to the workbook

| Script | Sheet | Key functions |
|---|---|---|
| `common.py` | 01 toolkit | `dot`, `matmul`, `transpose`, `softmax`, `mean`, `variance`, `relu` |
| `s01_math_toolkit.py` | 01 | worked examples of every tool |
| `s02_big_picture.py` | 02 | `compare_rnn_vs_attention` |
| `s03_embeddings.py` | 03 | `lookup`, `embed` |
| `s04_positional_encoding.py` | 04 | `positional_encoding`, `encoder_input` |
| `s05_query_key_value.py` | 05 | `soft_lookup`, `project_qkv` |
| `s06_scaled_attention.py` | 06 | `attention` (Equation 1, optional mask) |
| `s07_masking.py` | 07 | `causal` |
| `s08_multi_head.py` | 08 | `multi_head` |
| `s09_three_attentions.py` | 09 | `cross_attention` |
| `s10_add_and_norm.py` | 10 | `layer_norm`, `add_and_norm` |
| `s11_feed_forward.py` | 11 | `ffn`, `encoder_layer` |
| `s12_output_and_decoding.py` | 12 | `next_token_probs`, `greedy`, `beam_search`, `length_penalty` |
| `s13_loss_and_smoothing.py` | 13 | `smoothed_target`, `cross_entropy`, `perplexity` |
| `s14_optimizer_and_lr.py` | 14 | `learning_rate` (Equation 3), `adam_steps` |
| `s15_regularization.py` | 15 | `dropout` |
| `s16_why_self_attention.py` | 16 | `table1` |
| `s17_model_anatomy.py` | 17 | `shapes`, `count_params` |
| `s18_bleu.py` | 18 | `ngrams`, `clipped_precision`, `bleu` |
| `s19_results.py` | 19 | `training_flops` |

## The running example

As in the workbook, the sentence "the cat sat" flows through the scripts by import:
`s03` (embed) → `s04` (+ position) → `s05` (Q, K, V) → `s08` (multi-head) → `s10` (Add & Norm) → `s11` (feed-forward, Add & Norm).
`s11_feed_forward.encoder_layer(X)` computes one complete Transformer encoder layer.

## Differences from the Excel version

- Masking uses a true `-math.inf`. Excel has no infinity, so the workbook uses −1,000,000,000 instead. Both give a weight of exactly 0.
- The Excel sheets let you type over inputs. Here you edit the numbers at the top of a script, or call its functions with your own values.
