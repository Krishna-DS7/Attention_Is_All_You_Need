# How Networks Learn: Backpropagation (prerequisite, part 2)

How a network finds its weights: the loss, derivatives, gradient descent, the chain rule, and backpropagation, all worked by hand. Part 1, [`../neural_network_basics/`](../neural_network_basics/), covers the forward pass.

- **`Backpropagation_Explained.xlsx`**: a 13-sheet Excel course. Every calculation is a live formula, and every gradient is double-checked with a numeric "nudge". Start at **00 Start Here**.
- **`python/`**: the same examples in plain Python, one script per sheet, standard library only.

| # | Sheet | Python script |
|---|---|---|
| 01 | The learning loop: forward → loss → gradient → update | `s01_the_learning_loop.py` |
| 02 | Loss functions: MSE, MAE, binary and categorical cross-entropy | `s02_loss_functions.py` |
| 03 | Derivatives, partial derivatives, the gradient | `s03_derivatives.py` |
| 04 | Gradient descent and learning rates; fitting a line | `s04_gradient_descent.py` |
| 05 | The chain rule and computational graphs | `s05_chain_rule.py` |
| 06 | Backprop through one neuron, with a gradient check | `s06_backprop_one_neuron.py` |
| 07 | Why softmax + cross-entropy gives the gradient p − y | `s07_softmax_ce_gradient.py` |
| 08 | **End-to-end:** a full training step through a 2-layer network (9 gradients checked) | `s08_e2e_backprop_2layer.py` |
| 09 | **End-to-end:** a 40-epoch training loop (a neuron learns OR, fails at XOR) | `s09_e2e_training_loop.py` |
| 10 | Vanishing and exploding gradients, and the Transformer's defences | `s10_vanishing_gradients.py` |
| 11 | Optimizers: SGD, momentum, Adam | `s11_optimizers.py` |
| 12 | Glossary, and how the Transformer paper trains its model | — |

## Run

```bash
cd prerequisites/backpropagation/python
python run_all.py                  # every sheet, in order
python run_all.py 8                # just sheet 08
python -m unittest test_examples   # confirms the scripts match the workbook
```

Requires Python 3.8 or newer, with no packages to install. The scripts' results match the workbook at 24 checked points.
