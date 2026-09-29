# Neural Network Basics: prerequisite for the Transformer course

The building blocks that every part of the Transformer is made of: dot products, linear layers, activation functions and softmax. These cover the **forward pass** only. How weights are learned (loss, gradients, backpropagation) is listed as the next step on sheet 12.

- **`Neural_Network_Basics_Explained.xlsx`**: a 13-sheet Excel course. Every calculation is a live formula; start at **00 Start Here**.
- **`python/`**: the same examples in plain Python, one script per sheet, standard library only.

| # | Sheet | Python script |
|---|---|---|
| 01 | What is a neural net: inputs, weights, bias, activation, layers, parameters | `s01_what_is_a_neural_net.py` |
| 02 | Vectors and the dot product: length, angle, cosine, weighted voting | `s02_vectors_and_dot_product.py` |
| 03 | Matrices: shapes, matrix multiplication, batches, transpose, identity | `s03_matrices.py` |
| 04 | The neuron: z = w·x + b, logic gates, decision boundaries | `s04_the_neuron.py` |
| 05 | Linear layer: y = x·W + b; why stacked linear layers collapse | `s05_linear_layer.py` |
| 06 | Activation functions: step, sigmoid, tanh, ReLU, GELU, and their slopes | `s06_activation_functions.py` |
| 07 | Softmax: stability, temperature, the link to sigmoid | `s07_softmax.py` |
| 08 | Why layers: the XOR problem | `s08_why_layers_xor.py` |
| 09 | **End-to-end:** fruit classifier (ReLU → softmax) | `s09_e2e_fruit_classifier.py` |
| 10 | **End-to-end:** exam predictor (standardise → tanh → sigmoid) | `s10_e2e_exam_predictor.py` |
| 11 | **End-to-end:** the same blocks rearranged into attention | `s11_e2e_bridge_to_attention.py` |
| 12 | Glossary and next steps | — |

The end-to-end sheets and scripts follow two or more examples from input to output, showing the math at every stage and ending with a table of how the numbers transformed.

## Run

```bash
cd prerequisites/neural_network_basics/python
python run_all.py                  # every sheet, in order
python run_all.py 9                # just sheet 09
python -m unittest test_examples   # confirms the scripts match the workbook
```

Requires Python 3.8 or newer, with no packages to install. The scripts' results match the workbook at 19 checked points.
