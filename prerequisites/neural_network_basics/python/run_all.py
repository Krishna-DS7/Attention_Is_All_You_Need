"""Run every script in workbook order: python run_all.py  (or python run_all.py 9 to run only sheet 09)."""
import importlib
import sys

SCRIPTS = ["s01_what_is_a_neural_net", "s02_vectors_and_dot_product", "s03_matrices", "s04_the_neuron",
           "s05_linear_layer", "s06_activation_functions", "s07_softmax", "s08_why_layers_xor",
           "s09_e2e_fruit_classifier", "s10_e2e_exam_predictor", "s11_e2e_bridge_to_attention"]

if __name__ == "__main__":
    wanted = {int(a) for a in sys.argv[1:]}
    for name in SCRIPTS:
        if not wanted or int(name[1:3]) in wanted:
            importlib.import_module(name).main()
