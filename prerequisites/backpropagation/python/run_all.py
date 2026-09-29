"""Run every script in workbook order: python run_all.py  (or python run_all.py 8 to run only sheet 08)."""
import importlib
import sys

SCRIPTS = ["s01_the_learning_loop", "s02_loss_functions", "s03_derivatives", "s04_gradient_descent", "s05_chain_rule",
           "s06_backprop_one_neuron", "s07_softmax_ce_gradient", "s08_e2e_backprop_2layer", "s09_e2e_training_loop",
           "s10_vanishing_gradients", "s11_optimizers"]

if __name__ == "__main__":
    wanted = {int(a) for a in sys.argv[1:]}
    for name in SCRIPTS:
        if not wanted or int(name[1:3]) in wanted:
            importlib.import_module(name).main()
