"""Run every script in workbook order: python run_all.py  (or python run_all.py 6 to run only sheet 06)."""
import importlib
import sys

SCRIPTS = ["s01_math_toolkit", "s02_big_picture", "s03_embeddings", "s04_positional_encoding",
           "s05_query_key_value", "s06_scaled_attention", "s07_masking", "s08_multi_head",
           "s09_three_attentions", "s10_add_and_norm", "s11_feed_forward", "s12_output_and_decoding",
           "s13_loss_and_smoothing", "s14_optimizer_and_lr", "s15_regularization", "s16_why_self_attention",
           "s17_model_anatomy", "s18_bleu", "s19_results"]

if __name__ == "__main__":
    wanted = {int(a) for a in sys.argv[1:]}
    for name in SCRIPTS:
        if not wanted or int(name[1:3]) in wanted:
            importlib.import_module(name).main()
