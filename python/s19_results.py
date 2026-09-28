"""Sheet 19 · Results — Table 2 with the headline claims re-derived (paper §6, footnote 5)."""
from common import title, section, work

# model: (BLEU EN-DE, BLEU EN-FR, FLOPs EN-DE, FLOPs EN-FR)  — the paper's Table 2 (None = not reported)
TABLE2 = {
    "ByteNet": (23.75, None, None, None),
    "Deep-Att + PosUnk": (None, 39.2, None, 1.0e20),
    "GNMT + RL": (24.6, 39.92, 2.3e19, 1.4e20),
    "ConvS2S": (25.16, 40.46, 9.6e18, 1.5e20),
    "MoE": (26.03, 40.56, 2.0e19, 1.2e20),
    "Deep-Att + PosUnk Ensemble": (None, 40.4, None, 8.0e20),
    "GNMT + RL Ensemble": (26.30, 41.16, 1.8e20, 1.1e21),
    "ConvS2S Ensemble": (26.36, 41.29, 7.7e19, 1.2e21),
    "Transformer (base)": (27.3, 38.1, 3.3e18, 3.3e18),     # one figure, printed across both columns
    "Transformer (big)": (28.4, 41.8, 2.3e19, 2.3e19),
}
PREVIOUS = [m for m in TABLE2 if not m.startswith("Transformer")]


def training_flops(hours, gpus=8, tflops=9.5):
    """Footnote 5: training time × number of GPUs × sustained TFLOPS per GPU."""
    return hours * 3600 * gpus * tflops * 1e12


def main():
    title("19 · Results")
    best_de = max(TABLE2[m][0] for m in PREVIOUS if TABLE2[m][0])
    best_single_fr = max(TABLE2[m][1] for m in ["Deep-Att + PosUnk", "GNMT + RL", "ConvS2S", "MoE"])
    section("Checking the headline claims")
    work("best previous EN-DE", f"{best_de} (ConvS2S Ensemble)")
    work("big gain EN-DE", f"{TABLE2['Transformer (big)'][0] - best_de:+.2f} BLEU  ('more than 2.0')")
    work("base gain EN-DE", f"{TABLE2['Transformer (base)'][0] - best_de:+.2f} BLEU")
    work("big vs best single FR", f"{TABLE2['Transformer (big)'][1] - best_single_fr:+.2f} BLEU")
    work("cost ratio", f"ConvS2S Ens. ÷ Transformer base = {7.7e19 / 3.3e18:.1f}× cheaper")
    work("big FR cost ÷ MoE", f"{2.3e19 / 1.2e20:.1%}  ('less than 1/4')")
    work("note", "Table 2 says 41.8 EN-FR BLEU; §6.1's text says 41.0 (inconsistency in v7)")
    section("Footnote 5: re-deriving the FLOP estimates (P100 = 9.5 TFLOPS)")
    work("base, 12 hours", f"{training_flops(12):.2e}  (paper 3.3e18)")
    work("big, 3.5 days", f"{training_flops(84):.2e}  (paper 2.3e19)")


if __name__ == "__main__":
    main()
