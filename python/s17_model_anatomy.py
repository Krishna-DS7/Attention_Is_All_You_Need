"""Sheet 17 · Model anatomy — shapes and a from-scratch parameter count (paper §3, Table 3).

Assumptions (the paper doesn't list them): every linear layer has a bias, each
LayerNorm has γ and β, one shared embedding table, sinusoidal positions (no params).
"""
from common import title, section, work


def count_params(d, d_ff, N=6, vocab=37000):
    embedding = vocab * d
    attention = 4 * d * d + 4 * d            # W_Q, W_K, W_V, W_O + biases
    ffn = 2 * d * d_ff + d_ff + d
    layer_norm = 2 * d
    enc_layer = attention + ffn + 2 * layer_norm
    dec_layer = 2 * attention + ffn + 3 * layer_norm
    return {"embedding": embedding, "attention block": attention, "FFN": ffn, "encoder layer": enc_layer,
            "decoder layer": dec_layer, "TOTAL": embedding + N * enc_layer + N * dec_layer}


def shapes(n=10, m=9, d=512, h=8, d_ff=2048, vocab=37000):
    dk = d // h
    return [("token ids", f"{n}"), ("embedding × √d", f"{n} × {d}"), ("+ positional encoding", f"{n} × {d}"),
            ("Q, K, V per head", f"{n} × {dk}"), ("scores per head", f"{n} × {n}"), ("head output", f"{n} × {dk}"),
            ("concat heads", f"{n} × {h * dk}"), ("× W_O, Add & Norm", f"{n} × {d}"), ("FFN hidden", f"{n} × {d_ff}"),
            ("encoder layer output", f"{n} × {d}"), ("masked self-attn scores", f"{m} × {m}"),
            ("cross-attn scores", f"{m} × {n}"), ("decoder output", f"{m} × {d}"), ("logits", f"{m} × {vocab:,}")]


def main():
    title("17 · Model Anatomy")
    section("Shape tracker (base model, n = 10 English tokens, m = 9 German tokens)")
    for stage, shape in shapes():
        work(stage, shape)
    for name, d, dff, paper in [("base", 512, 2048, 65e6), ("big", 1024, 4096, 213e6)]:
        section(f"Parameter count — {name} (d_model = {d}, d_ff = {dff})")
        counts = count_params(d, dff)
        for k, v in counts.items():
            work(k, f"{v:,}")
        work("paper", f"{paper:,.0f}  → difference {counts['TOTAL'] / paper - 1:+.1%}")


if __name__ == "__main__":
    main()
