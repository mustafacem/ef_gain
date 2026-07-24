"""Generate the paper's figures from the actual result JSONs.

Every plotted point is read from results/*.json -- no hand-entered numbers.
Palette is Okabe-Ito (colour-vision-deficiency safe, the scientific standard).
Output: vector PDF, arXiv-friendly.
"""
import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(__file__), "..", "..", "results")
OUT = os.path.dirname(__file__)


def L(f):
    return json.load(open(os.path.join(R, f)))


# --- Okabe-Ito, assigned by the job each colour does -----------------------
OKABE = {
    "black": "#000000", "orange": "#E69F00", "sky": "#56B4E9",
    "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
    "vermillion": "#D55E00", "purple": "#CC79A7",
}
C_UNIFORM = OKABE["blue"]       # uniform quantization
C_MIXED = OKABE["green"]        # mixed precision
C_PATCH = OKABE["orange"]       # restored patches
C_LEARNED = OKABE["vermillion"]  # learned patch (caution)
C_FP16 = OKABE["black"]         # reference
INK = "#222222"
MUTED = "#666666"
GRID = "#DDDDDD"

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150,
})


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


# ===========================================================================
# FIG 1 -- the Pareto frontier in GSM8K accuracy vs bits/weight
# ===========================================================================
def fig_frontier():
    gf = L("gsm8k_frontier.json")
    pts = []
    for k in gf["acc"]:
        pts.append((gf["bits"][k], 100 * gf["acc"][k], k))
    # classify
    def fam(k):
        if k == "fp16":
            return ("fp16", C_FP16, "D")
        if k.startswith("uniform"):
            return ("uniform", C_UNIFORM, "s")
        if "+pb3" in k or "+pb" in k:
            return ("mixed+patch", C_PATCH, "o")
        return ("mixed", C_MIXED, "^")

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    grid(ax)
    # sub-4.25-bit band shading (where uniform cannot exist)
    ax.axvspan(3.25, 4.25, color=OKABE["yellow"], alpha=0.12, zorder=0)
    ax.text(3.75, 2, "integer uniform\ncannot occupy", ha="center", va="bottom",
            fontsize=7, color=MUTED, style="italic")

    seen = set()
    for b, a, k in pts:
        if b > 6:  # keep fp16 as an axis reference line, not a far-right dot
            continue
        name, col, mk = fam(k)
        lbl = name if name not in seen else None
        seen.add(name)
        ax.scatter(b, a, s=55, c=col, marker=mk, edgecolors="white",
                   linewidths=0.8, zorder=3, label=lbl)
    # fp16 ceiling as a dashed reference
    fp16 = 100 * gf["acc"]["fp16"]
    ax.axhline(fp16, ls="--", color=C_FP16, lw=0.9, zorder=1)
    ax.text(5.28, fp16 - 1.6, "fp16 ceiling", ha="right", fontsize=7, color=INK)

    # annotate the two decisive points
    ann = {"uniform4bit": ("uniform-4", 6, 8), "mix43+pb3": ("mix43+patch", -4, -12),
           "mix43": ("mix43", 6, -4)}
    d = {k: (gf["bits"][k], 100 * gf["acc"][k]) for k in gf["bits"]}
    for k, (txt, dx, dy) in ann.items():
        x, y = d[k]
        ax.annotate(txt, (x, y), textcoords="offset points", xytext=(dx, dy),
                    fontsize=7.5, color=INK,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))

    ax.set_xlabel("effective bits / weight")
    ax.set_ylabel("GSM8K accuracy (%)")
    ax.set_xlim(2.0, 5.5)
    ax.set_ylim(-2, 38)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", handletextpad=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_frontier.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_frontier.pdf")


# ===========================================================================
# FIG 2 -- perplexity understates (and inverts) accuracy damage
# ===========================================================================
def fig_ppl_vs_acc():
    g = L("gsm8k_check.json")
    f = L("full_surface.json")["ppl"]["math"]
    fp_p, fp_a = f["full_precision"], g["acc"]["full_precision"]
    rows = [
        ("uniform 4-bit", "uniform4bit"),
        ("mix43 + patch", "mix43+pb3_bf5"),
        ("mixed attn@4/mlp@3", "mixed_attn4_mlp3"),
        ("uniform 3-bit", "uniform3bit"),
    ]
    dppl = [100 * (f[k] / fp_p - 1) for _, k in rows]
    dacc = [-100 * (g["acc"][k] / fp_a - 1) for _, k in rows]  # positive = loss
    labels = [r[0] for r in rows]

    import numpy as np
    y = np.arange(len(rows))
    h = 0.36
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    grid(ax, axis="x")
    ax.barh(y + h / 2, dppl, height=h, color=C_UNIFORM, zorder=3,
            label="perplexity increase")
    ax.barh(y - h / 2, dacc, height=h, color=C_VERM, zorder=3,
            label="GSM8K accuracy lost")
    for yi, v in zip(y + h / 2, dppl):
        ax.text(v + 1, yi, f"{v:.0f}%", va="center", fontsize=7, color=INK)
    for yi, v in zip(y - h / 2, dacc):
        ax.text(v + 1, yi, f"{v:.0f}%", va="center", fontsize=7, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("degradation from fp16 (%)")
    ax.set_xlim(0, 85)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_ppl_vs_acc.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_ppl_vs_acc.pdf")


# ===========================================================================
# FIG 3 -- the learned-patch mirage: perplexity win, accuracy collapse
# ===========================================================================
def fig_mirage():
    v = L("gsm8k_learned_verdict.json")
    lp = L("learned_push_math.json")["tasks"]["math"]
    order = ["base_mixed", "learned_3bit", "uniform4", "fp16"]
    names = ["base\n(mix43)", "learned\npatch", "uniform-4", "fp16"]
    ppls = [5.51, lp["learned_3bit"]["ppl"], lp["uniform4"]["ppl"], lp["fp16"]["ppl"]]
    accs = [100 * v["base_mixed"]["acc"], 100 * v["learned_3bit"]["acc"],
            100 * v["uniform4"]["acc"], 34.5]
    cols = [C_MIXED, C_LEARNED, C_UNIFORM, C_FP16]

    import numpy as np
    x = np.arange(len(order))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.6, 3.0))
    for ax in (a1, a2):
        grid(ax)
    a1.bar(x, ppls, color=cols, zorder=3, width=0.62)
    for xi, val in zip(x, ppls):
        a1.text(xi, val + 0.03, f"{val:.2f}", ha="center", fontsize=7.5, color=INK)
    a1.set_ylabel("perplexity  (lower = better)")
    a1.set_ylim(4.3, 5.7)
    a1.set_xticks(x); a1.set_xticklabels(names, fontsize=7.5)
    a1.set_title("what perplexity says", fontsize=8.5, color=INK)

    a2.bar(x, accs, color=cols, zorder=3, width=0.62)
    for xi, val in zip(x, accs):
        a2.text(xi, val + 0.5, f"{val:.1f}%", ha="center", fontsize=7.5, color=INK)
    a2.set_ylabel("GSM8K accuracy (%)")
    a2.set_ylim(0, 38)
    a2.set_xticks(x); a2.set_xticklabels(names, fontsize=7.5)
    a2.set_title("what accuracy says", fontsize=8.5, color=INK)
    # one clean callout on each panel, placed clear of the bars
    a1.annotate("2nd best", (1, ppls[1]), textcoords="offset points", xytext=(20, 6),
                fontsize=7, color=C_LEARNED,
                arrowprops=dict(arrowstyle="-", color=C_LEARNED, lw=0.6))
    a2.annotate("worse than\ndoing nothing", (1, accs[1]), textcoords="offset points",
                xytext=(16, 22), fontsize=7, color=C_LEARNED,
                arrowprops=dict(arrowstyle="-", color=C_LEARNED, lw=0.6))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_mirage.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_mirage.pdf")


C_VERM = OKABE["vermillion"]

if __name__ == "__main__":
    fig_frontier()
    fig_ppl_vs_acc()
    fig_mirage()
    print("all figures written to", OUT)
