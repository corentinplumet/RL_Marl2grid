"""Encoder-only transfer, NLS inputs: frozen bus14 encoder, new WCCI heads.

Left: mean survival over all 50 chronics.  Right: the 24 difficult chronics.
Independent axes - the do-nothing reference differs by cohort.

    python3 encoder_frozen_transfer.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import os

LABELS = ["Basic\nbusbar graph",
          "Busbar +\nsame-substation\nedges",
          "Disaggregated\n+ physical\nmessage direction",
          "Disaggregated\nbidirectional\nmessage direction"]
ALL50     = [18.1, 41.5, 59.8, 56.2];  DN_ALL  = 56.0
DIFFICULT = [ 8.2,  8.5, 14.1, 13.4];  DN_DIFF = 8.34
COLS = ["#A9B0DC", "#7C86C8", "#2E9E8F", "#1F7A73"]
INK = "#413C3A"; MUT = "#8A8785"

fig, axes = plt.subplots(1, 2, figsize=(13.1, 5.0), dpi=200)
panels = ((axes[0], ALL50, DN_ALL, 72, "all 50 chronics",
           "mean survival (%)"),
          (axes[1], DIFFICULT, DN_DIFF, 18, "the 24 difficult chronics",
           "difficult-cohort survival (%)"))
for ax, vals, dn, ymax, title, ylab in panels:
    x = np.arange(len(vals))
    ax.bar(x, vals, color=COLS, width=0.64, zorder=3)
    ax.axhline(dn, ls="--", lw=1.7, color=INK, zorder=4)
    off = ymax * 0.022
    for xi, v in zip(x, vals):
        ax.text(xi, v + off, "%.1f%%" % v, ha="center", va="bottom",
                fontsize=15, fontweight="bold", color=INK, zorder=5)
    lab = ("do nothing  %.2f%%" % dn) if dn < 10 else ("do nothing  %.1f%%" % dn)
    ax.legend(handles=[Line2D([0], [0], ls="--", lw=1.7, color=INK, label=lab)],
              loc="upper left", frameon=False, fontsize=11.5, labelcolor=INK,
              bbox_to_anchor=(-0.012, 1.02), handlelength=1.9)
    ax.set_xticks(x); ax.set_xticklabels(LABELS, fontsize=10, color=INK)
    ax.set_ylim(0, ymax); ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#DDDDDD", lw=0.9); ax.xaxis.grid(False)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color("#BBBBBB")
    ax.tick_params(colors=INK, labelsize=11, length=0)
    ax.set_ylabel(ylab, fontsize=12, color=INK)
    ax.set_title(title, fontsize=15.5, color=INK, fontweight="bold", pad=16)

fig.suptitle("NLS  —  frozen encoder, trainable head", fontsize=13,
             color=MUT, y=0.975)
fig.subplots_adjust(left=0.062, right=0.988, top=0.83, bottom=0.17, wspace=0.24)
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "encoder_frozen_transfer.png"), facecolor="white")
print("wrote encoder_frozen_transfer.png")
