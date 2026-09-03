"""Final-result ladder: what crosses from bus14 to WCCI36.

All values are mean survival on the 24 difficult WCCI chronics.

    python3 transfer_ladder.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os

LABELS = ["Do nothing\n$\\it{reference}$",
          "Zero-shot\ntransfer\n$\\it{median\\ of\\ 8}$",
          "Trained from\nscratch on WCCI\n$\\it{median\\ of\\ 8}$",
          "Frozen encoder\n+ new head\n$\\it{best\\ of\\ 4}$",
          "Fine-tuned\nfrom bus14\n$\\it{median\\ of\\ 8}$",
          "Transferred +\nlocal gate\n$\\it{best\\ cell}$",
          "One-step greedy\nat $k=64$\n$\\it{queries\\ simulator}$"]
VALS = [8.34, 5.7, 9.6, 14.1, 18.1, 25.8, 67.9]
COLS = ["#8c8c8c", "#4F8FCC", "#B07AA1", "#7C86C8", "#D0661C", "#00A79F",
        "#4CAF50"]
DN = 8.34
INK = "#413C3A"

fig, ax = plt.subplots(figsize=(13.4, 4.6), dpi=200)
x = np.arange(len(VALS))
ax.bar(x, VALS, color=COLS, width=0.62, zorder=3)
ax.axhline(DN, ls="--", lw=1.6, color=INK, zorder=4)
for xi, v in zip(x, VALS):
    ax.text(xi, v + 1.7, "%.1f%%" % v, ha="center", va="bottom", fontsize=15,
            fontweight="bold", color=INK)
ax.legend(handles=[Line2D([0], [0], ls="--", lw=1.6, color=INK,
                          label="do-nothing reference (8.34%)")],
          loc="upper left", frameon=False, fontsize=11.5, labelcolor=INK,
          bbox_to_anchor=(0.01, 1.0))
ax.set_xticks(x); ax.set_xticklabels(LABELS, fontsize=11, color=INK)
ax.set_ylabel("Mean survival on the 24 difficult\nWCCI chronics (%)",
              fontsize=11.5, color=INK)
ax.set_ylim(0, 80); ax.set_axisbelow(True)
ax.yaxis.grid(True, color="#DDDDDD", lw=0.9); ax.xaxis.grid(False)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"): ax.spines[sp].set_color("#BBBBBB")
ax.tick_params(colors=INK, labelsize=10.5, length=0)
fig.subplots_adjust(bottom=0.26, top=0.965, left=0.088, right=0.99)
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "transfer_ladder.png"), facecolor="white")
print("wrote transfer_ladder.png")
