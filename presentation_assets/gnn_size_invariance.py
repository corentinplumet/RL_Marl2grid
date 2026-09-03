"""Reminder figure: why a GNN is size-invariant.

One shared function at every node (message passing), then a readout that
collapses however many node vectors into one fixed-length graph vector.

    python3 gnn_size_invariance.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
import os

INK="#413C3A"; MUT="#8A8785"
NODE_E="#5B6BC0"; NODE_F="#C5CAE9"
TEAL_E="#1F7A73"; TEAL_F="#B7E2DE"
EDGE="#A9A5A3"; BOX_E="#C9C5C3"

def scatter_graph(n, seed, spread=(1.65, 0.82), min_d=0.92, extra=2):
    """Irregular node layout plus a spanning tree and a few extra edges."""
    rng = np.random.default_rng(seed)
    pts = []; tries = 0
    while len(pts) < n:
        tries += 1
        if tries > 400:                      # relax rather than spin
            min_d *= 0.9; tries = 0
        c = (rng.uniform(-spread[0], spread[0]), rng.uniform(-spread[1], spread[1]))
        if all((c[0]-p[0])**2 + (c[1]-p[1])**2 > min_d**2 for p in pts):
            pts.append(c)
    order = list(rng.permutation(n))
    links = [(order[i], order[rng.integers(0, i)]) for i in range(1, n)]
    cand = [(i, j) for i in range(n) for j in range(i+1, n)
            if (i, j) not in links and (j, i) not in links]
    cand.sort(key=lambda e: (pts[e[0]][0]-pts[e[1]][0])**2
                          + (pts[e[0]][1]-pts[e[1]][1])**2)
    links += cand[:extra]
    return pts, links

def draw_graph(ax, pts, links, cx, cy, r=0.30, label=None):
    for i, j in links:
        ax.plot([pts[i][0]+cx, pts[j][0]+cx], [pts[i][1]+cy, pts[j][1]+cy],
                lw=1.5, color=EDGE, zorder=2)
    for x, y in pts:
        ax.add_patch(Circle((x+cx, y+cy), r, fc=TEAL_F, ec=TEAL_E, lw=1.8,
                            zorder=4))
        ax.text(x+cx, y+cy, "θ", ha="center", va="center", fontsize=8.5,
                color=TEAL_E, fontweight="bold", zorder=5)
    if label is not None:
        ax.text(cx, cy+label[1], label[0], ha="center", fontsize=11.5,
                color=INK)

def stage(ax, x0, x1, y0, y1, title):
    ax.add_patch(FancyBboxPatch((x0, y0), x1-x0, y1-y0,
        boxstyle="round,pad=0.05,rounding_size=0.22", fc="white", ec=BOX_E,
        lw=1.6, zorder=1))
    ax.text((x0+x1)/2, y1+0.34, title, ha="center", fontsize=12.5, color=INK,
            fontweight="bold")

def arrow(ax, x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
        mutation_scale=19, lw=2.2, color=TEAL_E, zorder=6))

fig, ax = plt.subplots(figsize=(13.2, 4.05), dpi=200)
ax.set_xlim(0, 25.2); ax.set_ylim(-4.35, 3.95); ax.axis("off"); ax.set_aspect("equal")

# ---- stage 1: any graph -------------------------------------------------
stage(ax, 0.5, 6.6, -3.05, 3.05, "any graph")
pA, lA = scatter_graph(5, 11, extra=1)
pB, lB = scatter_graph(7, 5, extra=2)
draw_graph(ax, pA, lA, 3.55, 1.85, label=("5 nodes", -1.42))
draw_graph(ax, pB, lB, 3.55, -1.55, label=("7 nodes", -1.40))
arrow(ax, 6.9, 8.3, 0)

# ---- stage 2: message passing ------------------------------------------
stage(ax, 8.6, 14.6, -3.05, 3.05, "message passing")
cx, cy = 11.6, 0.75
for p in [(-1.5, 1.15), (-1.6, -0.75), (0.6, 1.55)]:
    ax.add_patch(FancyArrowPatch((p[0]+cx, p[1]+cy), (cx, cy), arrowstyle="-|>",
        mutation_scale=15, lw=1.9, color=TEAL_E, zorder=3, shrinkA=12, shrinkB=20))
    ax.add_patch(Circle((p[0]+cx, p[1]+cy), 0.30, fc=NODE_F, ec=NODE_E, lw=1.6,
                        zorder=4))
ax.add_patch(Circle((cx, cy), 0.46, fc=TEAL_F, ec=TEAL_E, lw=2.0, zorder=5))
ax.text(cx, cy, "v", ha="center", va="center", fontsize=12, color=TEAL_E,
        fontweight="bold", zorder=6)
ax.text(11.6, -1.70, r"the same  $f_{\theta}$  at every node",
        ha="center", fontsize=11.5, color=INK)
ax.text(11.6, -2.55, "K rounds", ha="center", fontsize=11, color=MUT)
arrow(ax, 14.9, 16.3, 0)

# ---- stage 3: readout ---------------------------------------------------
stage(ax, 16.6, 24.7, -3.05, 3.05, "readout")
for k, (yc, n) in enumerate(((1.85, 5), (-1.55, 7))):
    for i in range(n):
        ax.add_patch(Rectangle((17.15 + i*0.34, yc-0.55), 0.24, 1.10,
                               fc=TEAL_F, ec=TEAL_E, lw=1.1, zorder=4))
    ax.text(17.15 + n*0.17, yc-1.28, "%d vectors" % n, ha="center",
            fontsize=10.5, color=MUT)
    ax.add_patch(FancyArrowPatch((17.15 + n*0.34 + 0.30, yc), (20.55, yc),
        arrowstyle="-|>", mutation_scale=14, lw=1.8, color=TEAL_E, zorder=5))
ax.add_patch(FancyBboxPatch((20.7, -0.85), 1.55, 1.70,
    boxstyle="round,pad=0.03,rounding_size=0.14", fc=TEAL_F, ec=TEAL_E,
    lw=1.9, zorder=5))
ax.text(21.48, 0.16, "mean", ha="center", fontsize=10.5, color=TEAL_E,
        fontweight="bold", zorder=6)
ax.text(21.48, -0.46, "or max", ha="center", fontsize=10.5, color=TEAL_E,
        fontweight="bold", zorder=6)
ax.add_patch(FancyArrowPatch((22.45, 0), (23.35, 0), arrowstyle="-|>",
    mutation_scale=15, lw=2.0, color=TEAL_E, zorder=5))
ax.add_patch(Rectangle((23.55, -0.75), 0.34, 1.50, fc="#7FCFC7", ec=TEAL_E,
                       lw=1.9, zorder=5))
ax.text(23.72, 1.15, "z", ha="center", fontsize=13, color=TEAL_E,
        fontweight="bold", style="italic")
ax.text(23.72, -1.30, "fixed length", ha="center", fontsize=10.5, color=MUT)

ax.text(12.6, -3.90, "the weights and the output size never depend on how "
        "many nodes there are", ha="center", fontsize=12.5, color=INK,
        fontweight="bold")

fig.subplots_adjust(left=0.004, right=0.996, top=0.90, bottom=0.005)
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "gnn_size_invariance.png"), facecolor="white")
print("wrote gnn_size_invariance.png")
