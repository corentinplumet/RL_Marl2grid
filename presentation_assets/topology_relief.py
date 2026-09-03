"""Slide-3 schematic: moving elements onto the second busbar relieves an overload.

Drawn as a double-busbar single-line diagram. Both busbars exist in both panels;
on the left every element is switched to the first busbar and the second is
empty, on the right one generator and one line have been moved onto it.
A spur that crosses a busbar without a dot is not connected to it.

Schematic only - the percentages are illustrative, not a load-flow solution.

    python3 topology_relief.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import os

# ----------------------------------------------------------------- knobs
LOAD_BEFORE = ("118%", "60%")      # (short corridor, long corridor)
LOAD_AFTER  = ("87%",  "91%")
TITLES      = ("second busbar empty", "second busbar in use")
EMPTY_LABEL = "empty"
DIP         = -1.15                # how far the long corridor detours
OUT         = "topology_relief.png"

INK="#413C3A"; MUT="#8A8785"
BUS_E="#5B6BC0"; BUS_F="#C5CAE9"
BUS_OFF="#A6ADDC"
GEN_E="#D5504B"; GEN_F="#F7CFCD"
LOD_E="#43A047"; LOD_F="#C8E6C9"
HOT="#C62828"; OK="#2E9E5B"; MOVE="#E08A00"

RAIL_X0, RAIL_X1 = 1.28, 3.86      # left substation rails
Y1, Y2 = 0.62, -0.62               # first / second busbar
XG1, XG2, XLT, XLB = 1.68, 2.32, 2.98, 3.56    # spur positions
XB = 7.15                          # right substation
YT, YBO = 1.80, -1.80              # corridor heights


def rail(ax, y, live):
    ax.add_patch(FancyBboxPatch((RAIL_X0, y - 0.11), RAIL_X1 - RAIL_X0, 0.22,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        lw=2.2 if live else 2.0, ec=BUS_E if live else BUS_OFF,
        fc=BUS_F if live else "#F3F4FB",
        ls="-" if live else (0, (3.4, 2.4)), zorder=4))


def vbar(ax, x, y0, y1):
    ax.add_patch(FancyBboxPatch((x - 0.11, y0), 0.22, y1 - y0,
        boxstyle="round,pad=0.02,rounding_size=0.09",
        lw=2.0, ec=BUS_E, fc=BUS_F, zorder=4))


def node(ax, x, y, t, ec, fc):
    ax.add_patch(Circle((x, y), 0.40, lw=2.0, ec=ec, fc=fc, zorder=5))
    ax.text(x, y, t, ha="center", va="center", fontsize=15,
            fontweight="bold", color=ec, zorder=6)


def spur(ax, x, y_from, y_to, color, lw):
    """Vertical spur; breaks where it crosses the busbar it does not join."""
    lo, hi = sorted((y_from, y_to))
    for cross in (Y1, Y2):
        if lo < cross < hi and abs(cross - y_to) > 1e-6:
            ax.plot([x, x], [lo, cross - 0.20], lw=lw, color=color, zorder=3,
                    solid_capstyle="round")
            ax.plot([x, x], [cross + 0.20, hi], lw=lw, color=color, zorder=3,
                    solid_capstyle="round")
            return
    ax.plot([x, x], [lo, hi], lw=lw, color=color, zorder=3,
            solid_capstyle="round")


def dot(ax, x, y, color=BUS_E):
    ax.add_patch(Circle((x, y), 0.105, lw=0, fc=color, zorder=6))


def arc(x0, x1, y, dip):
    t = np.linspace(0, 1, 160)
    return x0 + (x1 - x0) * t, y + dip * np.sin(np.pi * t)


def panel(ax, moved):
    ax.set_xlim(-0.15, 9.35); ax.set_ylim(-3.9, 2.9)
    ax.set_aspect("equal"); ax.axis("off")

    rail(ax, Y1, True)
    rail(ax, Y2, moved)
    if not moved:
        ax.text(RAIL_X1 + 0.16, Y2, EMPTY_LABEL, fontsize=13.5,
                style="italic", color=BUS_OFF, ha="left", va="center",
                fontweight="bold")
    vbar(ax, XB, -2.35, 2.35)

    node(ax, 0.55, YT, "G", GEN_E, GEN_F)
    node(ax, 0.55, YBO, "G", GEN_E, GEN_F)
    node(ax, 8.80, 0.0, "L", LOD_E, LOD_F)
    ax.plot([XB, 8.40], [0, 0], lw=2.2, color=INK, zorder=3,
            solid_capstyle="round")

    # upper generator: always on the first busbar
    ax.plot([0.95, XG1], [YT, YT], lw=2.2, color=INK, zorder=3,
            solid_capstyle="round")
    spur(ax, XG1, YT, Y1, INK, 2.2); dot(ax, XG1, Y1)

    # lower generator: the element the action moves
    y_gen = Y2 if moved else Y1
    c_gen = MOVE if moved else INK
    ax.plot([0.95, XG2], [YBO, YBO], lw=2.6 if moved else 2.2, color=c_gen,
            zorder=3, solid_capstyle="round")
    spur(ax, XG2, YBO, y_gen, c_gen, 2.6 if moved else 2.2)
    dot(ax, XG2, y_gen, MOVE if moved else BUS_E)

    lab = LOAD_AFTER if moved else LOAD_BEFORE

    # short corridor: always on the first busbar
    col, w = (OK, 4.2) if moved else (HOT, 6.4)
    spur(ax, XLT, Y1, YT, col, w); dot(ax, XLT, Y1)
    X, Y = arc(XLT, XB, YT, 0.0)
    if not moved:
        ax.plot(X, Y, lw=w + 7, color=HOT, alpha=0.16, zorder=2,
                solid_capstyle="round")
    ax.plot(X, Y, lw=w, color=col, zorder=3, solid_capstyle="round")
    ax.text((XLT + XB) / 2, YT + 0.46, lab[0], ha="center", va="bottom",
            fontsize=19, fontweight="bold", color=col, zorder=6)

    # long corridor: the line the action moves
    col2, w2 = (OK, 4.2) if moved else (MUT, 2.6)
    y_line = Y2 if moved else Y1
    spur(ax, XLB, y_line, YBO, MOVE if moved else col2, w2)
    dot(ax, XLB, y_line, MOVE if moved else BUS_E)
    X, Y = arc(XLB, XB, YBO, DIP)
    ax.plot(X, Y, lw=w2, color=col2, zorder=3, solid_capstyle="round")
    ax.text((XLB + XB) / 2, Y[len(Y) // 2] - 0.46, lab[1], ha="center",
            va="top", fontsize=19, fontweight="bold", color=col2, zorder=6)


fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.85), dpi=200)
panel(axes[0], False); panel(axes[1], True)
for ax, t in zip(axes, TITLES):
    ax.set_title(t, fontsize=17, color=INK, pad=0)
fig.subplots_adjust(left=0.005, right=0.995, top=0.95, bottom=0.01, wspace=0.09)
fig.patches.append(FancyArrowPatch((0.489, 0.545), (0.523, 0.545),
    transform=fig.transFigure, arrowstyle="-|>", mutation_scale=34,
    lw=3.4, color=INK))
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), OUT),
            facecolor="white")
print("wrote", OUT)
