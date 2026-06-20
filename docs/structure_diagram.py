"""
File dependency diagram for the Steel Defect Detection project.
Shows which files import which, so the project structure is immediately clear.

Run from the project root:
  python docs/structure_diagram.py

Saves: docs/structure_diagram.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

DOCS_DIR = Path(__file__).parent

# ── Define files and their roles ──────────────────────────────────────────────
# Each entry: (file, role, color)

FILES = [
    ("paths.py",    "config",  "#dfe6e9"),
    ("dataset.py",  "data",    "#d6eaf8"),
    ("plots.py",    "visual",  "#fdebd0"),
    ("train.py",    "entry",   "#d5f5e3"),
    ("demo.py",     "entry",   "#d5f5e3"),
    ("eda.py",      "entry",   "#f9ebea"),
    ("docs/",       "docs",    "#e8daef"),
]

# ── Define import dependencies ─────────────────────────────────────────────────
# (importer, imported)
EDGES = [
    ("train.py",   "paths.py"),
    ("train.py",   "dataset.py"),
    ("train.py",   "plots.py"),
    ("demo.py",    "paths.py"),
    ("demo.py",    "dataset.py"),
    ("eda.py",     "paths.py"),
    ("eda.py",     "dataset.py"),
    ("eda.py",     "plots.py"),
    ("dataset.py", "paths.py"),
    ("docs/",      "paths.py"),
]

# ── Layout: manually position nodes for clarity ───────────────────────────────
POSITIONS = {
    "paths.py":   (0.50, 0.82),
    "dataset.py": (0.28, 0.54),
    "plots.py":   (0.72, 0.54),
    "train.py":   (0.14, 0.24),
    "demo.py":    (0.36, 0.24),
    "eda.py":     (0.58, 0.24),
    "docs/":      (0.80, 0.24),
}

COLORS = {
    "config": "#dfe6e9",
    "data":   "#d6eaf8",
    "visual": "#fdebd0",
    "entry":  "#d5f5e3",
    "docs":   "#e8daef",
}

ROLE_LABELS = {
    "config": "Config",
    "data":   "Data",
    "visual": "Visualisation",
    "entry":  "Entry point",
    "docs":   "Diagrams",
}


def main():
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#f8f9fa")

    ax.text(0.5, 0.97, "Project File Dependencies",
            ha="center", va="top", fontsize=14,
            fontweight="bold", color="#2c3e50")

    # ── Draw edges first (so nodes appear on top) ─────────────────────────────
    for src, dst in EDGES:
        x0, y0 = POSITIONS[src]
        x1, y1 = POSITIONS[dst]
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#7f8c8d",
                lw=1.2,
                connectionstyle="arc3,rad=0.05"
            ),
            zorder=1
        )

    # ── Draw nodes ────────────────────────────────────────────────────────────
    for filename, role, color in FILES:
        x, y = POSITIONS[filename]
        box = mpatches.FancyBboxPatch(
            (x - 0.09, y - 0.048), 0.18, 0.096,
            boxstyle="round,pad=0.02",
            linewidth=1.4,
            edgecolor="#7f8c8d",
            facecolor=color,
            zorder=2
        )
        ax.add_patch(box)
        ax.text(x, y + 0.013, filename,
                ha="center", va="center",
                fontsize=9, fontweight="bold", zorder=3)
        ax.text(x, y - 0.019, ROLE_LABELS.get(role, role),
                ha="center", va="center",
                fontsize=7, color="#555", style="italic", zorder=3)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(facecolor=COLORS["config"], edgecolor="#7f8c8d", label="Config"),
        mpatches.Patch(facecolor=COLORS["data"],   edgecolor="#7f8c8d", label="Data"),
        mpatches.Patch(facecolor=COLORS["visual"], edgecolor="#7f8c8d", label="Visualisation"),
        mpatches.Patch(facecolor=COLORS["entry"],  edgecolor="#7f8c8d", label="Entry point"),
        mpatches.Patch(facecolor=COLORS["docs"],   edgecolor="#7f8c8d", label="Docs"),
    ]
    ax.legend(handles=legend_items, loc="lower left",
              fontsize=8, framealpha=0.9, edgecolor="#ccc")

    plt.tight_layout()
    out = DOCS_DIR / "structure_diagram.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
