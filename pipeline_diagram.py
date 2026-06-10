"""
Renders a side-by-side pipeline diagram for train.py and demo.py.
Output: results/pipeline_diagram.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from paths import RESULTS_DIR

# ── layout constants ──────────────────────────────────────────────────────────
BOX_W, BOX_H = 3.2, 0.52
GAP           = 0.28          # vertical gap between boxes
COL_LEFT      = 1.2           # x centre of train.py column
COL_RIGHT     = 6.0           # x centre of demo.py column
FIG_W, FIG_H  = 11, 13

COLORS = {
    "io":      "#dfe6e9",   # grey  — data / file I/O
    "process": "#d6eaf8",   # blue  — computation
    "model":   "#d5f5e3",   # green — model ops
    "output":  "#fdebd0",   # orange — saved artefacts
    "header":  "#2c3e50",   # dark  — column headers
}

# ── pipeline definitions ──────────────────────────────────────────────────────
# Each step: (label, color_key, optional_sub_label)
TRAIN_STEPS = [
    ("paths.py",                      "io",      "DATA_DIR · TRAIN_CSV · MODEL_PATH · RESULTS_DIR"),
    ("eda.py  →  run_eda()",          "process", "class balance · sample images"),
    ("Dataset Preparation",           "process", "label_df  |  70 / 15 / 15 split"),
    ("Augmentation & Transforms",     "process", "train: flip, jitter  |  val/test: resize+norm"),
    ("SteelDataset + DataLoader",     "process", "batch=32, pin_memory"),
    ("EfficientNet-B0\n(ImageNet pretrained)", "model", "replace classifier head → 2 classes"),
    ("Class-Weighted Loss + Adam\n+ CosineAnnealingLR", "model", "lr=1e-4 · epochs=15"),
    ("Training Loop",                 "model",   "save best val-acc → models/best_model.pth"),
    ("Training Curves",               "output",  "results/training_curves.png"),
    ("Test Set Evaluation",           "model",   "accuracy · AUC · classification report"),
    ("Confusion Matrix / ROC / PR",   "output",  "results/evaluation_metrics.png"),
    ("Failure Case Analysis",         "output",  "results/false_positives/negatives.png"),
    ("Inference Demo",                "output",  "results/demo_inference.png"),
]

DEMO_STEPS = [
    ("paths.py",                      "io",      "TRAIN_CSV · TRAIN_IMAGES · MODEL_PATH"),
    ("Rebuild Test Split",            "process", "same SEED=42  →  identical held-out set"),
    ("Load EfficientNet-B0",          "model",   "models/best_model.pth"),
    ("val_transform",                 "process", "resize 256² · ToTensor · Normalize"),
    ("Random Test Image",             "io",      "sample 1 row from test_df"),
    ("predict()",                     "model",   "softmax → P(defect), P(no defect)"),
    ("Plot: image + prob bars",       "output",  "shown interactively (plt.show)"),
]


def _draw_column(ax, steps, cx, y_start, title, title_color):
    """Draw a column of boxes with arrows; return list of box centre-y coords."""
    centres = []
    y = y_start
    for label, ckey, sublabel in steps:
        by = y - BOX_H / 2
        rect = mpatches.FancyBboxPatch(
            (cx - BOX_W / 2, by), BOX_W, BOX_H,
            boxstyle="round,pad=0.06",
            linewidth=1.2,
            edgecolor="#7f8c8d",
            facecolor=COLORS[ckey],
            zorder=3,
        )
        ax.add_patch(rect)
        ax.text(cx, y, label,
                ha="center", va="center", fontsize=8.5,
                fontweight="bold", zorder=4)
        if sublabel:
            ax.text(cx, y - 0.19, sublabel,
                    ha="center", va="center", fontsize=6.6,
                    color="#555", style="italic", zorder=4)
        centres.append(y)
        y -= BOX_H + GAP

    # arrows
    for i in range(len(centres) - 1):
        ax.annotate(
            "", xy=(cx, centres[i + 1] + BOX_H / 2 + 0.03),
            xytext=(cx, centres[i] - BOX_H / 2 - 0.03),
            arrowprops=dict(arrowstyle="-|>", color="#34495e", lw=1.3),
            zorder=5,
        )

    # column header
    ax.text(cx, y_start + 0.65, title,
            ha="center", va="center", fontsize=11,
            fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=title_color, linewidth=0))

    return centres


def _add_legend(ax, x, y):
    items = [
        (COLORS["io"],      "Data / File I/O"),
        (COLORS["process"], "Processing step"),
        (COLORS["model"],   "Model operation"),
        (COLORS["output"],  "Saved artefact"),
    ]
    for i, (color, label) in enumerate(items):
        p = mpatches.Patch(facecolor=color, edgecolor="#7f8c8d", label=label, linewidth=1)
        ax.text(x + 0.35, y - i * 0.38, label, va="center", fontsize=8, color="#2c3e50")
        rect = mpatches.FancyBboxPatch(
            (x, y - i * 0.38 - 0.13), 0.28, 0.26,
            boxstyle="round,pad=0.03", facecolor=color,
            edgecolor="#7f8c8d", linewidth=1, zorder=3
        )
        ax.add_patch(rect)


def main():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(-0.5, FIG_H)
    ax.axis("off")

    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")

    ax.text(FIG_W / 2, FIG_H - 0.35,
            "Steel Defect Detection — Pipeline Overview",
            ha="center", va="center", fontsize=14,
            fontweight="bold", color=COLORS["header"])

    y_start = FIG_H - 1.1
    _draw_column(ax, TRAIN_STEPS, COL_LEFT,  y_start, "train.py",  "#2980b9")
    _draw_column(ax, DEMO_STEPS,  COL_RIGHT, y_start, "demo.py",   "#27ae60")

    # shared artefact annotation: best_model.pth bridge
    train_model_y = y_start - 7 * (BOX_H + GAP)   # step index 7 = "Training Loop"
    demo_load_y   = y_start - 2 * (BOX_H + GAP)   # step index 2 = "Load EfficientNet-B0"

    mid_x = (COL_LEFT + COL_RIGHT) / 2
    ax.annotate(
        "", xy=(COL_RIGHT - BOX_W / 2 - 0.05, demo_load_y),
        xytext=(COL_LEFT  + BOX_W / 2 + 0.05, train_model_y),
        arrowprops=dict(arrowstyle="-|>", color="#8e44ad", lw=1.6,
                        connectionstyle="arc3,rad=-0.25"),
        zorder=6,
    )
    ax.text(mid_x + 0.1, (train_model_y + demo_load_y) / 2 + 0.1,
            "models/best_model.pth",
            ha="center", fontsize=7.5, color="#8e44ad",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                      edgecolor="#8e44ad", linewidth=1))

    _add_legend(ax, 0.25, 2.3)

    plt.tight_layout(pad=0.4)
    out = RESULTS_DIR / "pipeline_diagram.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
