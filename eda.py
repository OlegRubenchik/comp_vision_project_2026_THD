"""
Exploratory Data Analysis — Steel Defect Detection
Research and presentation file. Safe to delete — train.py does not import this.

Run this independently to explore the dataset before training:
  python eda.py
"""

import os
import random
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torchvision import transforms

from paths import TRAIN_CSV, TRAIN_IMAGES, RESULTS_DIR, CLASS_NAMES
from dataset import build_label_dataframe, IMG_SIZE
from plots import plot_class_distribution, plot_sample_images


def save_sample_images_per_class(label_df, image_dir, class_names, results_dir):
    """Save one raw image per class to results/sample_images/ for use in presentations."""
    out_dir = Path(results_dir) / "sample_images"
    out_dir.mkdir(exist_ok=True)

    for cls, name in enumerate(class_names):
        subset = label_df[label_df["label"] == cls]["image_id"].tolist()
        sample = random.choice(subset)
        img = Image.open(os.path.join(image_dir, sample))
        filename = f"class_{cls}_{name.lower().replace(' ', '_')}.jpg"
        img.save(out_dir / filename)
        print(f"Saved: {filename}")

    print(f"Sample images saved to: {out_dir}")


def save_augmentation_slide(image_dir, results_dir):
    """Save a side-by-side augmentation example for the presentation slide."""
    # Pick a defective image (class 1-4) so the augmentations are visible
    all_files = sorted(f for f in os.listdir(image_dir) if f.endswith(".jpg"))
    img_path  = os.path.join(image_dir, random.choice(all_files[500:600]))
    original  = Image.open(img_path).convert("RGB")

    resized         = transforms.Resize((IMG_SIZE, IMG_SIZE))(original)
    h_flipped       = transforms.RandomHorizontalFlip(p=1.0)(resized)
    v_flipped       = transforms.RandomVerticalFlip(p=1.0)(resized)
    color_jittered  = transforms.ColorJitter(brightness=0.4, contrast=0.4)(resized)

    panels = [
        (original,       f"Original\n{original.size[0]}×{original.size[1]}px"),
        (resized,        f"Resize\n{IMG_SIZE}×{IMG_SIZE}px"),
        (h_flipped,      "Horizontal\nFlip"),
        (v_flipped,      "Vertical\nFlip"),
        (color_jittered, "Color\nJitter"),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    fig.suptitle("Data Augmentation Pipeline", fontsize=14, fontweight="bold", y=1.02)

    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img, cmap="gray", aspect="auto")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    out = Path(results_dir) / "augmentation_slide.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out}")


def run_eda():
    print("\n── EDA ──")

    label_df = build_label_dataframe(TRAIN_CSV, TRAIN_IMAGES)

    # ── Stats ─────────────────────────────────────────────────────────────────
    counts = label_df["label"].value_counts().sort_index()
    print(f"Total images: {len(label_df)}")
    print("\nClass distribution:")
    for cls, name in enumerate(CLASS_NAMES):
        n = counts.get(cls, 0)
        print(f"  Class {cls} ({name}): {n} ({n/len(label_df)*100:.1f}%)")

    # ── Image size info ───────────────────────────────────────────────────────
    first = label_df.iloc[0]["image_id"]
    img   = Image.open(os.path.join(TRAIN_IMAGES, first))
    print(f"\nOriginal image size: {img.size} (W x H), mode: {img.mode}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_class_distribution(label_df, CLASS_NAMES, RESULTS_DIR)
    plot_sample_images(label_df, TRAIN_IMAGES, CLASS_NAMES, RESULTS_DIR)
    save_sample_images_per_class(label_df, TRAIN_IMAGES, CLASS_NAMES, RESULTS_DIR)
    save_augmentation_slide(TRAIN_IMAGES, RESULTS_DIR)


if __name__ == "__main__":
    run_eda()
