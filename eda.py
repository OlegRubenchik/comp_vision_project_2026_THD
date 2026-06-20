"""
Exploratory Data Analysis — Steel Defect Detection
Research and presentation file. Safe to delete — train.py does not import this.

Run this independently to explore the dataset before training:
  python eda.py
"""

from paths import TRAIN_CSV, TRAIN_IMAGES, RESULTS_DIR, CLASS_NAMES
from dataset import build_label_dataframe
from plots import plot_class_distribution, plot_sample_images
from PIL import Image
import os


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


if __name__ == "__main__":
    run_eda()
