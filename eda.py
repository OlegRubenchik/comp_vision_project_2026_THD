"""
Exploratory Data Analysis — Steel Defect Detection
"""

import os, random
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image



def run_eda(train_csv, train_images_dir, results_dir):
    print("\n── EDA ──")

    df = pd.read_csv(train_csv)
    print(f"CSV shape: {df.shape}")
    print(df.head())

    all_images = [f for f in os.listdir(train_images_dir) if f.endswith(".jpg")]
    total_images = len(all_images)
    print(f"\nTotal training images: {total_images}")

    defective_ids = set(df["ImageId"].unique())
    no_defect_ids = set(all_images) - defective_ids
    n_defect = len(defective_ids)
    n_no_defect = len(no_defect_ids)
    print(f"Images with defects:    {n_defect} ({n_defect/total_images*100:.1f}%)")
    print(f"Images without defects: {n_no_defect} ({n_no_defect/total_images*100:.1f}%)")

    class_counts = df["ClassId"].value_counts().sort_index()
    print(f"\nDefect class distribution:\n{class_counts}")

    # Plot 1: Class imbalance bar chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(["No Defect", "Defect"], [n_no_defect, n_defect],
                color=["#2ecc71", "#e74c3c"], edgecolor="black")
    axes[0].set_title("Binary Label Distribution", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Number of Images")
    for i, v in enumerate([n_no_defect, n_defect]):
        axes[0].text(i, v + 50, str(v), ha="center", fontweight="bold")

    axes[1].bar([f"Class {i}" for i in class_counts.index], class_counts.values,
                color=["#3498db", "#e67e22", "#9b59b6", "#1abc9c"], edgecolor="black")
    axes[1].set_title("Defect Class Distribution (among defective images)", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Number of Instances")
    for i, v in enumerate(class_counts.values):
        axes[1].text(i, v + 30, str(v), ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(results_dir / "eda_class_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: eda_class_distribution.png")

    # Plot 2: Sample images
    _show_sample_images(train_images_dir, defective_ids, no_defect_ids, results_dir=results_dir)

    # Image size sanity check
    sample_img = Image.open(os.path.join(train_images_dir, all_images[0]))
    print(f"\nOriginal image size: {sample_img.size} (W x H)")
    print(f"Mode: {sample_img.mode}")

    return all_images, defective_ids, no_defect_ids


def _show_sample_images(image_dir, defective_ids, no_defect_ids, n=4, results_dir=None):
    fig, axes = plt.subplots(2, n, figsize=(16, 5))
    fig.suptitle("Sample Images: No Defect (top) vs Defect (bottom)", fontsize=13, fontweight="bold")

    no_def_sample = random.sample(list(no_defect_ids), n)
    def_sample = random.sample(list(defective_ids), n)

    for i, (nd, d) in enumerate(zip(no_def_sample, def_sample)):
        img_nd = Image.open(os.path.join(image_dir, nd))
        img_d  = Image.open(os.path.join(image_dir, d))
        axes[0, i].imshow(img_nd, cmap="gray", aspect="auto")
        axes[0, i].set_title("No Defect", color="#2ecc71", fontweight="bold")
        axes[0, i].axis("off")
        axes[1, i].imshow(img_d, cmap="gray", aspect="auto")
        axes[1, i].set_title("Defect", color="#e74c3c", fontweight="bold")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(results_dir / "eda_sample_images.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: eda_sample_images.png")
