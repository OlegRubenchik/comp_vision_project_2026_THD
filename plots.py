"""
All plotting functions for the Steel Defect Detection project.
Used by both eda.py (exploration plots) and train.py (training result plots).
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# ─────────────────────────────────────────────────────────────────────────────
# EDA plots  (called from eda.py)
# ─────────────────────────────────────────────────────────────────────────────

def plot_class_distribution(label_df, class_names, results_dir):
    """Bar chart showing how many images belong to each class."""
    counts = label_df["label"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(
        [f"Class {i}\n{class_names[i]}" for i in range(len(class_names))],
        [counts.get(i, 0) for i in range(len(class_names))],
        color=["#95a5a6", "#e74c3c", "#e67e22", "#3498db", "#2ecc71"],
        edgecolor="black"
    )
    ax.set_title("Class Distribution (5-class)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of Images")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                str(int(bar.get_height())), ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(results_dir / "eda_class_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: eda_class_distribution.png")


def plot_sample_images(label_df, image_dir, class_names, results_dir, n_per_class=2):
    """Show n_per_class sample images for each class."""
    n_classes = len(class_names)
    fig, axes = plt.subplots(n_classes, n_per_class,
                             figsize=(n_per_class * 5, n_classes * 2.5))
    fig.suptitle("Sample Images per Class", fontsize=13, fontweight="bold")
    colors = ["#95a5a6", "#e74c3c", "#e67e22", "#3498db", "#2ecc71"]

    for cls in range(n_classes):
        subset  = label_df[label_df["label"] == cls]["image_id"].tolist()
        samples = random.sample(subset, min(n_per_class, len(subset)))

        for j in range(n_per_class):
            ax = axes[cls][j]
            if j < len(samples):
                img = Image.open(os.path.join(image_dir, samples[j]))
                ax.imshow(img, cmap="gray", aspect="auto")
            ax.axis("off")
            if j == 0:
                ax.set_ylabel(f"Class {cls}\n{class_names[cls]}",
                              color=colors[cls], fontweight="bold",
                              fontsize=9, rotation=0, labelpad=80)

    plt.tight_layout()
    plt.savefig(results_dir / "eda_sample_images.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: eda_sample_images.png")


# ─────────────────────────────────────────────────────────────────────────────
# Training result plots  (called from train.py)
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(history, epochs, results_dir):
    """Plot loss and accuracy curves for train and validation sets."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs_x = range(1, epochs + 1)

    axes[0].plot(epochs_x, history["train_loss"], label="Train", marker="o")
    axes[0].plot(epochs_x, history["val_loss"],   label="Val",   marker="o")
    axes[0].set_title("Loss per Epoch", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_x, history["train_acc"], label="Train", marker="o")
    axes[1].plot(epochs_x, history["val_acc"],   label="Val",   marker="o")
    axes[1].set_title("Accuracy per Epoch", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: training_curves.png")


def plot_confusion_matrix(test_labels, test_preds, class_names, results_dir):
    """Plot a labelled confusion matrix for the test set."""
    cm = confusion_matrix(test_labels, test_preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix — Test Set", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(results_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: confusion_matrix.png")
    return cm


def plot_per_class_accuracy(cm, class_names, results_dir):
    """Bar chart showing accuracy for each individual class."""
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(class_names, per_class_acc,
                  color=["#95a5a6", "#e74c3c", "#e67e22", "#3498db", "#2ecc71"],
                  edgecolor="black")
    ax.set_ylim(0, 1.1)
    ax.set_title("Per-Class Accuracy — Test Set", fontsize=13, fontweight="bold")
    ax.set_ylabel("Accuracy")
    for bar, val in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f"{val:.2f}", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(results_dir / "per_class_accuracy.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: per_class_accuracy.png")


def plot_failure_cases(test_dataset, test_preds, test_labels, class_names, results_dir, n=6):
    """Plot n misclassified images with their true and predicted labels."""
    wrong_idx = np.where(test_preds != test_labels)[0]
    print(f"Misclassified: {len(wrong_idx)} / {len(test_labels)} "
          f"({len(wrong_idx)/len(test_labels)*100:.1f}%)")

    sample = wrong_idx[:n]
    if len(sample) == 0:
        print("No misclassified images found.")
        return

    fig, axes = plt.subplots(1, len(sample), figsize=(4 * len(sample), 3))
    if len(sample) == 1:
        axes = [axes]
    fig.suptitle("Misclassified Examples", fontsize=13, fontweight="bold", color="#e74c3c")

    for ax, idx in zip(axes, sample):
        img, _ = test_dataset[idx]
        img_np = img.numpy().transpose(1, 2, 0)
        img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_np = np.clip(img_np, 0, 1)
        ax.imshow(img_np)
        ax.set_title(
            f"True:  {class_names[test_labels[idx]]}\n"
            f"Pred:  {class_names[test_preds[idx]]}",
            fontsize=8, color="#e74c3c"
        )
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(results_dir / "failure_cases.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: failure_cases.png")
