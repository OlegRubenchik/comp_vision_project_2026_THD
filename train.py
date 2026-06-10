"""
Steel Defect Detection
Computer Vision Project — DIT Deggendorf, Summer 2026
Authors: Oleg Rubenchik, Shamil Liman

Task: Binary classification — defect vs. no defect on steel surface images
Dataset: Severstal Steel Defect Detection (Kaggle)
Method: Fine-tuned EfficientNet-B0 CNN
"""

# ─────────────────────────────────────────────
# 0. CONFIGURATION
# ─────────────────────────────────────────────
from paths import TRAIN_CSV, TRAIN_IMAGES, RESULTS_DIR, MODEL_PATH

IMG_SIZE = 256          # resize both dims to this (original: 256x1600, we crop/resize)
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-4
NUM_WORKERS = 0         # set to 0 if Windows multiprocessing issues
SEED = 42

# ─────────────────────────────────────────────
# 1. IMPORTS
# ─────────────────────────────────────────────
import os, random, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights
from tqdm import tqdm
from eda import run_eda

warnings.filterwarnings("ignore")
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(f"Using device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ─────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────────
all_images, defective_ids, no_defect_ids = run_eda(TRAIN_CSV, TRAIN_IMAGES, RESULTS_DIR)


# ─────────────────────────────────────────────
# 3. DATASET PREPARATION
# ─────────────────────────────────────────────
print("\n── Dataset Preparation ──")

# Build label dataframe
label_df = pd.DataFrame({
    "image_id": all_images,
    "label": [1 if img in defective_ids else 0 for img in all_images]
})
print(f"Label distribution:\n{label_df['label'].value_counts()}")

# Train / val / test split: 70 / 15 / 15
train_df, temp_df = train_test_split(
    label_df, test_size=0.30, random_state=SEED, stratify=label_df["label"]
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=SEED, stratify=temp_df["label"]
)
print(f"\nTrain: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# Transforms
# Images are 256x1600 — we resize to 256x256 (square crop from center)
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class SteelDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["image_id"])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(row["label"], dtype=torch.long)
        return image, label


train_dataset = SteelDataset(train_df, TRAIN_IMAGES, train_transform)
val_dataset   = SteelDataset(val_df,   TRAIN_IMAGES, val_transform)
test_dataset  = SteelDataset(test_df,  TRAIN_IMAGES, val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

print(f"Batches — Train: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}")


# ─────────────────────────────────────────────
# 4. MODEL
# ─────────────────────────────────────────────
print("\n── Model ──")

model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

# Replace final classifier for binary classification
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 2)

model = model.to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params:     {total_params:,}")
print(f"Trainable params: {trainable_params:,}")

# Class weights to handle imbalance (more no-defect than defect in our binary case)
n_pos = label_df["label"].sum()
n_neg = len(label_df) - n_pos
pos_weight = n_neg / n_pos
print(f"\nClass weight for defect class: {pos_weight:.2f}")

class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float32).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


# ─────────────────────────────────────────────
# 5. TRAINING
# ─────────────────────────────────────────────
print("\n── Training ──")

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in tqdm(loader, leave=False, desc="Train"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, leave=False, desc="Eval"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    return (total_loss / total, correct / total,
            np.array(all_preds), np.array(all_labels), np.array(all_probs))


history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
best_val_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc, _, _, _ = eval_epoch(model, val_loader, criterion)
    scheduler.step()

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_PATH)
        marker = " ← best"
    else:
        marker = ""

    print(f"Epoch {epoch:2d}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}{marker}")

print(f"\nBest val accuracy: {best_val_acc:.4f}")


# ─────────────────────────────────────────────
# 6. TRAINING CURVES
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
epochs_range = range(1, EPOCHS + 1)

axes[0].plot(epochs_range, history["train_loss"], label="Train", marker="o")
axes[0].plot(epochs_range, history["val_loss"],   label="Val",   marker="o")
axes[0].set_title("Loss", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs_range, history["train_acc"], label="Train", marker="o")
axes[1].plot(epochs_range, history["val_acc"],   label="Val",   marker="o")
axes[1].set_title("Accuracy", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "training_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: training_curves.png")


# ─────────────────────────────────────────────
# 7. TEST SET EVALUATION
# ─────────────────────────────────────────────
print("\n── Test Evaluation ──")

# Load best model
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

_, test_acc, test_preds, test_labels, test_probs = eval_epoch(model, test_loader, criterion)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test AUC:      {roc_auc_score(test_labels, test_probs):.4f}")
print("\nClassification Report:")
print(classification_report(test_labels, test_preds,
                             target_names=["No Defect", "Defect"], digits=4))

# ── Confusion Matrix ──
cm = confusion_matrix(test_labels, test_preds)
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=["No Defect", "Defect"],
            yticklabels=["No Defect", "Defect"])
axes[0].set_title("Confusion Matrix", fontsize=13, fontweight="bold")
axes[0].set_ylabel("True Label"); axes[0].set_xlabel("Predicted Label")

# ── ROC Curve ──
fpr, tpr, _ = roc_curve(test_labels, test_probs)
auc = roc_auc_score(test_labels, test_probs)
axes[1].plot(fpr, tpr, color="#e74c3c", lw=2, label=f"AUC = {auc:.4f}")
axes[1].plot([0, 1], [0, 1], "k--", lw=1)
axes[1].set_title("ROC Curve", fontsize=13, fontweight="bold")
axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

# ── Precision-Recall Curve ──
prec, rec, _ = precision_recall_curve(test_labels, test_probs)
axes[2].plot(rec, prec, color="#3498db", lw=2)
axes[2].set_title("Precision-Recall Curve", fontsize=13, fontweight="bold")
axes[2].set_xlabel("Recall"); axes[2].set_ylabel("Precision")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "evaluation_metrics.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: evaluation_metrics.png")


# ─────────────────────────────────────────────
# 8. FAILURE CASE ANALYSIS
# ─────────────────────────────────────────────
print("\n── Failure Case Analysis ──")

# Find false positives and false negatives
fp_idx = np.where((test_preds == 1) & (test_labels == 0))[0]  # predicted defect, actually fine
fn_idx = np.where((test_preds == 0) & (test_labels == 1))[0]  # missed defect

print(f"False Positives (no defect → predicted defect): {len(fp_idx)}")
print(f"False Negatives (defect → predicted no defect): {len(fn_idx)}")

def show_failure_cases(dataset, indices, title, n=4):
    if len(indices) == 0:
        print(f"No {title} found.")
        return
    sample_idx = indices[:n]
    fig, axes = plt.subplots(1, len(sample_idx), figsize=(4 * len(sample_idx), 3))
    if len(sample_idx) == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=13, fontweight="bold", color="#e74c3c")
    for i, idx in enumerate(sample_idx):
        img, label = dataset[idx]
        # Denormalize for display
        img_np = img.numpy().transpose(1, 2, 0)
        img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_np = np.clip(img_np, 0, 1)
        prob = test_probs[idx]
        axes[i].imshow(img_np)
        axes[i].set_title(f"True: {'Defect' if label==1 else 'No Defect'}\n"
                          f"P(defect)={prob:.2f}", fontsize=9)
        axes[i].axis("off")
    plt.tight_layout()
    fname = title.lower().replace(" ", "_") + ".png"
    plt.savefig(RESULTS_DIR / fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {fname}")

show_failure_cases(test_dataset, fp_idx, "False Positives", n=4)
show_failure_cases(test_dataset, fn_idx, "False Negatives", n=4)


# ─────────────────────────────────────────────
# 9. LIVE INFERENCE DEMO
# ─────────────────────────────────────────────
print("\n── Demo: Single Image Inference ──")

def predict_image(image_path, model, transform, device):
    """Predict defect probability for a single image."""
    model.eval()
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        prob = torch.softmax(output, dim=1)[0, 1].item()
    pred = "DEFECT" if prob > 0.5 else "NO DEFECT"
    return pred, prob, img

# Pick one defective and one clean image for demo
demo_defect   = os.path.join(TRAIN_IMAGES, list(defective_ids)[0])
demo_no_defect = os.path.join(TRAIN_IMAGES, list(no_defect_ids)[0])

fig, axes = plt.subplots(1, 2, figsize=(12, 3))
for ax, path, true_label in zip(axes, [demo_no_defect, demo_defect], ["No Defect", "Defect"]):
    pred, prob, img = predict_image(path, model, val_transform, DEVICE)
    color = "#2ecc71" if pred == "NO DEFECT" else "#e74c3c"
    ax.imshow(img, aspect="auto", cmap="gray")
    ax.set_title(f"True: {true_label}\nPredicted: {pred} (p={prob:.3f})",
                 color=color, fontweight="bold")
    ax.axis("off")
plt.suptitle("Live Inference Demo", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "demo_inference.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: demo_inference.png")

print("\n── Done ──")
print(f"Best Val Accuracy: {best_val_acc:.4f}")
print(f"Test Accuracy:     {test_acc:.4f}")
print(f"Test AUC:          {roc_auc_score(test_labels, test_probs):.4f}")