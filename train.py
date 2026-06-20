"""
Steel Defect Detection — 5-Class Classification
Computer Vision Project — DIT Deggendorf, Summer 2026
Authors: Oleg Rubenchik, Shamil Liman

Task:
  Classify each steel surface image into one of 5 classes:
    0 = No Defect
    1 = Defect type 1
    2 = Defect type 2
    3 = Defect type 3
    4 = Defect type 4

Dataset: Severstal Steel Defect Detection (Kaggle)

Method:
  Transfer learning with EfficientNet-B0 (pretrained on ImageNet).
  Lecture reference: "Selected architectures" (Slides-3, p.217)

Pipeline:
  EDA → Dataset → Model → Train → Evaluate → Plots
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
from paths import TRAIN_CSV, TRAIN_IMAGES, RESULTS_DIR, MODEL_PATH, CLASS_NAMES, NUM_CLASSES

BATCH_SIZE  = 32
EPOCHS      = 15
LR          = 1e-4
SEED        = 42
NUM_WORKERS = 0

# ─────────────────────────────────────────────────────────────────────────────
# 1. IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import random, warnings
import numpy as np

from sklearn.metrics import classification_report

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights
from tqdm import tqdm

from dataset import build_label_dataframe, make_splits, make_dataloaders
from plots import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_per_class_accuracy,
    plot_failure_cases,
)

warnings.filterwarnings("ignore")

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# 2. DEVICE
# Lecture reference: "Efficient training — Use thousands of cores on a GPU"
#                   (Slides-3, p.210)
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    raise RuntimeError(
        "No GPU detected. Training on CPU will be very slow.\n"
        "Check that your CUDA drivers and PyTorch CUDA build are installed correctly.\n"
        "To force CPU anyway, comment out this line."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. LOAD & LABEL DATA
# ─────────────────────────────────────────────────────────────────────────────
label_df = build_label_dataframe(TRAIN_CSV, TRAIN_IMAGES)


# ─────────────────────────────────────────────────────────────────────────────
# 4. DATASET PREPARATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Dataset Preparation ──")
print(f"Label distribution:\n{label_df['label'].value_counts().sort_index()}")

train_df, val_df, test_df = make_splits(label_df, SEED)
print(f"\nTrain: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

train_dataset, val_dataset, test_dataset, \
train_loader, val_loader, test_loader = make_dataloaders(
    train_df, val_df, test_df, TRAIN_IMAGES, BATCH_SIZE, NUM_WORKERS
)
print(f"Batches — Train: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. MODEL — Transfer Learning with EfficientNet-B0
# Lecture reference: "Selected architectures" (Slides-3, p.217)
#   Pre-trained on 1.2M ImageNet images. We replace only the final layer
#   to output 5 scores instead of 1000 (fine-tuning).
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Model: EfficientNet-B0 with Transfer Learning ──")

model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

# Original final layer: Linear(1280 → 1000) for ImageNet
# Ours:                 Linear(1280 → 5)    for our 5 defect classes
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)

model = model.to(DEVICE)

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params:     {total_params:,}")
print(f"Trainable params: {trainable_params:,}")

# ── Loss, Optimizer, Scheduler ────────────────────────────────────────────────
# Class weights: penalise mistakes on rare classes more heavily
# Adam: adjusts weights after each batch to reduce loss
# CosineAnnealingLR: gradually reduces learning rate for smoother convergence

counts = label_df["label"].value_counts().sort_index()
class_weights = torch.tensor(
    [len(label_df) / (NUM_CLASSES * counts.get(c, 1)) for c in range(NUM_CLASSES)],
    dtype=torch.float32
).to(DEVICE)
print(f"\nClass weights: {class_weights.cpu().numpy().round(3)}")

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


# ─────────────────────────────────────────────────────────────────────────────
# 6. TRAINING FUNCTIONS
# Lecture reference: "Loss functions and optimization" (Slides-3, p.201)
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer):
    """One full pass over training data: forward → loss → backprop → update weights."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, leave=False, desc="Train"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()           # clear gradients from previous step
        outputs = model(images)         # forward pass
        loss    = criterion(outputs, labels)
        loss.backward()                 # backpropagation
        optimizer.step()                # update weights

        total_loss += loss.item() * images.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion):
    """Evaluate model on a data split — no weight updates."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, leave=False, desc="Eval"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs     = model(images)
            loss        = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            preds    = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += images.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


# ─────────────────────────────────────────────────────────────────────────────
# 7. TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Training ──")

history      = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
best_val_acc = 0.0
best_epoch   = 0

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc          = train_one_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc, _, _        = evaluate(model, val_loader, criterion)
    scheduler.step()

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch   = epoch
        torch.save(model.state_dict(), MODEL_PATH)
        marker = " ← best"
    else:
        marker = ""

    print(f"Epoch {epoch:2d}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}{marker}")

print(f"\nBest val accuracy: {best_val_acc:.4f} (epoch {best_epoch}/{EPOCHS})")


# ─────────────────────────────────────────────────────────────────────────────
# 8. TEST SET EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Test Evaluation ──")

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))

_, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion)
print(f"Test Accuracy: {test_acc:.4f}")
print("\nPer-class report:")
print(classification_report(test_labels, test_preds, target_names=CLASS_NAMES, digits=4))


# ─────────────────────────────────────────────────────────────────────────────
# 9. PLOTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Saving Plots ──")

plot_training_curves(history, EPOCHS, RESULTS_DIR)
cm = plot_confusion_matrix(test_labels, test_preds, CLASS_NAMES, RESULTS_DIR)
plot_per_class_accuracy(cm, CLASS_NAMES, RESULTS_DIR)
plot_failure_cases(test_dataset, test_preds, test_labels, CLASS_NAMES, RESULTS_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# 10. DONE
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Done ──")
print(f"Best Val Accuracy : {best_val_acc:.4f} (epoch {best_epoch}/{EPOCHS})")
print(f"Test Accuracy     : {test_acc:.4f}")
print(f"Model saved to    : {MODEL_PATH}")
