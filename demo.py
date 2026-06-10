"""
Steel Defect Detection — Live Demo (with ground truth)
Uses the held-out test split from train_images — model never saw these.
Run multiple times to get different random images.
"""

import os, random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights
from sklearn.model_selection import train_test_split

# ── CONFIG ──
from paths import TRAIN_CSV, TRAIN_IMAGES, MODEL_PATH

IMG_SIZE     = 256
SEED         = 42
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── REBUILD TEST SPLIT (same as main.py) ──
df = pd.read_csv(TRAIN_CSV)
all_images = [f for f in os.listdir(TRAIN_IMAGES) if f.endswith(".jpg")]
defective_ids = set(df["ImageId"].unique())

label_df = pd.DataFrame({
    "image_id": all_images,
    "label": [1 if img in defective_ids else 0 for img in all_images]
})

_, temp_df = train_test_split(label_df, test_size=0.30, random_state=SEED,
                               stratify=label_df["label"])
_, test_df = train_test_split(temp_df, test_size=0.50, random_state=SEED,
                               stratify=temp_df["label"])

print(f"Test split size: {len(test_df)} images")

# ── LOAD MODEL ──
model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ── PREDICT ──
def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]
        p_no_defect = probs[0].item()
        p_defect    = probs[1].item()
    pred = "DEFECT" if p_defect > 0.5 else "NO DEFECT"
    return pred, p_defect, p_no_defect, img

# ── PICK RANDOM IMAGE FROM TEST SPLIT ──
row = test_df.sample(1).iloc[0]
image_path = os.path.join(TRAIN_IMAGES, row["image_id"])
true_label = "DEFECT" if row["label"] == 1 else "NO DEFECT"

pred, p_defect, p_no_defect, img = predict(image_path)
correct = pred == true_label

# ── PLOT ──
fig, axes = plt.subplots(1, 2, figsize=(12, 4),
                          gridspec_kw={"width_ratios": [3, 1]})

pred_color = "#2ecc71" if pred == "NO DEFECT" else "#e74c3c"
result_str = "✓ CORRECT" if correct else "✗ WRONG"
result_color = "#2ecc71" if correct else "#e74c3c"

axes[0].imshow(img, cmap="gray", aspect="auto")
axes[0].set_title(
    f"True label: {true_label}     {result_str}",
    fontsize=12, fontweight="bold", color=result_color
)
axes[0].axis("off")

bars = axes[1].barh(["No Defect", "Defect"],
                    [p_no_defect, p_defect],
                    color=["#2ecc71", "#e74c3c"],
                    edgecolor="black")
axes[1].set_xlim(0, 1)
axes[1].set_title(f"Predicted: {pred}", fontsize=13,
                   fontweight="bold", color=pred_color)
axes[1].set_xlabel("Probability")
for bar, val in zip(bars, [p_no_defect, p_defect]):
    axes[1].text(val + 0.01, bar.get_y() + bar.get_height()/2,
                 f"{val:.3f}", va="center", fontweight="bold")

plt.suptitle(f"Image: {row['image_id']}", fontsize=9, color="gray")
plt.tight_layout()
plt.show()