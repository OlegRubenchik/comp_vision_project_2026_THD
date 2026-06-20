"""
Steel Defect Detection — Live Demo
Loads the trained model and predicts on a random image from the test split.

Run multiple times to see different images.
The test split is never shown to the model during training, so these are
genuine unseen predictions.
"""

import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights

from paths import TRAIN_CSV, TRAIN_IMAGES, MODEL_PATH, CLASS_NAMES, NUM_CLASSES
from dataset import build_label_dataframe, make_splits, val_transform

SEED   = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Rebuild the same test split as train.py ───────────────────────────────────
label_df = build_label_dataframe(TRAIN_CSV, TRAIN_IMAGES)
_, _, test_df = make_splits(label_df, SEED)
print(f"Test split: {len(test_df)} images")

# ── Load model ────────────────────────────────────────────────────────────────
model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model = model.to(DEVICE)
model.eval()

# ── Predict ───────────────────────────────────────────────────────────────────
def predict(image_path):
    """Return predicted class index, all class probabilities, and the PIL image."""
    from PIL import Image
    img    = Image.open(image_path).convert("RGB")
    tensor = val_transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
    return int(probs.argmax()), probs, img


# ── Pick a random test image ──────────────────────────────────────────────────
row        = test_df.sample(1).iloc[0]
image_path = os.path.join(TRAIN_IMAGES, row["image_id"])
true_class = int(row["label"])

pred_class, probs, img = predict(image_path)
correct = pred_class == true_class

# ── Plot ──────────────────────────────────────────────────────────────────────
bar_colors   = ["#95a5a6", "#e74c3c", "#e67e22", "#3498db", "#2ecc71"]
result_color = "#2ecc71" if correct else "#e74c3c"
result_str   = "CORRECT" if correct else "WRONG"

fig, axes = plt.subplots(1, 2, figsize=(12, 4),
                          gridspec_kw={"width_ratios": [3, 2]})

axes[0].imshow(img, cmap="gray", aspect="auto")
axes[0].set_title(f"True: {CLASS_NAMES[true_class]}     [{result_str}]",
                   fontsize=12, fontweight="bold", color=result_color)
axes[0].axis("off")

bars = axes[1].barh(CLASS_NAMES, probs, color=bar_colors, edgecolor="black")
axes[1].set_xlim(0, 1.15)
axes[1].set_title(f"Predicted: {CLASS_NAMES[pred_class]}",
                   fontsize=12, fontweight="bold", color=bar_colors[pred_class])
axes[1].set_xlabel("Probability (softmax)")
for bar, val in zip(bars, probs):
    axes[1].text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", fontweight="bold", fontsize=9)

plt.suptitle(f"Image: {row['image_id']}", fontsize=9, color="gray")
plt.tight_layout()
plt.show()
