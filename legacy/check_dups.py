"""
Duplicate / Near-Duplicate Check
Verifies there's no leakage between train/val/test splits.
This is the #1 reason results can look "too good to be true."

Run this locally where you have the dataset.
"""

import os
import pandas as pd
import imagehash
from PIL import Image
from sklearn.model_selection import train_test_split
from collections import defaultdict

from paths import TRAIN_CSV, TRAIN_IMAGES

SEED = 42

# ── Rebuild the exact same split as train.py ──
df = pd.read_csv(TRAIN_CSV)
all_images = [f for f in os.listdir(TRAIN_IMAGES) if f.endswith(".jpg")]
defective_ids = set(df["ImageId"].unique())

label_df = pd.DataFrame({
    "image_id": all_images,
    "label": [1 if img in defective_ids else 0 for img in all_images]
})

train_df, temp_df = train_test_split(
    label_df, test_size=0.30, random_state=SEED, stratify=label_df["label"]
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=SEED, stratify=temp_df["label"]
)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ── Step 1: Exact duplicate check (sanity check — should be 0) ──
train_ids = set(train_df["image_id"])
val_ids = set(val_df["image_id"])
test_ids = set(test_df["image_id"])

overlap_train_val = train_ids & val_ids
overlap_train_test = train_ids & test_ids
overlap_val_test = val_ids & test_ids

print(f"\n── Exact ID overlap (should all be 0) ──")
print(f"Train ∩ Val:  {len(overlap_train_val)}")
print(f"Train ∩ Test: {len(overlap_train_test)}")
print(f"Val ∩ Test:   {len(overlap_val_test)}")

# ── Step 2: Perceptual hash near-duplicate check ──
# This catches images that are visually near-identical (e.g. sequential crops
# from the same steel coil) even though they have different filenames.
print(f"\n── Computing perceptual hashes (this may take a few minutes) ──")

def compute_hash(image_id, image_dir):
    path = os.path.join(image_dir, image_id)
    img = Image.open(path)
    return imagehash.phash(img, hash_size=16)  # higher = more precise

# Hash all images, grouped by split
train_hashes = {}
for img_id in train_df["image_id"]:
    train_hashes[img_id] = compute_hash(img_id, TRAIN_IMAGES)
print(f"Hashed {len(train_hashes)} train images")

test_hashes = {}
for img_id in test_df["image_id"]:
    test_hashes[img_id] = compute_hash(img_id, TRAIN_IMAGES)
print(f"Hashed {len(test_hashes)} test images")

# ── Step 3: Find near-duplicates between train and test ──
# Hamming distance threshold: 0 = identical, <=5 = near-identical (tune as needed)
THRESHOLD = 5

near_dupes = []
for test_id, test_hash in test_hashes.items():
    for train_id, train_hash in train_hashes.items():
        dist = test_hash - train_hash  # Hamming distance
        if dist <= THRESHOLD:
            near_dupes.append((test_id, train_id, dist))

print(f"\n── Near-duplicate pairs found (Hamming distance <= {THRESHOLD}) ──")
print(f"Count: {len(near_dupes)}")
if near_dupes:
    print("\nSample pairs (test_image, train_image, distance):")
    for pair in near_dupes[:10]:
        print(f"  {pair}")
    pct = len(near_dupes) / len(test_df) * 100
    print(f"\n⚠️  {pct:.1f}% of test images have a near-duplicate in train.")
    print("This could be inflating your test accuracy.")
else:
    print("✓ No near-duplicates found. Your high accuracy is NOT due to leakage.")