"""
All data preparation for the Steel Defect Detection project.
Imported by train.py and demo.py.

Responsibilities:
  - Assign a class label (0-4) to every image
  - Split into train / val / test
  - Build PyTorch Dataset and DataLoader objects
"""

import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split


IMG_SIZE = 256

# ── Transforms ────────────────────────────────────────────────────────────────
# Training: augmentation (random flips, jitter) helps the model generalise
# Val/Test: only resize and normalise — no randomness, consistent evaluation
# ImageNet mean/std are used because EfficientNet was pretrained on ImageNet

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


# ── Label assignment ──────────────────────────────────────────────────────────

def _count_rle_pixels(rle_string):
    """Count the number of pixels in a run-length encoded mask.

    RLE format: start1 length1 start2 length2 ...
    We sum all the length values (every second number).
    """
    numbers = list(map(int, rle_string.split()))
    return sum(numbers[1::2])


def build_label_dataframe(train_csv, train_images_dir):
    """Return a DataFrame with columns [image_id, label].

    Label assignment:
      - 0   → image has no entry in the CSV (no defect)
      - 1-4 → defect class with the most pixels in that image
               (some images have multiple defect types; we pick the dominant one)

    The CSV only contains defective images. Absence from the CSV = No Defect.
    When an image has multiple defect classes we pick the one covering the
    most pixels — the most dominant defect.
    """
    df = pd.read_csv(train_csv)
    df["pixel_count"] = df["EncodedPixels"].apply(_count_rle_pixels)

    # For each image keep only the class with the most pixels
    dominant = (
        df.sort_values("pixel_count", ascending=False)  # biggest defect first
          .drop_duplicates(subset="ImageId")             # first row per image = dominant defect
          [["ImageId", "ClassId"]]                       # drop pixel_count and EncodedPixels
          .rename(columns={"ImageId": "image_id", "ClassId": "label"})  # match our column names
    )

    # All images on disk (includes images not in CSV = no defect)
    all_files = sorted(f for f in os.listdir(train_images_dir) if f.endswith(".jpg"))
    all_df = pd.DataFrame({"image_id": all_files})  # one row per image file

    # Left merge: every image stays, matched images get their defect label
    # images with no match in dominant (not in CSV) get NaN → filled with 0 (No Defect)
    label_df = all_df.merge(dominant, on="image_id", how="left")
    label_df["label"] = label_df["label"].fillna(0).astype(int)  # NaN → 0

    return label_df


# ── Dataset ───────────────────────────────────────────────────────────────────

class SteelDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df        = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["image_id"])
        image    = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(row["label"], dtype=torch.long)
        return image, label


# ── Split & DataLoaders ───────────────────────────────────────────────────────

def make_splits(label_df, seed):
    """Split label_df into train / val / test (70 / 15 / 15)."""
    # stratify ensures each split has the same class ratio as the full dataset
    # train_test_split only splits into two parts, so we do it twice:
    # step 1: 70% train, 30% temp
    train_df, temp_df = train_test_split(
        label_df, test_size=0.30, random_state=seed, stratify=label_df["label"]
    )
    # step 2: temp split 50/50 → 15% val, 15% test
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=seed, stratify=temp_df["label"]
    )
    return train_df, val_df, test_df


def make_dataloaders(train_df, val_df, test_df, image_dir, batch_size, num_workers):
    """Build Dataset and DataLoader objects for all three splits."""
    train_dataset = SteelDataset(train_df, image_dir, train_transform)
    val_dataset   = SteelDataset(val_df,   image_dir, val_transform)
    test_dataset  = SteelDataset(test_df,  image_dir, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader
