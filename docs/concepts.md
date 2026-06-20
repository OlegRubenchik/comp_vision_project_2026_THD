# Project Concepts & Code Explanations

Reference for understanding the key ideas behind this project.

---

## Label Assignment
**File:** `dataset.py` → `build_label_dataframe()`

The Kaggle CSV (`train.csv`) only lists **defective** images. There is no "No Defect" row — absence from the CSV means the image is clean.

**Problem 1 — No Defect images have no entry**
Every `.jpg` on disk is listed in a DataFrame. Then we do a left merge with the CSV. Images that didn't match get `label = 0` (No Defect).

**Problem 2 — One image can have multiple defect types**
The CSV can have multiple rows for the same image. We need one label per image for classification. Fix: count the pixels covered by each defect using RLE, keep only the class with the most pixels (dominant defect).

---

## RLE (Run-Length Encoding)
**File:** `dataset.py` → `_count_rle_pixels()`

How Kaggle stores defect masks compactly. Instead of a full pixel map, it stores pairs:

```
start length start length ...
```

Example: `5 3 12 2` means 3 pixels starting at position 5, 2 pixels starting at position 12.

`_count_rle_pixels()` sums all the `length` values (every second number) to get the total pixel count for a defect.

---

## Dominant Defect — pandas chain explained
**File:** `dataset.py` → `build_label_dataframe()`, lines 74–79

```python
dominant = (
    df.sort_values("pixel_count", ascending=False)  # biggest defect first
      .drop_duplicates(subset="ImageId")             # first row per image = dominant defect
      [["ImageId", "ClassId"]]                       # drop pixel_count and EncodedPixels
      .rename(columns={"ImageId": "image_id", "ClassId": "label"})
)
```

1. Sort by pixel count descending — largest defect comes first per image
2. Drop duplicates keeping only the first row per image — that's the dominant defect
3. Keep only the two columns we need
4. Rename to match column names used everywhere else

---

## Left Merge
**File:** `dataset.py` → `build_label_dataframe()`, lines 86–87

```python
label_df = all_df.merge(dominant, on="image_id", how="left")
label_df["label"] = label_df["label"].fillna(0).astype(int)
```

Left merge: every image stays in the result. Images with no match in `dominant` (not in CSV) get `NaN` in the label column. `fillna(0)` replaces those with 0 = No Defect. `.astype(int)` converts from float (pandas default after fillna) to integer class index.

---

## Stratified Split
**File:** `dataset.py` → `make_splits()`

```python
train_test_split(..., stratify=label_df["label"])
```

Preserves the class distribution in each subset. If Defect 2 is 5% of the full dataset, it will also be ~5% of train, val, and test. Without it, rare classes could end up mostly in one split.

`train_test_split` only splits into two parts at once, so we do it twice:
- Step 1: 70% train, 30% temp
- Step 2: temp split 50/50 → 15% val, 15% test

---

## Tensor
**Concept used throughout:** `dataset.py`, `train.py`

A multi-dimensional array of numbers — the core data structure PyTorch uses.

- 1D → list of numbers
- 2D → matrix
- 3D → image: `(channels, height, width)` — a 256×256 RGB image = tensor of shape `(3, 256, 256)`
- 4D → batch of images: `(batch_size, channels, height, width)`

PyTorch can run tensor operations on the GPU and tracks gradients through them for backpropagation.

---

## DataLoader
**File:** `dataset.py` → `make_dataloaders()`

Wraps a Dataset and handles batching:

- `batch_size=32` — groups 32 images into one tensor per iteration
- `shuffle=True` on train — randomises order each epoch so the model doesn't learn the data order
- `num_workers=0` — 0 = main process loads data (safe on Windows)
- `pin_memory=True` — loads into page-locked CPU memory for faster GPU transfer

---

## Transfer Learning vs Fine-Tuning
**File:** `train.py` → section 5 (MODEL)

**Feature extraction** — freeze all pretrained layers, only train the new final layer (~5000 weights). Faster, needs less data, lower ceiling.

**Fine-tuning** (what we do) — load pretrained weights as a starting point, let the whole network update. Higher accuracy, needs more data (~12k images is enough here).

---

## EfficientNet-B0 & the Final Layer
**File:** `train.py` → section 5, lines 106–111

EfficientNet-B0 is pretrained on ImageNet (1.2M images, 1000 classes). It already knows edges, textures, shapes.

Its `classifier` block has two layers:
- `[0]` — Dropout
- `[1]` — Linear (the classification layer)

We replace only `[1]`:
```python
# Original: Linear(1280 → 1000) for ImageNet
# Ours:     Linear(1280 → 5)    for our defect classes
in_features = model.classifier[1].in_features  # = 1280
model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
```

`in_features` is read from the existing layer instead of hardcoding 1280 so the code works if you swap to a different architecture.

---

## nn.Linear
**File:** `train.py` → section 5, line 111

A fully connected layer — every input connected to every output with a learnable weight.

```
output = input × weights + bias
```

- `input` — 1280-number feature vector from EfficientNet
- `weights` — 1280×5 matrix learned during training
- `output` — 5 scores (logits), one per class

Highest score = prediction. Scores go into CrossEntropyLoss during training, or softmax during inference to get probabilities that sum to 1.

---

## Class Weights
**File:** `train.py` → section 5, lines 125–130

```python
len(label_df) / (NUM_CLASSES * counts.get(c, 1))
```

Formula: `total_images / (5 × class_count)`. Rare classes get a higher weight. A wrong prediction on Defect 2 (225 images) hurts more than on No Defect (5900 images). Passed to `CrossEntropyLoss(weight=class_weights)`.

---

## Training — The Core Concept
**File:** `train.py` → `train_one_epoch()`

The model has ~5.3M weights. At the start they're random. Training adjusts them until predictions are good.

**One step:**

1. **Forward pass** — image goes through the model, produces 5 scores e.g. `[0.1, 0.6, 0.05, 0.2, 0.05]`
2. **Loss** — compare scores to true label. CrossEntropy converts scores to probabilities and measures how far off you are
3. **Backward pass** — `loss.backward()` walks back through every layer and computes: *"if I increase this weight slightly, does the loss go up or down?"* — that's the gradient
4. **Update** — `optimizer.step()` nudges every weight in the direction that reduces the loss

Repeat for every batch, every epoch. Weights gradually settle into values that produce correct predictions.

**Why `loss.backward()` and not `model.backward()`?**
PyTorch builds a computation graph during the forward pass — `loss` carries a reference to every operation that produced it, all the way back through the model. So calling `loss.backward()` automatically walks back through the entire model and stores gradients inside every parameter.

---

## Adam Optimizer
**File:** `train.py` → section 5, line 133

Adam adjusts weights after each batch using gradients from backpropagation. Unlike basic SGD (fixed learning rate for everything), Adam tracks a running average of past gradients per weight — weights that rarely update get a bigger step, frequently updated ones get smaller steps. Converges faster and requires less tuning.

---

## CosineAnnealingLR
**File:** `train.py` → section 5, line 134

Reduces the learning rate over training following a cosine curve — starts at `1e-4`, smoothly drops to near zero by epoch 15. Helps the model make smaller, more precise adjustments at the end of training instead of overshooting.

---

## evaluate() vs train_one_epoch()
**File:** `train.py` → sections 6 and 8

Key differences:

| | train_one_epoch | evaluate |
|---|---|---|
| `model.train()` / `model.eval()` | train | eval |
| `torch.no_grad()` | no | yes |
| `loss.backward()` | yes | no |
| `optimizer.step()` | yes | no |
| collects all preds | no | yes |

`model.eval()` disables dropout and changes batch norm — without it, evaluation results would vary each run.

`torch.no_grad()` skips building the computation graph since we don't need gradients — saves memory and runs faster.
