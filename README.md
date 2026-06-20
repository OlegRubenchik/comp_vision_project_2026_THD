# Steel Defect Detection

**Computer Vision Project — DIT Deggendorf, Summer 2026**
Authors: Oleg Rubenchik, Shamil Liman

---

## Task

Classify steel surface images into one of 5 classes:

| Class | Label |
|-------|-------|
| 0 | No Defect |
| 1 | Defect type 1 |
| 2 | Defect type 2 |
| 3 | Defect type 3 |
| 4 | Defect type 4 |

Dataset: [Severstal Steel Defect Detection (Kaggle)](https://www.kaggle.com/c/severstal-steel-defect-detection)

---

## Method

Transfer learning with **EfficientNet-B0** pretrained on ImageNet.
The final classification layer is replaced to output 5 class scores instead of 1000.
All weights are fine-tuned on the steel dataset.

---

## Project Structure

```
├── paths.py          # File paths and class names — edit this if you move the data
├── dataset.py        # Label assignment, train/val/test splits, DataLoaders
├── plots.py          # All matplotlib visualisation functions
├── train.py              # Training pipeline — run this to train the model
├── demo.py               # Live inference on a random test image
├── eda.py                # Exploratory data analysis (optional, for research only)
├── check_dups.py         # Perceptual hash check for train/test leakage (optional)
├── pipeline_diagram.py   # Generates the ML pipeline overview diagram
├── data/
│   ├── train.csv         # Kaggle annotations (ImageId, ClassId, EncodedPixels)
│   └── train_images/     # 12,568 steel surface images (.jpg)
├── models/
│   └── best_model.pth    # Saved after training
├── results/              # Runtime outputs — plots saved during training
└── docs/
    ├── structure_diagram.py   # Generates the file dependency diagram
    └── structure_diagram.png  # Visual map of how project files connect
```

---

## Installation

```bash
pip install -r requirements.txt
```

> **Note:** `torch` and `torchvision` in `requirements.txt` are built for **CUDA 12.1**.
> If you have a different CUDA version or no GPU, replace those two lines with the correct build from [pytorch.org](https://pytorch.org/get-started/locally/).

---

## How to Run

### 1. Explore the dataset (optional)
```bash
python eda.py
```
Prints class distribution and saves sample image plots to `results/`.

### 2. Train the model
```bash
python train.py
```
- Requires a CUDA-capable GPU
- Trains for 15 epochs, saves the best checkpoint to `models/best_model.pth`
- Saves all result plots to `results/`

### 3. Run live inference
```bash
python demo.py
```
Picks a random image from the held-out test split and shows the model's prediction with class probabilities.

---

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | ~92% |
| Best Val Accuracy | ~92% |

**Per-class accuracy:**

| Class | Accuracy |
|-------|----------|
| No Defect | 93% |
| Defect 1  | 89% |
| Defect 2  | 85% |
| Defect 3  | 88% |
| Defect 4  | 86% |

Defect 2 scores lowest due to having only ~225 training images vs ~5900 for No Defect.

---

## Key Design Decisions

**Label assignment** — The Kaggle CSV only contains defective images. Images absent from the CSV are assigned class 0 (No Defect). When an image has multiple defect types, the class covering the most pixels is used.

**Class weights** — The dataset is heavily imbalanced. `CrossEntropyLoss` is weighted so errors on rare classes are penalised more.

**Checkpoint saving** — The model is saved whenever validation accuracy improves, not just at the final epoch. This avoids overfitting in later epochs.
