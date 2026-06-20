# Presentation Script — Steel Defect Detection
15 minutes total. Demo on slide 9.

---

## Slide 1 — Title
*"Steel Defect Detection using Computer Vision"*

We built a system that looks at steel surface images and classifies them — either clean, or one of four defect types. The goal is to automate what would otherwise be a manual quality control process in a factory.

---

## Slide 2 — Problem & Dataset

The dataset comes from a Kaggle competition by Severstal, a Russian steel manufacturer. They provided 12,568 images of steel surfaces taken on a production line.

The challenge: the images look very similar to a human eye. Defects are subtle — scratches, patches, inclusions — and the lighting and texture vary a lot between images. On top of that, the dataset is heavily imbalanced. Most images have no defect at all, and some defect types appear only a few hundred times.

The original Kaggle task was segmentation — draw a mask around every defect pixel. We turned it into a classification problem: given an image, predict which class it belongs to.

**5 classes:**
- 0 — No Defect
- 1, 2, 3, 4 — Four types of surface defects

---

## Slide 3 — Approach

We used a Convolutional Neural Network. CNNs are the standard for image recognition because they learn spatial features — edges, textures, patterns — directly from pixel data, rather than needing hand-crafted features.

Specifically we used **transfer learning** with EfficientNet-B0. Instead of training a network from scratch — which would require millions of images and days of compute — we started from a model already trained on ImageNet, which contains 1.2 million images across 1000 categories.

That model already knows how to detect edges, shapes, and textures. We kept all of that and only replaced the final layer — which originally output 1000 class scores — with a new layer that outputs 5 scores, one per defect class. Then we fine-tuned the entire network on our data.

---

## Slide 4 — Data Pipeline

**Label assignment** was the first real problem we had to solve.

The Kaggle CSV only contains rows for defective images. There's no "No Defect" entry — if an image isn't in the CSV, it's clean. So we listed every image on disk and merged it with the CSV. Images with no match got label 0.

The second problem: some images have multiple defect types. The CSV can have two or three rows for the same image — one for Defect 1, one for Defect 3, for example. For classification we need a single label. We resolved this by counting the pixels covered by each defect using the RLE masks in the CSV, and assigning the image to whichever defect type covered the most area — the dominant defect.

**Splits:** we used a stratified 70/15/15 train/validation/test split. Stratified means each split has the same class distribution as the full dataset — important when some classes have only a few hundred samples.

**Class imbalance:** No Defect has ~5900 training images. Defect 2 has only ~225. Without correction, the model would just predict "No Defect" for everything and get decent accuracy by sheer majority. We handled this by passing class weights to the loss function — errors on rare classes are penalised more heavily.

---

## Slide 5 — Model

EfficientNet-B0 is a compact but powerful architecture. It uses a technique called compound scaling — instead of just making the network deeper or wider, it scales depth, width, and input resolution together in a balanced ratio.

The network processes each image through many convolutional layers and compresses it down to a vector of 1280 numbers — a learned representation of the image. That vector is then fed into the final linear layer which maps it to 5 class scores.

We replaced that final layer and left everything else pretrained. Total parameters: ~5.3 million, all trainable.

---

## Slide 6 — Training

**Loss function:** CrossEntropyLoss with class weights. It measures how wrong the model's predicted probabilities are against the true label. The class weights scale errors so rare classes matter more.

**Optimizer:** Adam. After each batch of 32 images, Adam computes gradients — how much each weight contributed to the error — and nudges every weight slightly in the direction that reduces the loss. Adam is adaptive: it tracks per-weight learning history, so rarely-updated weights get bigger steps.

**Scheduler:** CosineAnnealingLR. The learning rate starts at 0.0001 and smoothly decreases to near zero over 15 epochs following a cosine curve. This helps the model make smaller, more precise adjustments at the end of training instead of overshooting.

**Checkpoint saving:** we save the model whenever validation accuracy improves, not just at the end. This is important — the best model in our run was at epoch 8. If we had just taken the final epoch we'd have a slightly worse model.

One thing we explicitly did not add was grid search for hyperparameters. Given the training time per run, manual tuning of a reasonable configuration was the practical choice.

---

## Slide 7 — Results

| Metric | Value |
|--------|-------|
| Test Accuracy | ~92% |
| Best Val Accuracy | ~92% |

**Per-class:**

| Class | Accuracy |
|-------|----------|
| No Defect | 93% |
| Defect 1  | 89% |
| Defect 2  | 85% |
| Defect 3  | 88% |
| Defect 4  | 86% |

Defect 2 is the weakest at 85%. This is expected — it has only 225 training examples compared to 5900 for No Defect. The class weights help, but there's a hard limit to what you can do with that little data.

Training accuracy reached ~95% against validation ~92% — a small gap, showing mild overfitting, but not severe enough to matter in practice.

---

## Slide 8 — Failure Cases

The model struggles most with Defect 2 — thin, low-contrast linear marks that are easy to miss. These also happen to look visually similar to some No Defect images with natural surface variation.

We also found an interesting label edge case: one image visibly had scratches but was labeled No Defect because it had no entry in the CSV. After investigating, we confirmed this is correct — Severstal's QC threshold means very faint scratches below a certain severity are intentionally not labeled. The model is working off industrial labels, not visual intuition.

---

## Slide 9 — Live Demo

*[Run demo.py a few times]*

This picks a random image from the held-out test set — images the model has never seen during training — and shows the prediction alongside softmax probabilities for each class.

The bar chart shows the model's confidence. On clear cases you'll see one bar close to 1.0 and the rest near zero. On harder cases the distribution spreads out more.

---

## Slide 10 — Conclusion & Limitations

We built a 5-class steel defect classifier reaching 92% test accuracy using transfer learning on EfficientNet-B0.

**Limitations:**
- Defect 2 is underperforming due to limited training data — more data or data augmentation specific to that class would help
- We do classification, not segmentation — we know what defect type is present but not where it is in the image
- The dominant-defect label strategy means images with multiple defect types are only partially represented

**What could be improved:**
- Segmentation would give more actionable output for a real factory
- More training data for rare classes, or synthetic augmentation
- Hyperparameter search if compute allows

---

*End of script.*
