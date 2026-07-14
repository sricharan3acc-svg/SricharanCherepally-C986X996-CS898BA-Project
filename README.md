# Automated Casting Defect Detection — CS898BA Project

**Author:** Sricharan Cherepally (WSUID: c986x996)

Computer vision system for detecting defects in cast metal parts, comparing a classical feature-engineering approach against CNN transfer learning.

## Dataset
"Real-Life Industrial Dataset of Casting Product" (Kaggle, ravirajsinh45). 7,348 grayscale images (300×300), 2 classes: defective / normal. Split: 6,633 train (3,758 defective / 2,875 normal), 715 test (453 defective / 262 normal).

## Repository Structure

```
src/
  preprocessing.py            - CLAHE, thresholding+morphology, Canny, blob analysis
  visualize_pipeline.py       - generates before/after pipeline visualizations
  features.py                 - LBP + GLCM + edge density + blob feature extraction
  extract_dataset_features.py - runs feature extraction across full train/test split
  train_baseline.py           - trains/evaluates SVM and Random Forest (Approach A)
  train_cnn.py                - trains/evaluates MobileNetV2 transfer learning (Approach B)
  sanity_check.py             - validates blob-detection separation across samples
outputs/
  preprocessing_examples/     - sample pipeline visualizations
  features/                   - extracted feature arrays (X/y, train/test)
  baseline_results/           - classical model metrics, confusion matrices
  cnn_results/                - CNN metrics, confusion matrix
AI_Log.md                     - log of AI-assisted development
README.md                     - this file
hello_world.py                - initial environment check
```

## Setup

```
pip install opencv-python numpy scikit-learn scikit-image matplotlib pillow torch torchvision
```

## Running the Pipeline

Edit the `DATA_DIR` / `DATA_ROOT` path near the top of each script to point at your local copy of the dataset, then run in order from inside `src/`:

```
python visualize_pipeline.py
python extract_dataset_features.py   # ~5 min, processes all 7,348 images
python train_baseline.py
python train_cnn.py                   # requires internet access for pretrained weights
```

## Approach A: Classical Feature Extraction + ML

Thresholding and morphological operations isolate the part's rim and inner ring. Texture descriptors (Local Binary Patterns, Gray-Level Co-occurrence Matrix properties), edge density, and blob/void statistics are combined into a 29-dimension feature vector and fed into an SVM (RBF kernel) and a Random Forest classifier.

## Approach B: CNN Transfer Learning

Images are preprocessed (CLAHE contrast enhancement) and fed into a pretrained MobileNetV2 with a fine-tuned classification head, output as a defective/normal probability per image.

## Results (Midterm)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| SVM (RBF) | 91.1% | 91.5% | 94.7% | 93.1% |
| **Random Forest** | **92.9%** | 94.1% | 94.7% | 94.4% |
| CNN (MobileNetV2, pretrained, first pass) | 86.2% | 84.0% | 96.5% | 89.8% |

All metrics evaluated on the full 715-image held-out test set.

## Discussion

The classical Random Forest baseline currently outperforms the first-pass CNN. This is expected: the CNN run only trained a frozen-backbone classifier head on a 1,200-image subset for 4 epochs, versus the classical model's use of the full 6,633-image training set. The final presentation will explore fine-tuning deeper layers and training on the complete dataset to test whether the CNN can close this gap, consistent with the literature's expectation that transfer learning should outperform hand-engineered features given sufficient training.

## Roadblocks & Pivots

- **Naive blob/void counting alone does not cleanly separate defective from normal images.** CLAHE amplifies the metal surface's natural grain, producing threshold speckle that inflates "candidate void" counts on both classes. Pivoted to rely primarily on LBP/GLCM texture descriptors, with blob stats as a secondary signal — this is what the Random Forest baseline's strong performance is actually built on.
- **Sandbox network restrictions initially blocked pretrained CNN weight downloads** during development (only package registries were reachable, not `download.pytorch.org`). Resolved by re-running the identical script on a local machine with full internet access, confirming `Pretrained ImageNet weights loaded: True`.

## Submission Deliverables

- GitHub repository (this repo)
- Unlisted/public YouTube video walkthrough
- Slide deck used in the recording