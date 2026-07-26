# Automated Casting Defect Detection - CS898BA Project

**Author:** Sricharan Cherepally (WSUID: c986x996)

Computer vision system for detecting defects in cast metal parts, comparing a classical feature-engineering approach against CNN transfer learning.

## Dataset
"Real-Life Industrial Dataset of Casting Product" (Kaggle, ravirajsinh45). 7,348 grayscale images (300x300), 2 classes: defective / normal. Split: 6,633 train (3,758 defective / 2,875 normal), 715 test (453 defective / 262 normal).

## Repository Structure

src/
preprocessing.py - CLAHE, thresholding+morphology, Canny, blob analysis
visualize_pipeline.py - generates before/after pipeline visualizations
features.py - LBP + GLCM + edge density + blob feature extraction
features_no_clahe.py - ablation variant of features.py, skips CLAHE
extract_dataset_features.py - runs feature extraction across full train/test split
run_clahe_ablation.py - trains classical models on no-CLAHE features for comparison
train_baseline.py - trains/evaluates SVM and Random Forest (Approach A)
train_cnn.py - initial CNN transfer learning run (Approach B, midterm)
train_cnn_experiment.py - configurable CNN training for hyperparameter sweeps
train_cnn_clean.py - final CNN run with train/test duplicates removed
save_model_for_demo.py - trains and saves the Random Forest model for the live demo
live_demo.py - virtual demonstration: live predictions on real test images
sanity_check.py - validates blob-detection separation across samples
outputs/
preprocessing_examples/ - sample pipeline visualizations
features/ - extracted feature arrays (X/y, train/test)
baseline_results/ - classical model metrics, confusion matrices
cnn_results/ - initial CNN metrics, confusion matrix (midterm)
hyperparam_results/ - results.csv from the 5-experiment hyperparameter sweep
cnn_clean_results/ - final leakage-verified CNN metrics
clahe_ablation/ - CLAHE ablation feature arrays and results
demo_model.joblib - saved Random Forest model used by live_demo.py
AI_Log.md - log of AI-assisted development
README.md - this file
hello_world.py - initial environment check


## Setup

pip install opencv-python numpy scikit-learn scikit-image matplotlib pillow torch torchvision joblib


## Running the Pipeline

Edit the DATA_DIR / DATA_ROOT path near the top of each script to point at your local copy of the dataset, then run in this order from inside src/:

python visualize_pipeline.py
python extract_dataset_features.py
python train_baseline.py
python run_clahe_ablation.py
python train_cnn.py
python train_cnn_experiment.py --name <name> --epochs N --lr X --unfreeze_layers K
python train_cnn_clean.py
python save_model_for_demo.py
python live_demo.py


## Approach A: Classical Feature Extraction + ML

Thresholding and morphological operations isolate the part's rim and inner ring. Texture descriptors (Local Binary Patterns, Gray-Level Co-occurrence Matrix properties), edge density, and blob/void statistics are combined into a 29-dimension feature vector and fed into an SVM (RBF kernel) and a Random Forest classifier.

## Approach B: CNN Transfer Learning

Images are preprocessed (CLAHE contrast enhancement) and fed into a pretrained MobileNetV2. The final model unfreezes the last 20 layers of the backbone for fine-tuning, trained for 12 epochs on the full training set (with train/test duplicate images removed).

## Image Analysis Evaluation

Tested whether CLAHE contrast enhancement actually improves classical model performance by extracting a second feature set with CLAHE skipped, then training identical SVM/Random Forest models for direct comparison.

| Model | With CLAHE | Without CLAHE |
|---|---|---|
| SVM (RBF) | 91.1% acc / 93.1% F1 | 93.7% acc / 95.2% F1 |
| Random Forest | 92.9% acc / 94.4% F1 | 92.3% acc / 94.2% F1 |

CLAHE's effect was small and mixed rather than clearly positive - see Roadblocks & Pivots for discussion.

## Hyperparameter Optimization

Five CNN configurations were tested on the full training set:

| Experiment | Unfreeze Layers | LR | Batch | Epochs | Accuracy | F1 |
|---|---|---|---|---|---|---|
| 1. Frozen backbone | 0 | 1e-3 | 32 | 6 | 92.6% | 94.3% |
| 2. Partial unfreeze | 20 | 1e-4 | 32 | 6 | 99.58% | 99.67% |
| 3. Full unfreeze | all | 1e-5 | 32 | 6 | 99.58% | 99.67% |
| 4. Bigger batch | 20 | 1e-4 | 64 | 6 | 99.72% | 99.78% |
| 5. More epochs | 20 | 1e-4 | 32 | 12 | 99.86% | 99.89% |

Unfreezing backbone layers was the single largest factor. Full vs. partial unfreezing made no measurable difference. Batch size and additional epochs each gave small further gains. Configuration 5 was selected as the final model.

## Data Integrity Check

Because the tuned CNN's near-100% result was high enough to warrant scrutiny, the train/test split was checked for exact duplicate images (via file hashing). 64 of 715 test images (9%) were found to be exact duplicates also present in the training set - entirely concentrated in the normal class (24.4% of normal test images; 0% of defective test images). These 64 images were removed from the training set, and the winning hyperparameter configuration was retrained from scratch. The result held: 99.86% accuracy on the same untouched test set, confirming the score was not an artifact of data leakage.

## Results (Final)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| SVM (RBF) | 91.1% | 91.5% | 94.7% | 93.1% |
| Random Forest | 92.9% | 94.1% | 94.7% | 94.4% |
| CNN, first pass (midterm) | 86.2% | 84.0% | 96.5% | 89.8% |
| CNN, tuned + leakage-verified | 99.86% | 100% | 99.8% | 99.9% |

All metrics evaluated on the full 715-image held-out test set.

## Virtual Demonstration

live_demo.py loads the saved Random Forest model and runs live predictions on 6 real test images (3 defective, 3 normal) never seen during training, displaying each image with its prediction, confidence, and correctness in a single updating window alongside console output. See submitted demo video. Live result: 5/6 correct (83.3%).

## Discussion

The final CNN result confirms what the literature review predicted: transfer learning outperforms hand-engineered features, but only once properly fine-tuned. The midterm's frozen-backbone CNN (86.2%) badly underperformed the classical baseline; unfreezing the backbone and training on the full dataset closed and then reversed that gap entirely (99.86%). The classical Random Forest remains a fast, interpretable, low-compute alternative and is retained as the project's baseline throughout.

## Roadblocks & Pivots

- Naive blob/void counting alone does not cleanly separate defective from normal images. CLAHE amplifies the metal surface's natural grain, producing threshold speckle that inflates "candidate void" counts on both classes. Pivoted to rely primarily on LBP/GLCM texture descriptors, with blob stats as a secondary signal.
- CLAHE's contribution to classical model accuracy was smaller and more mixed than assumed in the original pitch - tested directly via ablation rather than taken on faith (see Image Analysis Evaluation above).
- Sandbox network restrictions initially blocked pretrained CNN weight downloads during midterm development (only package registries were reachable, not download.pytorch.org). Resolved by running training locally with full internet access.
- Train/test data leakage was discovered and fixed before trusting the final CNN result (see Data Integrity Check above) - 64 duplicate images removed from training, result reconfirmed on clean data.

## Submission Deliverables

- GitHub repository (this repo)
- Unlisted/public YouTube video: main presentation
- Unlisted/public YouTube video: virtual demonstration
- Slide deck used in the recording