# AI Usage Log

This log documents AI-assisted development for the CS898BA Casting Defect Detection project, per the course's AI usage transparency requirement.

## Pitch Stage

Initial project setup (topic selection, dataset identification, pitch slide deck creation and QA, presentation talking points) was completed with Claude's assistance prior to the pitch submission. See commit history for `hello_world.py`, initial `README.md`, and the pitch slide deck for that stage's output.

## Midterm Development Session — July 12, 2026

### Entry 1: Preprocessing Pipeline Design
**Tool:** Claude (claude.ai)
**Prompt:** Asked Claude to build the 4-step image processing pipeline described in the pitch deck (CLAHE contrast enhancement, thresholding + morphological cleanup, Canny edge detection, connected-component/blob analysis) and validate it on real dataset images.
**Response synopsis:** Claude wrote `preprocessing.py` implementing all 4 steps, then generated before/after visualizations on sample defective and normal images.
**Resulting change:** Added `src/preprocessing.py` and `src/visualize_pipeline.py`. Initial blob-detection logic falsely flagged background artifacts outside the part as "voids" — Claude identified this via a sanity check across 25 samples per class and fixed it by restricting void search to the part's own silhouette.

### Entry 2: Feature Extraction
**Tool:** Claude (claude.ai)
**Prompt:** Asked Claude to build a feature extraction module combining texture descriptors (as referenced in the pitch: LBP, GLCM) with the blob/edge features from the pipeline, to feed a classical ML classifier.
**Response synopsis:** Claude wrote `src/features.py` (29-dimension feature vector: LBP histogram, GLCM properties, edge density, blob stats) and `src/extract_dataset_features.py` to run it across the full 7,348-image train/test split.
**Resulting change:** Added `src/features.py`, `src/extract_dataset_features.py`, and generated `outputs/features/` (X/y arrays for train and test).

### Entry 3: Classical Baseline Training (Approach A)
**Tool:** Claude (claude.ai)
**Prompt:** Asked Claude to train and evaluate SVM and Random Forest classifiers on the extracted features, using the dataset's real train/test split.
**Response synopsis:** Claude wrote `src/train_baseline.py`, trained both models, and reported real metrics: Random Forest reached 92.9% accuracy / 94.4% F1; SVM (RBF) reached 91.1% accuracy / 93.1% F1, both on the full 715-image held-out test set.
**Resulting change:** Added `src/train_baseline.py` and `outputs/baseline_results/` (confusion matrices, classification reports, metrics.json). Locally re-ran on my own machine and confirmed matching results (92.87% RF accuracy, 91.05% SVM accuracy).

### Entry 4: CNN Transfer Learning (Approach B)
**Tool:** Claude (claude.ai)
**Prompt:** Asked Claude to implement the MobileNetV2 transfer-learning approach described in the pitch.
**Response synopsis:** Claude wrote `src/train_cnn.py`. In Claude's own sandbox, pretrained ImageNet weights could not be downloaded (network restricted to package registries only), so an initial run used random initialization as a documented limitation.
**Resulting change:** Added `src/train_cnn.py` and `outputs/cnn_results/`. Re-ran the identical script on my local machine with normal internet access — successfully downloaded pretrained weights (confirmed "Pretrained ImageNet weights loaded: True") and produced the true transfer-learning result: 86.2% accuracy / 89.8% F1 on the full test set. This came in below the classical Random Forest baseline in this first pass, since only the classifier head was trained on a 1,200-image subset for 4 epochs. Documented as a roadblock for the final presentation, where deeper fine-tuning and the full training set will be tested.

### Entry 5: Repository Organization
**Tool:** Claude (claude.ai)
**Prompt:** Asked Claude how to structure and commit all the above work into the GitHub repo following the "incremental commits with meaningful messages" requirement.
**Response synopsis:** Claude recommended splitting the work into 5 logical commits (preprocessing, feature extraction, classical baseline, CNN, sanity-check script) and provided exact staging/commit-message guidance for GitHub Desktop.
**Resulting change:** 5 new commits pushed to `main`, bringing the repo from pitch-stage (hello_world.py only) to a working midterm codebase with real, reproducible results.

### Entry 6: README and AI Log Documentation
**Tool:** Claude (claude.ai)
**Prompt:** Asked Claude to draft an updated README.md (setup instructions, results table, roadblocks) and this AI_Log.md, reflecting the actual work and results from this session.
**Response synopsis:** Claude generated both files as direct downloads to avoid encoding/formatting issues from copy-pasting into Word or the chat window.
**Resulting change:** Updated `README.md` and `AI_Log.md` with real content reflecting this session's work.
