"""
save_model_for_demo.py

Trains the Random Forest classifier on the cached feature arrays and
saves it to disk (via joblib), so the live demo script can load it
instantly instead of retraining every time.
"""

import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

# ---- EDIT THIS PATH FOR YOUR MACHINE ----
FEAT_DIR = r"C:\Users\chere\OneDrive\Documents\GitHub\SricharanCherepally-C986X996-CS898BA-Project\outputs\features"
MODEL_OUT = r"C:\Users\chere\OneDrive\Documents\GitHub\SricharanCherepally-C986X996-CS898BA-Project\outputs\demo_model.joblib"
# ------------------------------------------

def main():
    X_train = np.load(os.path.join(FEAT_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(FEAT_DIR, "y_train.npy"))

    print(f"Training Random Forest on {X_train.shape[0]} images...")
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    joblib.dump(rf, MODEL_OUT)
    print(f"Saved trained model to {MODEL_OUT}")


if __name__ == "__main__":
    main()
