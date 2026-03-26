"""
Autism Detection Model Training
=================================
Features: A1-A10 (AQ questionnaire), age, gender, jaundice, family_autism
Target:   Class/ASD (0 = No ASD, 1 = ASD)
Models:   XGBoost vs RandomForest (5-fold StratifiedKFold CV)
Output:   trained_model.pkl, scaler.pkl
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    print("Warning: XGBoost not installed. Only RandomForest will be trained.")
    XGBOOST_AVAILABLE = False

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "autism.csv"
MODEL_PATH = BASE_DIR / "models" / "trained_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

# ─── Load & Clean Data ────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # Drop clearly irrelevant / leaky columns
    drop_cols = [
        "ID",
        "result",          # data leakage – this is a noisy AQ sum
        "age_desc",        # constant "18 and more"
        "contry_of_res",   # too sparse (60+ countries)
        "ethnicity",       # noisy / sparse
        "relation",        # who filled form – not clinically useful
        "used_app_before", # irrelevant for new users
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Target
    df.rename(columns={"Class/ASD": "target"}, inplace=True)

    # Encode binary categorical columns
    df["gender"]         = df["gender"].map({"m": 1, "f": 0})
    df["jaundice"]       = df["jaundice"].map({"yes": 1, "no": 0})
    df["austim"]         = df["austim"].map({"yes": 1, "no": 0})

    # Drop any remaining NaN rows (from unknown ethnicity / unmapped vals)
    before = len(df)
    df.dropna(inplace=True)
    after = len(df)
    if before != after:
        print(f"Warning: Dropped {before - after} rows with NaN values")

    print(f"Clean dataset: {df.shape[0]} rows")
    print(f"   Class balance:\n{df['target'].value_counts()}\n")
    return df

# ─── Feature Engineering ──────────────────────────────────────────────────────
def prepare_features(df):
    feature_cols = [
        "A1_Score", "A2_Score", "A3_Score", "A4_Score", "A5_Score",
        "A6_Score", "A7_Score", "A8_Score", "A9_Score", "A10_Score",
        "age", "gender", "jaundice", "austim",
    ]
    X = df[feature_cols].astype(float)
    y = df["target"].astype(int)
    return X, y, feature_cols

# ─── Model Evaluation Helper ─────────────────────────────────────────────────
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc    = accuracy_score(y_test, y_pred)
    auc    = roc_auc_score(y_test, y_prob)

    print(f"\n{'='*50}")
    print(f"  {name} — Test Results")
    print(f"{'='*50}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['No ASD', 'ASD'])}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n")
    return acc, auc

# ─── Main Training ────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("   Autism Detection Model Training")
    print("="*60 + "\n")

    df = load_data()
    X, y, feature_cols = prepare_features(df)

    # Train/test split (stratified to preserve class ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train set: {X_train.shape[0]} | Test set: {X_test.shape[0]}")

    # Scale
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # SMOTE on training data to handle class imbalance
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train_sc, y_train)
    print(f"After SMOTE  - Train: {X_train_res.shape[0]} samples, Class balance: {np.bincount(y_train_res)}")

    # ── 5-fold CV ─────────────────────────────────────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf_cv = cross_val_score(rf, X_train_res, y_train_res, cv=cv, scoring="roc_auc")
    print(f"\nRandomForest CV ROC-AUC: {rf_cv.mean():.4f} +/- {rf_cv.std():.4f}")
    rf.fit(X_train_res, y_train_res)
    rf_acc, rf_auc = evaluate_model("Random Forest", rf, X_test_sc, y_test)
    results["RandomForest"] = {"model": rf, "acc": rf_acc, "auc": rf_auc}

    # XGBoost
    if XGBOOST_AVAILABLE:
        pos_ratio = (y_train_res == 0).sum() / (y_train_res == 1).sum()
        xgb = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=pos_ratio,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        xgb_cv = cross_val_score(xgb, X_train_res, y_train_res, cv=cv, scoring="roc_auc")
        print(f"\nXGBoost CV ROC-AUC: {xgb_cv.mean():.4f} +/- {xgb_cv.std():.4f}")
        xgb.fit(X_train_res, y_train_res)
        xgb_acc, xgb_auc = evaluate_model("XGBoost", xgb, X_test_sc, y_test)
        results["XGBoost"] = {"model": xgb, "acc": xgb_acc, "auc": xgb_auc}

    # ── Pick Best Model ────────────────────────────────────────────────────────
    best_name = max(results, key=lambda k: results[k]["auc"])
    best_model = results[best_name]["model"]
    print(f"\nBest model: {best_name} (AUC={results[best_name]['auc']:.4f}, Acc={results[best_name]['acc']:.4f})")

    # ── Feature Importance (RF) ───────────────────────────────────────────────
    if best_name == "RandomForest":
        imp = sorted(zip(feature_cols, best_model.feature_importances_), key=lambda x: -x[1])
        print("\nFeature Importances:")
        for feat, score in imp:
            print(f"   {feat:<15} {score:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    print(f"\nSaved model  -> {MODEL_PATH}")
    print(f"Saved scaler -> {SCALER_PATH}")
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
