"""
Autism AQ-10 Screening — Model Training Pipeline (v2)
=====================================================

This trains a probabilistic *screening assistant*, not an autism "detector".
Its job is to flag risk, expose *why* (SHAP), and route users to a
professional evaluation. See the data-leakage caveat in the README.

Pipeline
--------
1. Load merged dataset (Adult + Adolescent + Child) with `age_group`.
2. Clean: fix age outliers, impute missing age by cohort median.
3. Feature-engineer: encode binaries, one-hot `age_group`.
4. Split -> train / calibration / test (stratified, leakage-safe).
5. Scale (StandardScaler) -> SMOTE (train only) -> fit RF & XGBoost.
6. Select best by cross-val ROC-AUC, then CALIBRATE probabilities
   (CalibratedClassifierCV, prefit on a held-out calibration set).
7. Report HONEST metrics: AUC, sensitivity/recall, specificity,
   precision, F1, Brier score, confusion matrix (held-out test set).
8. SHAP explainability: global importance + saved summary plot.
9. Version + persist: scaler.pkl, trained_model.pkl (calibrated),
   base_model.pkl (tree, for SHAP), and model_metadata.json
   (version, training date, dataset hash, full metrics).

Artifacts are written to backend/models/ ; reports to backend/models/reports/.
"""
from __future__ import annotations

import json
import hashlib
import pickle
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score, brier_score_loss,
    confusion_matrix, classification_report,
)
from imblearn.over_sampling import SMOTE

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    print("Warning: XGBoost not installed. Only RandomForest will be trained.")
    XGBOOST_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("Warning: SHAP not installed. Skipping explainability artifacts.")
    SHAP_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_MERGED = BASE_DIR / "data" / "autism_merged.csv"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = MODELS_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "trained_model.pkl"        # calibrated -> used by API
BASE_MODEL_PATH = MODELS_DIR / "base_model.pkl"      # tree -> used by SHAP
SCALER_PATH = MODELS_DIR / "scaler.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

RANDOM_STATE = 42

# Feature contract — ORDER MATTERS (scaler + model rely on it). The API reads
# this same list from model_metadata.json so inference stays in lock-step.
AQ_COLS = [f"A{i}_Score" for i in range(1, 11)]
AGE_GROUPS = ["adolescent", "adult", "child"]        # sorted, deterministic
FEATURE_COLS = (
    AQ_COLS
    + ["age", "gender", "jaundice", "austim"]
    + [f"age_group_{g}" for g in AGE_GROUPS]
)

# Age boundaries used to DERIVE age_group from raw age at inference time.
# (child 4-11, adolescent 12-16, adult 17+ in the UCI cohorts.)
AGE_BINS = {"child_max": 11, "adolescent_max": 16}


# ─── Data loading & cleaning ──────────────────────────────────────────────────
def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def derive_age_group(age: float) -> str:
    if age <= AGE_BINS["child_max"]:
        return "child"
    if age <= AGE_BINS["adolescent_max"]:
        return "adolescent"
    return "adult"


def load_data() -> tuple[pd.DataFrame, str, str]:
    if not DATA_MERGED.exists():
        raise SystemExit(
            "Merged dataset not found: data/autism_merged.csv\n"
            "Run first:  python data/download_datasets.py && python data/prepare_dataset.py"
        )
    source = "merged (adult+adolescent+child)"
    df = pd.read_csv(DATA_MERGED)
    print(f"Loaded {source}: {df.shape[0]} rows, {df.shape[1]} columns")
    return df, source, sha256_of_file(DATA_MERGED)


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Target -> 1/0
    df.rename(columns={"Class/ASD": "target"}, inplace=True)
    df["target"] = (
        df["target"].astype(str).str.strip().str.upper()
        .map({"YES": 1, "NO": 0, "1": 1, "0": 0})
    )

    # Normalise possible jaundice spelling from legacy file
    if "jundice" in df.columns and "jaundice" not in df.columns:
        df.rename(columns={"jundice": "jaundice"}, inplace=True)

    # Encode binary categoricals (robust to yes/no or m/f or pre-encoded ints)
    def map_binary(series, mapping):
        s = series.astype(str).str.strip().str.lower()
        return s.map(mapping)

    df["gender"] = map_binary(df["gender"], {"m": 1, "f": 0, "1": 1, "0": 0})
    df["jaundice"] = map_binary(df["jaundice"], {"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0})
    df["austim"] = map_binary(df["austim"], {"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0})

    # AQ scores -> int
    for c in AQ_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Age: kill impossible outliers, then impute by cohort/group median
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df.loc[(df["age"] < 1) | (df["age"] > 100), "age"] = np.nan

    # age_group: use provided column if present, else derive from age
    if "age_group" not in df.columns:
        df["age_group"] = df["age"].apply(lambda a: derive_age_group(a) if pd.notna(a) else "adult")

    # Impute missing age with median of its age_group
    df["age"] = df.groupby("age_group")["age"].transform(lambda s: s.fillna(s.median()))
    df["age"] = df["age"].fillna(df["age"].median())

    # One-hot age_group with a fixed, deterministic column set
    for g in AGE_GROUPS:
        df[f"age_group_{g}"] = (df["age_group"] == g).astype(int)

    # Drop rows still missing a target or AQ answer
    before = len(df)
    df.dropna(subset=["target"] + AQ_COLS + ["gender", "jaundice", "austim"], inplace=True)
    if before != len(df):
        print(f"   Dropped {before - len(df)} rows with unrecoverable missing values")

    df["target"] = df["target"].astype(int)
    print(f"Clean dataset: {df.shape[0]} rows")
    print(f"   Class balance:\n{df['target'].value_counts().to_string()}\n")
    return df


# ─── Metrics ──────────────────────────────────────────────────────────────────
def full_metrics(name: str, y_true, y_pred, y_prob) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0   # recall / TPR
    specificity = tn / (tn + fp) if (tn + fp) else 0.0   # TNR
    m = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
        "pr_auc": round(average_precision_score(y_true, y_prob), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "sensitivity_recall": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "brier_score": round(brier_score_loss(y_true, y_prob), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    print(f"\n{'='*56}\n  {name} — Held-out Test Metrics\n{'='*56}")
    print(f"  Accuracy            : {m['accuracy']:.4f}")
    print(f"  ROC-AUC             : {m['roc_auc']:.4f}")
    print(f"  PR-AUC              : {m['pr_auc']:.4f}")
    print(f"  Sensitivity/Recall  : {m['sensitivity_recall']:.4f}  (catches true cases)")
    print(f"  Specificity         : {m['specificity']:.4f}")
    print(f"  Precision           : {m['precision']:.4f}")
    print(f"  F1                  : {m['f1']:.4f}")
    print(f"  Brier (calibration) : {m['brier_score']:.4f}  (lower = better)")
    print(f"  Confusion matrix    : TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"\n{classification_report(y_true, y_pred, target_names=['No ASD', 'ASD'], zero_division=0)}")
    return m


# ─── SHAP explainability ──────────────────────────────────────────────────────
def explain_with_shap(base_model, X_test_scaled, feature_names) -> dict:
    if not SHAP_AVAILABLE:
        return {}
    print("Computing SHAP explanations…")
    explainer = shap.TreeExplainer(base_model)
    shap_values = explainer.shap_values(X_test_scaled)

    # Binary classifiers may return a list [class0, class1]; take positive class
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values
    if sv.ndim == 3:                      # (n, features, classes)
        sv = sv[:, :, 1]

    mean_abs = np.abs(sv).mean(axis=0)
    importance = sorted(
        zip(feature_names, mean_abs.tolist()), key=lambda x: -x[1]
    )
    print("Top global SHAP feature importance:")
    for feat, val in importance[:8]:
        print(f"   {feat:<22} {val:.4f}")

    if MPL_AVAILABLE:
        try:
            shap.summary_plot(
                sv, X_test_scaled, feature_names=feature_names,
                plot_type="bar", show=False,
            )
            plt.tight_layout()
            out = REPORTS_DIR / "shap_summary.png"
            plt.savefig(out, dpi=130, bbox_inches="tight")
            plt.close()
            print(f"Saved SHAP summary plot -> {out}")
        except Exception as e:
            print(f"   (SHAP plot skipped: {e})")

    return {feat: round(val, 6) for feat, val in importance}


# ─── Versioning ───────────────────────────────────────────────────────────────
def next_version() -> int:
    if METADATA_PATH.exists():
        try:
            prev = json.loads(METADATA_PATH.read_text())
            return int(prev.get("version", 0)) + 1
        except Exception:
            pass
    return 1


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("   Autism AQ-10 Screening — Training Pipeline (v2)")
    print("=" * 60 + "\n")

    df, source, data_hash = load_data()
    df = clean_and_engineer(df)

    X = df[FEATURE_COLS].astype(float)
    y = df["target"].astype(int)

    # train / calibration / test  (60 / 20 / 20, stratified)
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    X_tr, X_cal, y_tr, y_cal = train_test_split(
        X_tmp, y_tmp, test_size=0.25, random_state=RANDOM_STATE, stratify=y_tmp
    )
    print(f"Split -> train {len(X_tr)} | calibration {len(X_cal)} | test {len(X_test)}")

    # Scale (fit on train only)
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_cal_sc = scaler.transform(X_cal)
    X_test_sc = scaler.transform(X_test)

    # SMOTE on training only
    sm = SMOTE(random_state=RANDOM_STATE)
    X_tr_res, y_tr_res = sm.fit_resample(X_tr_sc, y_tr)
    print(f"After SMOTE  - train: {X_tr_res.shape[0]} samples, balance {np.bincount(y_tr_res)}\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    candidates = {}

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_split=2,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf_cv = cross_val_score(rf, X_tr_res, y_tr_res, cv=cv, scoring="roc_auc")
    print(f"RandomForest CV ROC-AUC: {rf_cv.mean():.4f} +/- {rf_cv.std():.4f}")
    candidates["RandomForest"] = {"estimator": rf, "cv_auc": float(rf_cv.mean())}

    # XGBoost
    if XGBOOST_AVAILABLE:
        pos_ratio = (y_tr_res == 0).sum() / max((y_tr_res == 1).sum(), 1)
        xgb = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=pos_ratio,
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1,
        )
        xgb_cv = cross_val_score(xgb, X_tr_res, y_tr_res, cv=cv, scoring="roc_auc")
        print(f"XGBoost      CV ROC-AUC: {xgb_cv.mean():.4f} +/- {xgb_cv.std():.4f}")
        candidates["XGBoost"] = {"estimator": xgb, "cv_auc": float(xgb_cv.mean())}

    # Pick best by CV AUC
    best_name = max(candidates, key=lambda k: candidates[k]["cv_auc"])
    base_model = candidates[best_name]["estimator"]
    print(f"\nBest base model: {best_name} (CV AUC={candidates[best_name]['cv_auc']:.4f})")

    # Fit base model on resampled training data
    base_model.fit(X_tr_res, y_tr_res)

    # ── Calibrate probabilities on the held-out (real, non-SMOTE) calib set ────
    # Always use sigmoid (Platt scaling) for this dataset.
    # Isotonic calibration collapses to a near-binary step function here because
    # the base model is near-perfect (label is derived from the AQ-10 score itself),
    # leaving 97%+ of outputs at exactly 0.0 or 1.0 — meaningless as a confidence.
    # Sigmoid is constrained to a smooth monotone curve and stays well-behaved.
    cal_method = "sigmoid"
    calibrated = CalibratedClassifierCV(FrozenEstimator(base_model), method=cal_method)
    calibrated.fit(X_cal_sc, y_cal)
    print(f"Calibrated probabilities using '{cal_method}' method on {len(X_cal)} samples")

    # ── Honest evaluation on the untouched test set ────────────────────────────
    y_pred_base = base_model.predict(X_test_sc)
    y_prob_base = base_model.predict_proba(X_test_sc)[:, 1]
    y_pred = calibrated.predict(X_test_sc)
    y_prob = calibrated.predict_proba(X_test_sc)[:, 1]

    print(f"\nBrier before calibration: {brier_score_loss(y_test, y_prob_base):.4f}")
    print(f"Brier after  calibration: {brier_score_loss(y_test, y_prob):.4f}")

    metrics = full_metrics(f"{best_name} (calibrated)", y_test, y_pred, y_prob)

    # ── SHAP ──────────────────────────────────────────────────────────────────
    shap_importance = explain_with_shap(base_model, X_test_sc, FEATURE_COLS)

    # ── Persist artifacts ──────────────────────────────────────────────────────
    version = next_version()
    today = dt.date.today().strftime("%Y%m%d")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(calibrated, f)
    with open(BASE_MODEL_PATH, "wb") as f:
        pickle.dump(base_model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    metadata = {
        "version": version,
        "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
        "model_type": best_name,
        "calibration_method": cal_method,
        "data_source": source,
        "dataset_sha256": data_hash,
        "n_samples": int(len(df)),
        "n_train": int(len(X_tr)), "n_calibration": int(len(X_cal)), "n_test": int(len(X_test)),
        "class_balance": {"no_asd": int((y == 0).sum()), "asd": int((y == 1).sum())},
        "feature_cols": FEATURE_COLS,
        "age_group_bins": AGE_BINS,
        "age_groups": AGE_GROUPS,
        "cv_auc_by_model": {k: round(v["cv_auc"], 4) for k, v in candidates.items()},
        "test_metrics": metrics,
        "shap_global_importance": shap_importance,
        "artifacts": {
            "calibrated_model": MODEL_PATH.name,
            "base_model": BASE_MODEL_PATH.name,
            "scaler": SCALER_PATH.name,
        },
        "disclaimer": (
            "Screening assistant only — NOT a clinical diagnosis. Labels in the "
            "UCI dataset are derived from the AQ-10 score itself; high accuracy "
            "partly reflects re-learning the questionnaire's scoring rule."
        ),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    (REPORTS_DIR / f"metrics_v{version}_{today}.json").write_text(json.dumps(metadata, indent=2))

    print(f"\nSaved calibrated model -> {MODEL_PATH}")
    print(f"Saved base model (SHAP) -> {BASE_MODEL_PATH}")
    print(f"Saved scaler            -> {SCALER_PATH}")
    print(f"Saved metadata          -> {METADATA_PATH}")
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
