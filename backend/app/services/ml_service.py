"""
ML Inference Service
====================
Loads the trained *calibrated* model once at startup (singleton pattern) and
provides an async-compatible `predict` function.

Changes vs v1
-------------
* Feature contract now read from model_metadata.json (single source of truth),
  including the derived `age_group` one-hot columns.
* Returns calibrated probabilities (meaningful confidence).
* Adds per-prediction SHAP explanation: which answers pushed the score up/down.
* Backwards compatible: if the new artifacts/metadata are missing, falls back
  to the legacy 14-feature contract so the app still boots.
"""
import asyncio
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from config import settings

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

MODELS_DIR = Path(settings.MODEL_PATH).resolve().parent
METADATA_PATH = MODELS_DIR / "model_metadata.json"
BASE_MODEL_PATH = MODELS_DIR / "base_model.pkl"

# Legacy fallback contract (v1 model)
LEGACY_FEATURES = [
    "A1_Score", "A2_Score", "A3_Score", "A4_Score", "A5_Score",
    "A6_Score", "A7_Score", "A8_Score", "A9_Score", "A10_Score",
    "age", "gender", "jaundice", "austim",
]
AGE_BINS = {"child_max": 11, "adolescent_max": 16}
AGE_GROUPS = ["adolescent", "adult", "child"]

# Human-readable AQ-10 labels for explanations (matches the frontend wording)
AQ_LABELS = {
    "A1_Score": "Notices small sounds others don't",
    "A2_Score": "Focuses on whole picture vs details",
    "A3_Score": "Ease of doing several things at once",
    "A4_Score": "Switching back after interruption",
    "A5_Score": "Reading between the lines",
    "A6_Score": "Telling if a listener is bored",
    "A7_Score": "Working out characters' intentions",
    "A8_Score": "Likes collecting categories of info",
    "A9_Score": "Reading feelings from faces",
    "A10_Score": "Working out people's intentions",
    "age": "Age",
    "gender": "Gender",
    "jaundice": "Jaundice at birth",
    "austim": "Family history of autism",
}


def _derive_age_group(age: float) -> str:
    if age <= AGE_BINS["child_max"]:
        return "child"
    if age <= AGE_BINS["adolescent_max"]:
        return "adolescent"
    return "adult"


class ModelService:
    def __init__(self):
        self.model = None          # calibrated model used for probabilities
        self.base_model = None     # tree model used for SHAP
        self.scaler = None
        self.feature_cols = LEGACY_FEATURES
        self.metadata = {}
        self._explainer = None

    def load(self):
        """Load model artefacts from disk — called once at startup."""
        with open(settings.MODEL_PATH, "rb") as f:
            self.model = pickle.load(f)
        with open(settings.SCALER_PATH, "rb") as f:
            self.scaler = pickle.load(f)

        # Feature contract from metadata (falls back to legacy)
        if METADATA_PATH.exists():
            try:
                self.metadata = json.loads(METADATA_PATH.read_text())
                self.feature_cols = self.metadata.get("feature_cols", LEGACY_FEATURES)
            except Exception:
                self.feature_cols = LEGACY_FEATURES

        # Optional base tree model + SHAP explainer
        if SHAP_AVAILABLE and BASE_MODEL_PATH.exists():
            try:
                with open(BASE_MODEL_PATH, "rb") as f:
                    self.base_model = pickle.load(f)
                self._explainer = shap.TreeExplainer(self.base_model)
            except Exception:
                self._explainer = None

    @property
    def version(self) -> int:
        return int(self.metadata.get("version", 0))

    def _get_risk_level(self, confidence: float) -> str:
        if confidence >= 0.70:
            return "High Risk"
        elif confidence >= 0.40:
            return "Moderate Risk"
        return "Low Risk"

    def _build_feature_frame(self, raw: dict) -> pd.DataFrame:
        """Assemble a single-row DataFrame matching self.feature_cols exactly."""
        age = float(raw["age"])
        group = _derive_age_group(age)
        full = {
            "A1_Score": raw["a1"], "A2_Score": raw["a2"], "A3_Score": raw["a3"],
            "A4_Score": raw["a4"], "A5_Score": raw["a5"], "A6_Score": raw["a6"],
            "A7_Score": raw["a7"], "A8_Score": raw["a8"], "A9_Score": raw["a9"],
            "A10_Score": raw["a10"],
            "age": age,
            "gender": 1 if raw["gender"] == "m" else 0,
            "jaundice": int(raw["jaundice"]),
            "austim": int(raw["family_autism"]),
        }
        # Derived one-hot age_group columns (only added if model expects them)
        for g in AGE_GROUPS:
            full[f"age_group_{g}"] = int(group == g)

        # Select exactly the columns the model was trained on, in order
        row = {c: full.get(c, 0) for c in self.feature_cols}
        return pd.DataFrame([row])[self.feature_cols]

    def _explain(self, X_scaled: np.ndarray) -> list[dict]:
        """Top contributing features for THIS prediction (signed SHAP values)."""
        if self._explainer is None:
            return []
        try:
            sv = self._explainer.shap_values(X_scaled)
            if isinstance(sv, list):
                sv = sv[1]
            sv = np.asarray(sv)
            if sv.ndim == 3:
                sv = sv[:, :, 1]
            contribs = sv[0]
            ranked = sorted(
                zip(self.feature_cols, contribs.tolist()),
                key=lambda x: -abs(x[1]),
            )[:5]
            return [
                {
                    "feature": feat,
                    "label": AQ_LABELS.get(feat, feat),
                    "impact": round(float(val), 4),
                    "direction": "increases" if val > 0 else "decreases",
                }
                for feat, val in ranked
            ]
        except Exception:
            return []

    async def predict(
        self,
        a1: int, a2: int, a3: int, a4: int, a5: int,
        a6: int, a7: int, a8: int, a9: int, a10: int,
        age: float,
        gender: str,       # 'm' or 'f'
        jaundice: bool,
        family_autism: bool,
    ) -> dict:
        """Run inference in a thread pool to avoid blocking the event loop."""
        raw = {
            "a1": a1, "a2": a2, "a3": a3, "a4": a4, "a5": a5,
            "a6": a6, "a7": a7, "a8": a8, "a9": a9, "a10": a10,
            "age": age, "gender": gender,
            "jaundice": jaundice, "family_autism": family_autism,
        }

        def _run():
            features = self._build_feature_frame(raw)
            scaled = self.scaler.transform(features)

            prediction = int(self.model.predict(scaled)[0])
            raw_conf = float(self.model.predict_proba(scaled)[0][1])
            # Clip to [0.03, 0.97] — exact 0% or 100% is not meaningful for a
            # screening tool and misleads users into false certainty.
            confidence = round(float(np.clip(raw_conf, 0.03, 0.97)), 4)
            explanation = self._explain(scaled)

            return {
                "prediction": prediction,
                "confidence": round(confidence, 4),
                "risk_level": self._get_risk_level(confidence),
                "aq_score": a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10,
                "explanation": explanation,
                "model_version": self.version,
            }

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _run)


# Singleton — imported by routes
model_service = ModelService()
