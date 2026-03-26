"""
ML Inference Service
Loads the trained model once at startup (singleton pattern)
and provides async-compatible predict function.
"""
import asyncio
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from config import settings


class ModelService:
    def __init__(self):
        self.model = None
        self.scaler = None

    def load(self):
        """Load model artefacts from disk — called once at startup."""
        with open(settings.MODEL_PATH, "rb") as f:
            self.model = pickle.load(f)
        with open(settings.SCALER_PATH, "rb") as f:
            self.scaler = pickle.load(f)

    def _get_risk_level(self, confidence: float) -> str:
        if confidence >= 0.70:
            return "High Risk"
        elif confidence >= 0.40:
            return "Moderate Risk"
        return "Low Risk"

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

        def _run():
            gender_enc = 1 if gender == "m" else 0
            features = pd.DataFrame([{
                "A1_Score": a1, "A2_Score": a2, "A3_Score": a3, "A4_Score": a4, "A5_Score": a5,
                "A6_Score": a6, "A7_Score": a7, "A8_Score": a8, "A9_Score": a9, "A10_Score": a10,
                "age": age,
                "gender": gender_enc,
                "jaundice": int(jaundice),
                "austim": int(family_autism),
            }])

            scaled = self.scaler.transform(features)
            prediction = int(self.model.predict(scaled)[0])
            confidence = float(self.model.predict_proba(scaled)[0][1])

            return {
                "prediction": prediction,
                "confidence": round(confidence, 4),
                "risk_level": self._get_risk_level(confidence),
                "aq_score": a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10,
            }

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _run)


# Singleton — imported by routes
model_service = ModelService()
