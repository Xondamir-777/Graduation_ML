"""
models/rf_climate.py

Random Forest model for climate-based disease prediction.

This model is trained separately from the CNN and provides:
  1. Disease probability vector via predict_proba()
  2. Combined vector (RF probabilities + raw normalized climate features)
     for multimodal fusion in the neural network.

Key design decisions:
  - Ensures fixed output dimension [N, NUM_DISEASE_CLASSES]
    even if some classes are missing in training split (padding).
  - StandardScaler is used for climate normalization.
  - Supports direct integration with FusionHead in multimodal model.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score

from config import (
    OUTPUTS_DIR, CLIMATE_FEATURES, NUM_DISEASE_CLASSES,
    RANDOM_SEED
)

RF_PATH     = OUTPUTS_DIR / "rf_climate.pkl"
SCALER_PATH = OUTPUTS_DIR / "climate_scaler.pkl"


class RFClimateModel:
    """
    Wrapper around sklearn RandomForestClassifier.

    Outputs:
      predict_proba() → [N, NUM_DISEASE_CLASSES]
      predict_proba_and_raw() → [N, NUM_DISEASE_CLASSES + NUM_CLIMATE_FEATURES]

    The second output is used in FusionHead, combining:
      - RF semantic prediction (softmax-like probabilities)
      - raw normalized climate signals
    """

    def __init__(self, n_estimators: int = 300, max_depth: int = None):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_SEED,
            min_samples_leaf=3,
        )

        self.scaler = StandardScaler()
        self.feature_cols: list[str] = []
        self.trained = False

    # ─────────────────────────────────────────────────────────────
    # Feature handling
    # ─────────────────────────────────────────────────────────────

    def _select_cols(self, df: pd.DataFrame) -> list[str]:
        return [c for c in CLIMATE_FEATURES if c in df.columns]

    def _to_array(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        X = df[self.feature_cols].values.astype(np.float32)
        return self.scaler.fit_transform(X) if fit else self.scaler.transform(X)

    def _pad_proba(self, proba: np.ndarray) -> np.ndarray:
        """
        Ensures fixed output size [N, NUM_DISEASE_CLASSES]
        even if not all classes appear in training data.
        """
        n_classes = proba.shape[1]

        if n_classes < NUM_DISEASE_CLASSES:
            pad = np.zeros(
                (proba.shape[0], NUM_DISEASE_CLASSES - n_classes),
                dtype=np.float32
            )
            proba = np.concatenate([proba, pad], axis=1)

        return proba.astype(np.float32)

    # ─────────────────────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────────────────────

    def fit(self, df_train: pd.DataFrame, df_val: pd.DataFrame | None = None):
        self.feature_cols = self._select_cols(df_train)

        X_train = self._to_array(df_train, fit=True)
        y_train = df_train["disease_class_idx"].values

        print(f"[RF] Training RandomForest on {len(X_train)} samples "
              f"with {len(self.feature_cols)} features...")

        self.model.fit(X_train, y_train)
        self.trained = True

        train_f1 = f1_score(y_train, self.model.predict(X_train), average="macro")
        print(f"  Train macro-F1: {train_f1:.4f}")

        if df_val is not None:
            X_val = self._to_array(df_val)
            y_val = df_val["disease_class_idx"].values

            val_pred = self.model.predict(X_val)
            val_f1 = f1_score(y_val, val_pred, average="macro")

            print(f"  Val macro-F1:   {val_f1:.4f}")
            print(classification_report(y_val, val_pred, zero_division=0))

        self.save()
        return self

    # ─────────────────────────────────────────────────────────────
    # Inference
    # ─────────────────────────────────────────────────────────────

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """→ [N, NUM_DISEASE_CLASSES]"""
        X = self._to_array(df)
        return self._pad_proba(self.model.predict_proba(X))

    def predict_proba_and_raw(self, df: pd.DataFrame) -> np.ndarray:
        """
        → [N, NUM_DISEASE_CLASSES + NUM_CLIMATE_FEATURES]

        Used for multimodal fusion:
        [RF probabilities || normalized climate features]
        """
        X_scaled = self._to_array(df)
        proba = self._pad_proba(self.model.predict_proba(X_scaled))

        return np.concatenate([proba, X_scaled], axis=1)

    def predict_proba_from_array(self, X_scaled: np.ndarray) -> np.ndarray:
        """Input already scaled → returns [N, NUM_DISEASE_CLASSES]"""
        return self._pad_proba(self.model.predict_proba(X_scaled))

    def predict_proba_and_raw_from_array(self, X_scaled: np.ndarray) -> np.ndarray:
        """[N, features] → [N, NUM_DISEASE_CLASSES + features]"""
        proba = self._pad_proba(self.model.predict_proba(X_scaled))
        return np.concatenate([proba, X_scaled], axis=1)

    # ─────────────────────────────────────────────────────────────
    # Feature importance
    # ─────────────────────────────────────────────────────────────

    def feature_importance(self) -> pd.DataFrame:
        if not self.trained:
            raise RuntimeError("RF is not trained. Call fit() first.")

        return (
            pd.DataFrame({
                "feature": self.feature_cols,
                "importance": self.model.feature_importances_,
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    # ─────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────

    def save(self, path: Path = RF_PATH):
        joblib.dump({
            "model": self.model,
            "scaler": self.scaler,
            "feature_cols": self.feature_cols,
        }, path)

        print(f"[RF] Saved to: {path}")

    @classmethod
    def load(cls, path: Path = RF_PATH) -> "RFClimateModel":
        data = joblib.load(path)

        obj = cls()
        obj.model = data["model"]
        obj.scaler = data["scaler"]
        obj.feature_cols = data["feature_cols"]
        obj.trained = True

        return obj