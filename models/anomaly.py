"""
models/anomaly.py
-----------------
Isolation Forest–based anomaly detector for financial transactions.
Includes training, prediction, persistence (joblib), and evaluation helpers.
"""

import os
import logging
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

logger = logging.getLogger(__name__)

# ── Feature columns used for anomaly detection ────────────────────────────────
FEATURE_COLS = [
    "amount",
    "sender_old_balance",
    "sender_new_balance",
    "receiver_old_balance",
    "receiver_new_balance",
    "balance_diff_sender",
    "balance_diff_receiver",
    "amount_to_sender_bal",
    "is_round_amount",
    "type_encoded",
]

MODEL_PATH  = "models/isolation_forest.joblib"
SCALER_PATH = "models/scaler.joblib"


class AnomalyDetector:
    """
    Wraps Isolation Forest for FinOps anomaly detection.

    Usage
    -----
    detector = AnomalyDetector()
    detector.train(ledger_fe_df)
    predictions = detector.predict(new_df)   # -1 = anomaly, 1 = normal
    detector.save()
    detector.load()
    """

    def __init__(self, contamination: float = 0.02, n_estimators: int = 100,
                 random_state: int = 42):
        self.contamination = contamination
        self.model  = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self._fitted = False

    # ── Private helpers ───────────────────────────────────────────────────────

    def _prepare(self, df: pd.DataFrame) -> np.ndarray:
        """Select and fill feature columns → numpy array."""
        available = [c for c in FEATURE_COLS if c in df.columns]
        missing   = set(FEATURE_COLS) - set(available)
        if missing:
            logger.warning(f"Missing feature columns (will fill with 0): {missing}")
        X = df[available].fillna(0).values
        return X

    # ── Public API ────────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame) -> "AnomalyDetector":
        """Fit scaler + Isolation Forest on df."""
        X = self._prepare(df)
        X_scaled = self.scaler.fit_transform(X)
        logger.info(f"Training Isolation Forest on {X_scaled.shape[0]:,} samples …")
        self.model.fit(X_scaled)
        self._fitted = True
        logger.info("Isolation Forest training complete.")
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Returns a Series of {-1 (anomaly), 1 (normal)}.
        Anomaly score < 0 → flagged.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call train() or load() first.")
        X = self._prepare(df)
        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)
        return pd.Series(preds, index=df.index, name="anomaly_flag")

    def anomaly_scores(self, df: pd.DataFrame) -> pd.Series:
        """Raw anomaly scores (lower = more anomalous)."""
        if not self._fitted:
            raise RuntimeError("Model not fitted.")
        X = self._prepare(df)
        X_scaled = self.scaler.transform(X)
        scores = self.model.decision_function(X_scaled)
        return pd.Series(scores, index=df.index, name="anomaly_score")

    def evaluate(self, df: pd.DataFrame, y_true_col: str = "is_fraud") -> dict:
        """
        Evaluate against a known label column.
        Isolation Forest labels:  -1 → predicted fraud (1), 1 → predicted normal (0)
        """
        if y_true_col not in df.columns:
            logger.warning(f"Column '{y_true_col}' not found – skipping evaluation.")
            return {}
        preds_raw = self.predict(df)
        y_pred = (preds_raw == -1).astype(int)
        y_true = df[y_true_col].astype(int)

        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        cm     = confusion_matrix(y_true, y_pred)
        logger.info(f"Isolation Forest evaluation:\n{classification_report(y_true, y_pred, zero_division=0)}")
        return {"report": report, "confusion_matrix": cm}

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, model_path: str = MODEL_PATH, scaler_path: str = SCALER_PATH):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(self.model,  model_path)
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Model saved → {model_path}")
        logger.info(f"Scaler saved → {scaler_path}")

    def load(self, model_path: str = MODEL_PATH, scaler_path: str = SCALER_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No saved model at {model_path}")
        self.model   = joblib.load(model_path)
        self.scaler  = joblib.load(scaler_path)
        self._fitted = True
        logger.info(f"Model loaded from {model_path}")
        return self


# ── Stand-alone test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from data_processing import load_data, inject_discrepancies, engineer_features

    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(level=logging.INFO)

    raw = load_data(n_rows=20_000)
    ledger, _ = inject_discrepancies(raw)
    ledger_fe = engineer_features(ledger)

    det = AnomalyDetector(contamination=0.02)
    det.train(ledger_fe)
    preds = det.predict(ledger_fe)
    n_anomalies = (preds == -1).sum()
    print(f"\n🔍 Anomalies detected: {n_anomalies:,} / {len(preds):,}")

    eval_result = det.evaluate(ledger_fe)
    if eval_result:
        r = eval_result["report"]
        print(f"   Precision (fraud): {r.get('1', {}).get('precision', 0):.3f}")
        print(f"   Recall    (fraud): {r.get('1', {}).get('recall',    0):.3f}")

    det.save()
    print("✅ Model saved.")
