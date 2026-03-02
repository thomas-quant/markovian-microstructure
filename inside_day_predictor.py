"""
Inside Day Predictor — Logistic Regression Pre-Filter
======================================================
Predicts whether the next day will be an inside bar.
Used as a pre-filter in combined_test.py to reduce the dominant PCX+ICT failure mode
(~70% of misses are inside bars / consolidation days).

Features (all computed from daily resampled OHLCV data):
  1. prior_range_atr_ratio    — prior day range / 14-day ATR
  2. prior_body_wick_ratio    — prior day body size / total range
  3. volume_ratio             — current day volume / prior day volume
  4. close_proximity_ratio    — |prior close - prior midpoint| / prior range
  5. consecutive_expansion_count — days since last inside bar
  6. day_of_week              — 0=Monday ... 4=Friday
  7. atr_regime               — rolling 14-day ATR / 60-day average ATR

Usage
-----
    from inside_day_predictor import InsideDayPredictor

    predictor = InsideDayPredictor()
    metrics   = predictor.fit(daily_df)          # returns train/val/test metrics
    proba     = predictor.predict_proba(daily_df) # P(next bar is inside)
    flags     = predictor.predict(daily_df)       # binary: 1 = likely inside
"""

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    classification_report,
)


FEATURE_COLUMNS = [
    "prior_range_atr_ratio",
    "prior_body_wick_ratio",
    "volume_ratio",
    "close_proximity_ratio",
    "consecutive_expansion_count",
    "day_of_week",
    "atr_regime",
]


class InsideDayPredictor:
    """
    Logistic regression classifier that predicts whether the NEXT day will be
    an inside bar (high <= prior high AND low >= prior low).

    Train/val/test split is time-ordered (60/20/20) to avoid look-ahead bias.
    Classification threshold is tuned on the validation set to maximise precision
    (we prefer high precision to reduce false-positive filter suppression).
    """

    def __init__(self, threshold: float = 0.5):
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ])
        self.threshold = threshold
        self._is_fitted = False

    # ── Feature Engineering ────────────────────────────────────────────────

    def engineer_features(self, daily: pd.DataFrame) -> pd.DataFrame:
        """
        Add all 7 feature columns to a copy of the daily DataFrame.
        Handles both capitalised (combined_test.py style) and lowercase columns.
        """
        # Normalise column names
        col = {c.lower(): c for c in daily.columns}
        O = col.get("open", "Open")
        H = col.get("high", "High")
        L = col.get("low", "Low")
        C = col.get("close", "Close")
        V = col.get("volume", "Volume")
        has_volume = V in daily.columns

        df = daily.copy()

        # ── Ranges & ATR ──────────────────────────────────────────────────
        df["_range"]      = df[H] - df[L]
        df["_atr_14"]     = df["_range"].rolling(14).mean()
        df["_prior_range"] = df["_range"].shift(1)
        df["_prior_range_safe"] = df["_prior_range"].replace(0, np.nan)

        # 1. prior_range_atr_ratio
        df["prior_range_atr_ratio"] = df["_prior_range"] / df["_atr_14"].shift(1)

        # 2. prior_body_wick_ratio
        df["_body"]       = (df[C] - df[O]).abs()
        df["_prior_body"] = df["_body"].shift(1)
        df["prior_body_wick_ratio"] = df["_prior_body"] / df["_prior_range_safe"]

        # 3. volume_ratio
        if has_volume:
            df["_prior_vol"] = df[V].shift(1)
            df["volume_ratio"] = df[V] / df["_prior_vol"].replace(0, np.nan)
        else:
            df["volume_ratio"] = np.nan

        # 4. close_proximity_ratio (|prior close - prior midpoint| / prior range)
        df["_prior_high"] = df[H].shift(1)
        df["_prior_low"]  = df[L].shift(1)
        df["_prior_mid"]  = (df["_prior_high"] + df["_prior_low"]) / 2
        df["_prior_close"] = df[C].shift(1)
        df["close_proximity_ratio"] = (
            (df["_prior_close"] - df["_prior_mid"]).abs() / df["_prior_range_safe"]
        )

        # 5. consecutive_expansion_count (days since last inside bar)
        is_inside = (df[H] <= df["_prior_high"]) & (df[L] >= df["_prior_low"])
        streak = np.zeros(len(df), dtype=float)
        count = 0.0
        for i in range(len(df)):
            if is_inside.iloc[i]:
                count = 0.0
            else:
                count += 1.0
            streak[i] = count
        df["consecutive_expansion_count"] = streak

        # 6. day_of_week
        df["day_of_week"] = df.index.dayofweek

        # 7. atr_regime (14-day ATR / 60-day rolling mean of 14-day ATR)
        df["_atr_60_avg"] = df["_atr_14"].rolling(60).mean()
        df["atr_regime"]  = df["_atr_14"] / df["_atr_60_avg"].replace(0, np.nan)

        # Drop internal work columns
        df.drop(
            columns=[c for c in df.columns if c.startswith("_")],
            inplace=True,
        )
        return df

    # ── Labelling ──────────────────────────────────────────────────────────

    @staticmethod
    def label_next_inside(daily: pd.DataFrame) -> pd.Series:
        """
        Create a binary label: 1 if the NEXT bar is an inside bar.
        next_high <= today_high AND next_low >= today_low.
        """
        col = {c.lower(): c for c in daily.columns}
        H = col.get("high", "High")
        L = col.get("low", "Low")
        next_high = daily[H].shift(-1)
        next_low  = daily[L].shift(-1)
        return ((next_high <= daily[H]) & (next_low >= daily[L])).astype(int)

    # ── Fit / Evaluate ─────────────────────────────────────────────────────

    def fit(self, daily: pd.DataFrame) -> dict:
        """
        Fit the logistic regression on the daily DataFrame using a
        time-ordered 60/20/20 train/val/test split.

        Returns a dict of metrics including validation AUC and test metrics.
        """
        df     = self.engineer_features(daily)
        labels = self.label_next_inside(daily)
        df["_target"] = labels

        df_clean = df[FEATURE_COLUMNS + ["_target"]].dropna()
        n        = len(df_clean)

        if n < 100:
            raise ValueError(f"Insufficient data for fitting: {n} clean rows")

        train_end = int(n * 0.60)
        val_end   = int(n * 0.80)

        X      = df_clean[FEATURE_COLUMNS].values
        y      = df_clean["_target"].values

        X_train, y_train = X[:train_end],       y[:train_end]
        X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
        X_test,  y_test  = X[val_end:],          y[val_end:]

        self.pipeline.fit(X_train, y_train)
        self._is_fitted = True

        # ── Tune threshold on val set (maximise precision) ────────────────
        val_probs     = self.pipeline.predict_proba(X_val)[:, 1]
        best_thresh   = 0.5
        best_precision = 0.0

        for t in np.arange(0.30, 0.91, 0.05):
            preds = (val_probs >= t).astype(int)
            if preds.sum() >= 5:          # require at least 5 positive predictions
                p = precision_score(y_val, preds, zero_division=0.0)
                if p > best_precision:
                    best_precision = p
                    best_thresh    = t

        self.threshold = round(best_thresh, 2)

        # ── Test set evaluation ────────────────────────────────────────────
        test_probs = self.pipeline.predict_proba(X_test)[:, 1]
        test_preds = (test_probs >= self.threshold).astype(int)

        test_auc = (
            roc_auc_score(y_test, test_probs)
            if len(np.unique(y_test)) > 1
            else 0.5
        )
        test_prec = precision_score(y_test, test_preds, zero_division=0.0)
        test_rec  = recall_score(y_test, test_preds, zero_division=0.0)

        # ── Validation AUC (for acceptance criterion check) ───────────────
        val_auc = (
            roc_auc_score(y_val, val_probs)
            if len(np.unique(y_val)) > 1
            else 0.5
        )

        # ── Feature coefficients ──────────────────────────────────────────
        clf    = self.pipeline.named_steps["clf"]
        coefs  = dict(zip(FEATURE_COLUMNS, clf.coef_[0]))

        metrics = {
            "n_train":           train_end,
            "n_val":             val_end - train_end,
            "n_test":            n - val_end,
            "inside_rate_train": float(y_train.mean()),
            "inside_rate_all":   float(y.mean()),
            "threshold":         self.threshold,
            "val_auc":           round(val_auc, 4),
            "val_precision":     round(best_precision, 4),
            "test_auc":          round(test_auc, 4),
            "test_precision":    round(test_prec, 4),
            "test_recall":       round(test_rec, 4),
            "n_predicted_inside_test": int(test_preds.sum()),
            "feature_coefs":     coefs,
        }
        return metrics

    # ── Inference ──────────────────────────────────────────────────────────

    def predict_proba(self, daily: pd.DataFrame) -> np.ndarray:
        """
        Return P(next bar is inside day) for every row in daily.
        Rows without sufficient feature data return NaN.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        df         = self.engineer_features(daily)
        valid_mask = df[FEATURE_COLUMNS].notna().all(axis=1)
        proba      = np.full(len(daily), np.nan)

        if valid_mask.sum() > 0:
            X = df.loc[valid_mask, FEATURE_COLUMNS].values
            proba[valid_mask.values] = self.pipeline.predict_proba(X)[:, 1]

        return proba

    def predict(self, daily: pd.DataFrame) -> np.ndarray:
        """
        Return binary prediction for each row: 1 = likely next inside bar, 0 = not.
        NaN rows (insufficient features) return 0 (not filtered out).
        """
        proba = self.predict_proba(daily)
        out   = np.where(np.isnan(proba), 0, (proba >= self.threshold).astype(float))
        return out

    def evaluate(self, daily: pd.DataFrame) -> dict:
        """Alias for fit() — fits the model and returns full evaluation metrics."""
        return self.fit(daily)

    # ── Pretty Print ────────────────────────────────────────────────────────

    def print_report(self, metrics: dict) -> None:
        print("\n── Inside Day Predictor (Track 3) ────────────────────────")
        print(f"  Train / Val / Test:  {metrics['n_train']} / {metrics['n_val']} / {metrics['n_test']}")
        print(f"  Base inside rate:    {metrics['inside_rate_all']:.1%}")
        print(f"  Tuned threshold:     {metrics['threshold']:.2f}")
        print(f"  Val  AUC:            {metrics['val_auc']:.4f}  (acceptance: >0.60)")
        print(f"  Test AUC:            {metrics['test_auc']:.4f}")
        print(f"  Test precision:      {metrics['test_precision']:.4f}")
        print(f"  Test recall:         {metrics['test_recall']:.4f}")
        print(f"  Test: predicted inside days = {metrics['n_predicted_inside_test']}")

        print("\n  Feature coefficients (logistic regression):")
        for feat, coef in sorted(
            metrics["feature_coefs"].items(), key=lambda x: -abs(x[1])
        ):
            direction = "+" if coef > 0 else "-"
            print(f"    {feat:<35s}  {direction}{abs(coef):.4f}")

        if metrics["val_auc"] > 0.60:
            print("\n  ✓ Val AUC > 0.60 — meaningful predictive power confirmed")
        else:
            print("\n  ✗ Val AUC <= 0.60 — insufficient predictive power")


# ── Standalone test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    from ict_daily_bias import build_daily_bars

    daily   = build_daily_bars("data/nq_1m.parquet")
    pred    = InsideDayPredictor()
    metrics = pred.fit(daily)
    pred.print_report(metrics)
