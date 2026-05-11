"""LightGBM LambdaRank model for Qlib.

LambdaRank requires:
  1. Integer relevance labels (0..N-1)
  2. Group sizes (one per query/day)

We adapt Qlib's DatasetH output (multi-indexed by datetime, instrument) by:
  - Converting the (already CSRankNorm'd or sector-neutral'd) float label to
    integer bins per day [0..n_bins-1] using qcut.
  - Computing group sizes = number of instruments per day.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from qlib.model.base import Model
from qlib.data.dataset import DatasetH


class LGBRankModel(Model):
    def __init__(self, n_bins: int = 32, early_stopping_rounds: int = 50,
                 num_boost_round: int = 1000, **kwargs):
        self.n_bins = n_bins
        self.early_stopping_rounds = early_stopping_rounds
        self.num_boost_round = num_boost_round
        # LambdaRank params
        self.params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [10, 30, 50],
            "verbosity": -1,
        }
        self.params.update(kwargs)
        self.model = None

    @staticmethod
    def _prep(df: pd.DataFrame, n_bins: int):
        """Returns (X, y_int, group_sizes). df is multi-indexed by (datetime, instrument)."""
        # Sort by datetime so groups are contiguous
        df = df.sort_index(level="datetime")
        # Split features and label
        feature_cols = [c for c in df.columns if c[0] == "feature"]
        label_cols = [c for c in df.columns if c[0] == "label"]
        X = df[feature_cols].values
        y_float = df[label_cols].values.flatten()

        # Per-day quantize label to [0, n_bins-1]
        dates = df.index.get_level_values("datetime")
        y_int = np.zeros_like(y_float, dtype=np.int32)
        group_sizes = []
        # Iterate in the same sorted order
        for date, idx in pd.Series(np.arange(len(df)), index=dates).groupby(level=0):
            sub = y_float[idx.values]
            # qcut may fail on too few unique values; fall back to rank-based binning
            try:
                bins = pd.qcut(sub, n_bins, labels=False, duplicates="drop")
                if bins.max() < 1:
                    raise ValueError("collapsed bins")
            except Exception:
                # fallback: rank then floor to n_bins
                ranks = pd.Series(sub).rank(pct=True).values
                bins = np.minimum((ranks * n_bins).astype(int), n_bins - 1)
            y_int[idx.values] = bins
            group_sizes.append(len(idx))
        return X, y_int, group_sizes

    def fit(self, dataset: DatasetH, **kwargs):
        df_train = dataset.prepare("train", col_set=["feature", "label"], data_key="learn")
        df_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key="learn")
        df_train = df_train.dropna()
        df_valid = df_valid.dropna()

        Xtr, ytr, gtr = self._prep(df_train, self.n_bins)
        Xva, yva, gva = self._prep(df_valid, self.n_bins)

        train_set = lgb.Dataset(Xtr, label=ytr, group=gtr)
        valid_set = lgb.Dataset(Xva, label=yva, group=gva, reference=train_set)

        self.model = lgb.train(
            self.params,
            train_set,
            num_boost_round=self.num_boost_round,
            valid_sets=[train_set, valid_set],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(self.early_stopping_rounds),
                lgb.log_evaluation(period=50),
            ],
        )

    def predict(self, dataset: DatasetH, segment: str = "test") -> pd.Series:
        df_test = dataset.prepare(segment, col_set=["feature", "label"], data_key="infer")
        feature_cols = [c for c in df_test.columns if c[0] == "feature"]
        X = df_test[feature_cols].values
        scores = self.model.predict(X)
        return pd.Series(scores, index=df_test.index, name="score")
