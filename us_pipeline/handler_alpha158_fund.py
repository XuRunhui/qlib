"""Custom Qlib data handler that combines Alpha158 features with our
fundamental factors loaded from parquet.

Implementation: load Alpha158 features+label normally via Qlib, then merge our
parquet's fund_* columns into the 'feature' group. This keeps the MultiIndex
(feature/label) intact, which downstream Qlib processors require.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from qlib.contrib.data.handler import Alpha158, _DEFAULT_LEARN_PROCESSORS, check_transform_proc
from qlib.contrib.data.loader import Alpha158DL
from qlib.data.dataset.handler import DataHandlerLP

ROOT = Path(__file__).parent
DEFAULT_FUND_PARQUET = ROOT / "data" / "fundamental_factors.parquet"


class Alpha158WithFundamentals(Alpha158):
    """Alpha158 + fundamental factors loaded from a parquet file.

    The fundamental columns are merged into the 'feature' MultiIndex group
    after the parent Alpha158 finishes its data setup.
    """

    def __init__(
        self,
        *args,
        fundamentals_parquet: str | None = None,
        **kwargs,
    ):
        if fundamentals_parquet is None:
            fundamentals_parquet = str(DEFAULT_FUND_PARQUET)
        self._fundamentals_parquet = fundamentals_parquet
        super().__init__(*args, **kwargs)

    def setup_data(self, *args, **kwargs):
        """Run Alpha158 data setup, then merge fundamental factors into _data."""
        super().setup_data(*args, **kwargs)
        fund = self._load_fundamentals()
        if fund is None or fund.empty:
            return

        # self._data is a DataFrame with MultiIndex columns level 0 in {feature, label}
        # and indexed by (datetime, instrument). We add fund_* columns into the
        # 'feature' top-level group via reindex/join.
        df = self._data
        # Align on (datetime, instrument)
        merged_features = fund.reindex(df.index)
        # Add as feature.* columns
        new_cols = pd.MultiIndex.from_product([["feature"], merged_features.columns])
        merged_features.columns = new_cols
        # Concat columns; keep label group untouched
        feature_df = df["feature"]
        label_df = df["label"]
        feature_full = pd.concat([feature_df, merged_features.droplevel(0, axis=1).reindex(feature_df.index)], axis=1)
        # Re-attach as MultiIndex
        feature_full.columns = pd.MultiIndex.from_product([["feature"], feature_full.columns])
        full = pd.concat([feature_full, label_df.add_prefix("").pipe(
            lambda d: d.set_axis(pd.MultiIndex.from_product([["label"], d.columns]), axis=1)
        )], axis=1)
        self._data = full
        # Re-run learn processors on the augmented data so DK_L is updated.
        # Easiest path: also update _infer / _learn frames the same way.
        if hasattr(self, "_infer") and self._infer is not None:
            self._infer = self._reapply_to(self._infer, fund, "infer")
        if hasattr(self, "_learn") and self._learn is not None:
            self._learn = self._reapply_to(self._learn, fund, "learn")

    def _reapply_to(self, df: pd.DataFrame, fund: pd.DataFrame, kind: str) -> pd.DataFrame:
        """Append fundamental columns to a processed (infer or learn) frame."""
        merged_features = fund.reindex(df.index)
        feature_df = df["feature"]
        label_df = df["label"]
        feature_full = pd.concat([feature_df, merged_features], axis=1)
        feature_full.columns = pd.MultiIndex.from_product([["feature"], feature_full.columns])
        full = pd.concat([feature_full, label_df.pipe(
            lambda d: d.set_axis(pd.MultiIndex.from_product([["label"], d.columns]), axis=1)
        )], axis=1)
        return full

    def _load_fundamentals(self) -> pd.DataFrame:
        df = pd.read_parquet(self._fundamentals_parquet)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["instrument"] = df["instrument"].str.upper()
        df = df.set_index(["datetime", "instrument"]).sort_index()
        factor_cols = [c for c in df.columns if c.startswith("fund_")]
        return df[factor_cols]
