"""Custom Qlib data handler that combines Alpha158 features with news factors.

Same pattern as Alpha158WithFundamentals — load Alpha158 normally, then merge
news factors into the feature group.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from qlib.contrib.data.handler import Alpha158, _DEFAULT_LEARN_PROCESSORS, check_transform_proc
from qlib.contrib.data.loader import Alpha158DL
from qlib.data.dataset.handler import DataHandlerLP

ROOT = Path(__file__).parent
DEFAULT_NEWS_PARQUET = ROOT / "data" / "news_factors.parquet"


class Alpha158WithNews(Alpha158):
    """Alpha158 + 9 news factors loaded from parquet."""

    def __init__(self, *args, news_parquet: str | None = None, **kwargs):
        if news_parquet is None:
            news_parquet = str(DEFAULT_NEWS_PARQUET)
        self._news_parquet = news_parquet
        super().__init__(*args, **kwargs)

    def setup_data(self, *args, **kwargs):
        super().setup_data(*args, **kwargs)
        news = self._load_news()
        if news is None or news.empty:
            return

        df = self._data
        merged_features = news.reindex(df.index)
        feature_df = df["feature"]
        label_df = df["label"]
        feature_full = pd.concat([feature_df, merged_features], axis=1)
        feature_full.columns = pd.MultiIndex.from_product([["feature"], feature_full.columns])
        full = pd.concat([feature_full, label_df.pipe(
            lambda d: d.set_axis(pd.MultiIndex.from_product([["label"], d.columns]), axis=1)
        )], axis=1)
        self._data = full

        if hasattr(self, "_infer") and self._infer is not None:
            self._infer = self._reapply_to(self._infer, news)
        if hasattr(self, "_learn") and self._learn is not None:
            self._learn = self._reapply_to(self._learn, news)

    def _reapply_to(self, df: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
        merged_features = news.reindex(df.index)
        feature_df = df["feature"]
        label_df = df["label"]
        feature_full = pd.concat([feature_df, merged_features], axis=1)
        feature_full.columns = pd.MultiIndex.from_product([["feature"], feature_full.columns])
        full = pd.concat([feature_full, label_df.pipe(
            lambda d: d.set_axis(pd.MultiIndex.from_product([["label"], d.columns]), axis=1)
        )], axis=1)
        return full

    def _load_news(self) -> pd.DataFrame:
        df = pd.read_parquet(self._news_parquet)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["instrument"] = df["instrument"].str.upper()
        df = df.set_index(["datetime", "instrument"]).sort_index()
        factor_cols = [c for c in df.columns if c.startswith("news_")]
        return df[factor_cols]
