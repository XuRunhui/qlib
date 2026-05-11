"""Custom Qlib processor for sector-neutral cross-sectional rank normalization.

Replaces CSRankNorm: instead of ranking all 503 stocks per day, rank within
each (date, sector) group, then standard-normalize.

Usage in YAML:
    learn_processors:
        - class: DropnaLabel
        - class: SectorNeutralRank
          module_path: us_pipeline.sector_processor
          kwargs:
              fields_group: label
              sector_csv: us_pipeline/data/instruments/sectors.csv
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from qlib.data.dataset.processor import Processor


class SectorNeutralRank(Processor):
    """Rank label within each (date, sector) group, then standard-normalize.

    Equivalent to CSRankNorm but partitioned by sector. This removes the
    component of returns that comes from sector-wide moves (sector beta),
    leaving only stock-specific alpha for the model to learn.
    """

    def __init__(self, fields_group: str = "label", sector_csv: str | None = None):
        self.fields_group = fields_group
        if sector_csv is None:
            sector_csv = str(Path(__file__).parent / "data" / "instruments" / "sectors.csv")
        sectors = pd.read_csv(sector_csv)
        # Map symbol -> sector. Qlib stores instruments uppercase.
        self._sector_map = dict(zip(sectors["symbol"].str.upper(), sectors["sector"]))

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        # df is multi-indexed by (datetime, instrument); columns are like ("label", "LABEL0")
        if isinstance(df.columns, pd.MultiIndex):
            cols = [c for c in df.columns if c[0] == self.fields_group]
        else:
            cols = [c for c in df.columns if self.fields_group in str(c)]

        if not cols:
            return df

        # Attach sector as an index level temporarily
        instruments = df.index.get_level_values("instrument").str.upper()
        sectors = pd.Index(
            [self._sector_map.get(i, "Unknown") for i in instruments],
            name="sector",
        )

        for col in cols:
            s = df[col]
            # Group by (datetime, sector), rank within each group, then standard-normalize
            grouper = [df.index.get_level_values("datetime"), sectors]
            ranked = s.groupby(grouper).transform(
                lambda x: (x.rank(pct=True) - 0.5) * 3.46  # uniform [-1.73, 1.73] ~ N(0,1) variance
            )
            df[col] = ranked

        return df

    def is_for_infer(self) -> bool:
        # Only used during fit/learn (label transform), not at inference
        return False
