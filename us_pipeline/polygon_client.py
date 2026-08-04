"""Shared Polygon REST access: client construction and aggregate-bar fetching.

Used by both the bulk downloader (download_polygon.py) and the chart app
(viz/server.py), so the retry policy lives in one place.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from dotenv import load_dotenv
from polygon import RESTClient
from polygon.exceptions import BadResponse

ROOT = Path(__file__).parent

# Polygon Stocks Starter tier: ~5-year rolling window. Minute bars start ~2021-08.
TIER_EARLIEST = "2021-08-02"


class TickerNotFound(Exception):
    """Polygon has no data for this ticker (404 / NOT_FOUND)."""


def get_client() -> RESTClient:
    """Build a RESTClient from POLYGON_API_KEY (env or us_pipeline/../.env)."""
    load_dotenv(ROOT.parent / ".env")
    api_key = os.environ.get("POLYGON_API_KEY")
    if not api_key:
        raise SystemExit("POLYGON_API_KEY not set in environment or .env")
    return RESTClient(api_key)


def fetch_aggs(
    client: RESTClient,
    ticker: str,
    multiplier: int,
    timespan: str,
    from_: str,
    to: str,
    adjusted: bool = True,
    retries: int = 4,
) -> list:
    """Fetch aggregate bars, retrying transient failures with exponential backoff.

    Returns Polygon Agg objects (attributes: timestamp, open, high, low, close,
    volume, vwap, transactions). Raises TickerNotFound if the symbol has no data,
    RuntimeError if every attempt failed.
    """
    last_err = None
    for attempt in range(retries):
        try:
            # Polygon's REST list_aggs auto-paginates.
            return list(
                client.list_aggs(
                    ticker=ticker,
                    multiplier=multiplier,
                    timespan=timespan,
                    from_=from_,
                    to=to,
                    adjusted=adjusted,
                    sort="asc",
                    limit=50000,
                )
            )
        except BadResponse as e:
            msg = str(e)
            if "NOT_FOUND" in msg or "404" in msg:
                raise TickerNotFound(ticker) from e
            last_err = e
        except Exception as e:
            last_err = e
        time.sleep(2 ** attempt)
    raise RuntimeError(f"{ticker}: {last_err}")
