"""Convert downloaded Polygon CSVs into Qlib's binary format.

This wraps scripts/dump_bin.py but performs two preparation steps first:
  1. Stage CSVs into a temp dir with renamed columns the Qlib loader expects
     (the file's columns are already correct, but we add explicit lower-case
     OHLCV column names matching Alpha158's `$open, $high, $low, $close,
     $volume, $vwap`).
  2. Build instruments/all.txt and instruments/sp500.txt.

After this script runs, Qlib can be initialized with
  qlib.init(provider_uri="us_pipeline/data/qlib_bin", region="us")

Usage:
  python us_pipeline/to_qlib_bin.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
RAW_DIR = ROOT / "data" / "raw"
QLIB_DIR = ROOT / "data" / "qlib_bin"
INSTR_LIST = ROOT / "data" / "instruments" / "sp500.txt"
DUMP_BIN = Path(__file__).resolve().parent.parent / "scripts" / "dump_bin.py"


def stage_csvs(staged_dir: Path) -> tuple[int, str, str]:
    """Copy raw CSVs to staged dir keeping only Qlib-required columns.

    Returns (n_files, min_date, max_date).
    """
    if staged_dir.exists():
        shutil.rmtree(staged_dir)
    staged_dir.mkdir(parents=True)

    keep = ["date", "open", "high", "low", "close", "volume", "vwap"]
    csvs = sorted(RAW_DIR.glob("*.csv"))
    min_d, max_d = "9999-12-31", "0000-01-01"
    n = 0
    for src in csvs:
        df = pd.read_csv(src)
        if df.empty:
            continue
        df = df[keep].sort_values("date").drop_duplicates("date")
        # `factor` column needed by Qlib for adjusted prices; Polygon already
        # delivers split-adjusted prices, so factor is constant 1.0.
        df["factor"] = 1.0
        df.to_csv(staged_dir / src.name, index=False)
        if df["date"].iloc[0] < min_d:
            min_d = df["date"].iloc[0]
        if df["date"].iloc[-1] > max_d:
            max_d = df["date"].iloc[-1]
        n += 1
    return n, min_d, max_d


def run_dump_bin(staged_dir: Path) -> None:
    if QLIB_DIR.exists():
        shutil.rmtree(QLIB_DIR)
    QLIB_DIR.mkdir(parents=True)
    cmd = [
        sys.executable,
        str(DUMP_BIN),
        "dump_all",
        "--data_path", str(staged_dir),
        "--qlib_dir", str(QLIB_DIR),
        "--freq", "day",
        "--max_workers", "8",
        "--include_fields", "open,high,low,close,volume,vwap,factor",
        "--date_field_name", "date",
        # symbol auto-derived from filename when missing
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def write_sp500_instruments() -> None:
    """Qlib's instruments/<name>.txt format: <symbol>\t<start>\t<end> per line."""
    inst_dir = QLIB_DIR / "instruments"
    all_file = inst_dir / "all.txt"
    if not all_file.exists():
        print("WARN: instruments/all.txt not produced by dump_bin")
        return
    rows = [line.split("\t") for line in all_file.read_text().splitlines() if line.strip()]
    by_symbol = {r[0].upper(): r for r in rows}

    sp500_set = {t.strip().upper() for t in INSTR_LIST.read_text().splitlines() if t.strip()}
    out_lines = []
    missing = []
    for sym in sorted(sp500_set):
        if sym in by_symbol:
            out_lines.append("\t".join(by_symbol[sym]))
        else:
            missing.append(sym)

    sp500_file = inst_dir / "sp500.txt"
    sp500_file.write_text("\n".join(out_lines) + "\n")
    print(f"Wrote {len(out_lines)} symbols to {sp500_file}")
    if missing:
        print(f"  ({len(missing)} S&P 500 symbols not in qlib data, e.g. {missing[:5]})")


def main() -> None:
    staged = ROOT / "data" / "_staged_csv"
    print("Staging CSVs...")
    n, min_d, max_d = stage_csvs(staged)
    print(f"  staged {n} files, date range {min_d} -> {max_d}")

    print("\nDumping to Qlib bin...")
    run_dump_bin(staged)

    print("\nBuilding sp500 instrument list...")
    write_sp500_instruments()

    print(f"\nDone. Qlib data at: {QLIB_DIR}")
    print("Test with:")
    print('  python -c "import qlib; qlib.init(provider_uri=\'us_pipeline/data/qlib_bin\', region=\'us\'); '
          'from qlib.data import D; print(D.features([\\"AAPL\\"], [\\"$close\\"], '
          'start_time=\'2026-04-01\').tail())"')


if __name__ == "__main__":
    main()
