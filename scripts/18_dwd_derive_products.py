"""Derive P1 (delete-all-flags) and P3 (conservative FAIL-only) products from the
DWD P0 product, using the SAME deterministic rule/stuck logic as the AVAMET
pipeline (imported from 11_measurement_fault_suite.py).

Row-level deletion semantics, matching the AVAMET products:
- P3 (constraint-selected conservative FAIL-only): null the measurement columns
  on rows flagged FAIL; SUSPECT (pressure stuck) rows are KEPT.
- P1 (delete-all-flags): null the measurement columns on rows flagged FAIL OR
  SUSPECT.

This lets 13_spatial_quarantine_archive.py run unchanged on DWD.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd

MEAS_COLS = ["pressure_hpa", "wind_mean", "wind_gust", "wind_dir"]


def _load(mod_path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, mod_path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=repo / "runs_additional" / "DWD" / "results_v3")
    args = ap.parse_args()

    suite = _load(Path(__file__).resolve().parent / "11_measurement_fault_suite.py", "measurement_fault_suite")

    prod_dir = args.results_dir / "products"
    p0 = pd.read_parquet(prod_dir / "P0" / "data.parquet")
    p0["timestamp_utc"] = pd.to_datetime(p0["timestamp_utc"], utc=True)
    p0 = p0.sort_values(["station_id", "timestamp_utc"]).reset_index(drop=True)

    fail, suspect = suite.rule_flags(p0)
    print(f"FAIL rate={fail.mean():.5f}  SUSPECT(stuck) rate={suspect.mean():.5f}")

    cols = [c for c in MEAS_COLS if c in p0.columns]

    p3 = p0.copy()
    p3.loc[fail.to_numpy(), cols] = pd.NA
    (prod_dir / "P3").mkdir(parents=True, exist_ok=True)
    p3.to_parquet(prod_dir / "P3" / "data.parquet", index=False)

    p1 = p0.copy()
    p1.loc[(fail | suspect).to_numpy(), cols] = pd.NA
    (prod_dir / "P1").mkdir(parents=True, exist_ok=True)
    p1.to_parquet(prod_dir / "P1" / "data.parquet", index=False)

    def rate(df, col):
        return float(df[col].isna().mean())

    print(f"P0 wind_gust missing={rate(p0,'wind_gust'):.5f}  pressure missing={rate(p0,'pressure_hpa'):.5f}")
    print(f"P3 wind_gust missing={rate(p3,'wind_gust'):.5f}  pressure missing={rate(p3,'pressure_hpa'):.5f}")
    print(f"P1 wind_gust missing={rate(p1,'wind_gust'):.5f}  pressure missing={rate(p1,'pressure_hpa'):.5f}")
    print(f"Wrote {prod_dir / 'P1' / 'data.parquet'} and {prod_dir / 'P3' / 'data.parquet'}")


if __name__ == "__main__":
    main()
