"""Assign quality strata to the DWD multivariate product (protocol parallel to AVAMET).

Computes a TRAIN corruption index per station from deterministic rule-trigger
rates (hard FAIL rate + pressure-stuck SUSPECT rate), ranks the stations, and
assigns a balanced 10/10/10 clean/medium/dirty stratification, mirroring the
AVAMET station-selection protocol. Writes the stratum column back into the P0
product so scripts 11/13/15 can run unchanged on DWD.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd


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
    ap.add_argument("--train-end", type=str, default="2025-01-01")
    ap.add_argument("--n-per-stratum", type=int, default=10)
    args = ap.parse_args()

    suite = _load(Path(__file__).resolve().parent / "11_measurement_fault_suite.py", "measurement_fault_suite")

    p0_path = args.results_dir / "products" / "P0" / "data.parquet"
    df = pd.read_parquet(p0_path)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    train_end = pd.Timestamp(args.train_end, tz="UTC")
    train = df[df["timestamp_utc"] < train_end].copy()

    fail, _ = suite.rule_flags(train)
    stuck = suite.compute_stuck_flags(train)
    train = train.assign(_fail=fail.to_numpy(), _stuck=stuck.to_numpy())

    per = train.groupby("station_id").agg(
        fail_rate=("_fail", "mean"),
        stuck_rate=("_stuck", "mean"),
        n=("_fail", "size"),
    )
    # corruption index: equal-weight hard + soft trigger rates
    per["corruption_index"] = per["fail_rate"] + per["stuck_rate"]
    per = per.sort_values("corruption_index")
    k = args.n_per_stratum
    order = per.index.tolist()
    strata = {}
    for s in order[:k]:
        strata[s] = "clean"
    for s in order[-k:]:
        strata[s] = "dirty"
    for s in order[k:-k]:
        strata[s] = "medium"

    df["stratum"] = df["station_id"].map(strata).fillna("medium")
    df.to_parquet(p0_path, index=False)
    per["stratum"] = per.index.map(strata)
    per.to_csv(args.results_dir / "dwd_station_strata.csv")
    print(per[["fail_rate", "stuck_rate", "corruption_index", "stratum"]].to_string())
    print(f"\nAssigned strata -> {p0_path}")
    print(df.groupby("stratum")["station_id"].nunique().to_string())


if __name__ == "__main__":
    main()
