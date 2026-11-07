"""Worked example on a small synthetic network. Runs in seconds and needs no licensed data.

    python examples/quickstart.py

It builds twelve stations that report at 5, 10 or 20 minutes and share the same weather.
Two neighbours of one station die, and another station develops a pressure offset.
The example then shows the four ideas of the workflow:

1. effective redundancy: how many references are really available;
2. false isolation: a healthy station outvoted by failed references;
3. detection: the offset is found by comparing with neighbours;
4. the action: the value is quarantined, not deleted, and an audit record is written.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import neighbour_reference as nb  # noqa: E402

TAU = 8.0  # pressure threshold (hPa)
QUORUM = 3
TOLERANCE_MIN = 14
OUT = ROOT / "examples" / "output"


def make_network(seed: int = 7):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2025-01-01", periods=7 * 288, freq="5min", tz="UTC")
    weather = 6 * np.sin(np.arange(len(t)) / 400.0) + np.cumsum(rng.normal(0, 0.01, len(t)))
    stations = {}
    for i in range(12):
        step = (1, 2, 4)[i % 3]  # 5, 10 or 20 minutes
        altitude_effect = rng.uniform(-40, 0)
        p = 1013 + altitude_effect + weather + rng.normal(0, 0.2, len(t))
        stations[f"S{i:02d}"] = pd.Series(p, index=t).iloc[::step]
    return stations, t


def references(stations, target, index, tolerance):
    """Anomalies of the five other stations next to the target, on the target's times."""
    names = sorted(stations)
    i = names.index(target)
    chosen = [names[(i + k) % len(names)] for k in range(1, 6)]
    cols = {}
    for s in chosen:
        v = stations[s]
        valid = v.where((v > 500) & (v < 1100))  # hard-invalid values never enter the vote
        cols[s] = nb.align_series(valid - valid.median(), index, tolerance)
    return pd.DataFrame(cols, index=index)


def main() -> None:
    stations, t = make_network()

    print("1. Effective redundancy for station S00 (five references requested)")
    target = stations["S00"]
    exact = references(stations, "S00", target.index, 0).notna().sum(axis=1)
    aligned = references(stations, "S00", target.index, TOLERANCE_MIN).notna().sum(axis=1)
    print(f"   exact timestamps: median m = {exact.median():.0f}, records with m <= 2: {(exact <= 2).mean():.0%}")
    print(f"   aligned:          median m = {aligned.median():.0f}, records with m <= 2: {(aligned <= 2).mean():.0%}")
    print(f"   a median of {int(aligned.median())} references tolerates {nb.tolerated_failures(int(aligned.median()))} failed references")

    print("\n2. False isolation of healthy station S00 when references die")
    dead = {k: v.copy() for k, v in stations.items()}
    window = t[1000:1300]
    for s in ("S01", "S02", "S03"):
        dead[s].loc[dead[s].index.isin(window)] = 0.0  # sentinel value
    idx = target.index[target.index.isin(window)]
    anom = (target - target.median()).reindex(idx)

    naive = pd.DataFrame({s: nb.align_series(dead[s] - stations[s].median(), idx, TOLERANCE_MIN) for s in ("S01", "S02", "S03", "S04", "S05")})
    guarded = references(dead, "S00", idx, TOLERANCE_MIN)
    naive_flag = (anom - naive.median(axis=1)).abs() > TAU
    med = nb.guarded_median(guarded.to_numpy(), TAU, QUORUM, refhealth=True)
    guard_flag = (anom - med).abs() > TAU
    print(f"   three of five references dead, no guard: {naive_flag.mean():.0%} of healthy records quarantined")
    print(f"   with guard: {guard_flag.mean():.0%} quarantined; {np.isnan(med).mean():.0%} abstained, because only two references remain")

    print("\n3. Detecting a real offset at station S06")
    s = "S06"
    faulty = stations[s].copy()
    start, stop = t[800], t[800 + 144]  # twelve hours
    faulty.loc[(faulty.index >= start) & (faulty.index < stop)] += 12.0
    data = dict(stations)
    data[s] = faulty
    idx = faulty.index
    ref = references(data, s, idx, TOLERANCE_MIN)
    med = nb.guarded_median(ref.to_numpy(), TAU, QUORUM)
    residual = (faulty - stations[s].median()) - med
    flag = residual.abs() > TAU
    truth = (idx >= start) & (idx < stop)
    print(f"   recall {flag[truth].mean():.2f}; false flags outside the fault {flag[~truth].sum()} of {(~truth).sum()}")

    print("\n4. Action: quarantine with an audit record")
    audit = pd.DataFrame({
        "timestamp": idx[flag.to_numpy()],
        "station": s,
        "variable": "pressure",
        "value_hpa": faulty[flag].round(2).to_numpy(),
        "evidence": "neighbour comparison",
        "residual_hpa": residual[flag].round(2).to_numpy(),
        "references": ref[flag.to_numpy()].notna().sum(axis=1).to_numpy(),
        "action": "quarantine",
    })
    OUT.mkdir(exist_ok=True)
    audit.to_csv(OUT / "audit_record.csv", index=False)
    print(f"   {len(audit)} records withheld; values kept in {OUT / 'audit_record.csv'}")
    print(audit.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
