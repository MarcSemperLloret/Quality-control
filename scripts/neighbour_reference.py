"""Neighbour reference used by the spatial QC scripts.

A target station is compared with the median of its references. This module holds
the three safeguards described in the manuscript:

* time alignment: a reference is its latest value within a tolerance, because stations
  reporting at different intervals seldom share exact timestamps;
* guard: invalid references are removed and the action abstains below a quorum;
* reference health: a reference far from the median of the other references is removed.

Targets are never aligned: only references are.

Environment variables (defaults are a sample-only setup; the AVAMET manuscript
configuration additionally requires NB_MODE=pool and NB_POOL_DIR):

  NB_TOL_MIN    reference tolerance in minutes; 0 means exact timestamps (default 14)
  NB_MODE       orig: references only from the frame passed in
                sample: also the other stations of the evaluation sample (default)
                pool: sample plus the network pool in NB_POOL_DIR
  NB_POOL_DIR   folder with neighbors.json and pool_extra.parquet
  NB_QUORUM     minimum healthy references (default 3)
  NB_REFHEALTH  1 applies the reference-health screen (default 1)
  NB_WSCALE     km/h per wind unit: 1 for km/h data, 3.6 for m/s data (default 1)

The originally submitted behaviour is NB_MODE=orig NB_TOL_MIN=0 NB_QUORUM=2 NB_REFHEALTH=0.
"""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

TOL = float(os.environ.get("NB_TOL_MIN", "14"))
MODE = os.environ.get("NB_MODE", "sample")
POOL_DIR = os.environ.get("NB_POOL_DIR", "")
QUORUM = int(os.environ.get("NB_QUORUM", "3"))
REFHEALTH = os.environ.get("NB_REFHEALTH", "1") == "1"
WSCALE = float(os.environ.get("NB_WSCALE", "1"))
TRAIN_END = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
REF_HEALTH_MIN_PEERS = 3
REF_HEALTH_MAD_Z = 3.0
BG = None
_extra = None


def _nanmedian(values: np.ndarray, axis: int = 1) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(values, axis=axis)


def align_series(values: pd.Series, index: pd.DatetimeIndex, tolerance_min: float) -> pd.Series:
    """Latest value of a station at each time in index, within the tolerance."""
    v = values.dropna().sort_index()
    if tolerance_min > 0:
        return v.reindex(index, method="ffill", tolerance=pd.Timedelta(minutes=tolerance_min))
    return v.reindex(index)


def reference_health_filter(ref: np.ndarray, threshold: float) -> np.ndarray:
    """Remove references far from the median of the other references. Uses no target data."""
    ref = np.asarray(ref, dtype=float)
    out = ref.copy()
    finite = np.isfinite(ref)
    for j in range(ref.shape[1]):
        peers = ref.copy()
        peers[:, j] = np.nan
        count = np.isfinite(peers).sum(axis=1)
        med = _nanmedian(peers)
        mad = _nanmedian(np.abs(peers - med[:, None]))
        cutoff = np.maximum(threshold, REF_HEALTH_MAD_Z * 1.4826 * mad)
        bad = finite[:, j] & (count >= REF_HEALTH_MIN_PEERS) & (np.abs(ref[:, j] - med) > cutoff)
        out[bad, j] = np.nan
    return out


def guarded_median(ref: np.ndarray, threshold: float, quorum: int = 3, refhealth: bool = True) -> np.ndarray:
    """Median of healthy references, or NaN (abstain) below the quorum. Invalid values must be NaN."""
    r = reference_health_filter(ref, threshold) if refhealth else np.asarray(ref, dtype=float)
    med = _nanmedian(r)
    med[np.isfinite(r).sum(axis=1) < quorum] = np.nan
    return med


def tolerated_failures(m: int) -> int:
    """Failed references a median of m references can absorb."""
    return max((m - 1) // 2, 0)


def extra():
    global _extra
    if MODE != "pool":
        return None
    if _extra is None:
        _extra = pd.read_parquet(Path(POOL_DIR) / "pool_extra.parquet")
        _extra["timestamp_utc"] = pd.to_datetime(_extra["timestamp_utc"], utc=True)
    return _extra


def neighbors(default):
    if MODE != "pool":
        return default
    return json.loads((Path(POOL_DIR) / "neighbors.json").read_text())


def medians(base):
    out = pd.Series(base, dtype=float)
    e = extra()
    if e is not None:
        m = e[e.timestamp_utc < TRAIN_END].groupby("station_id").pressure_hpa.median()
        out = pd.concat([out, m[~m.index.isin(out.index)]])
    return out


def u90(base):
    e = extra()
    if e is None:
        return base
    u = e[e.timestamp_utc < TRAIN_END].groupby("station_id").wind_gust.quantile(0.90)
    return pd.concat([base, u[~u.index.isin(base.index)]])


def wide(df, col, index):
    """Reference matrix for one variable on the target time index."""
    frames = [df[["station_id", "timestamp_utc", col]]]
    have = set(df.station_id.unique())
    if MODE in ("sample", "pool") and BG is not None:
        b = BG[~BG.station_id.isin(have)]
        frames.append(b[["station_id", "timestamp_utc", col]])
        have |= set(b.station_id.unique())
    e = extra()
    if e is not None:
        frames.append(e.loc[~e.station_id.isin(have), ["station_id", "timestamp_utc", col]])
    src = pd.concat(frames, ignore_index=True).dropna(subset=[col])
    cols = {}
    for s, g in src.groupby("station_id"):
        cols[s] = align_series(g.set_index("timestamp_utc")[col], index, TOL)
    for s in df.station_id.unique():
        if s not in cols:
            cols[s] = pd.Series(index=index, dtype=float)
    return pd.DataFrame(cols, index=index)


def anom(df, index, base_medians):
    w = wide(df, "pressure_hpa", index)
    return w.subtract(medians(base_medians).reindex(w.columns), axis="columns")


def screen(neigh, threshold):
    if not REFHEALTH:
        return neigh
    return pd.DataFrame(reference_health_filter(neigh.to_numpy(dtype=float), threshold), index=neigh.index, columns=neigh.columns)


def regrid(df, step="10min"):
    """Put every station of a pressure frame on a regular grid, for the neighbour-failure experiment."""
    if TOL <= 0:
        return df
    t = df.timestamp_utc
    grid = pd.date_range(t.min().floor(step), t.max(), freq=step)
    out = []
    for s, g in df.groupby("station_id"):
        v = align_series(g.set_index("timestamp_utc").pressure_hpa, grid, TOL)
        out.append(pd.DataFrame({"station_id": s, "timestamp_utc": grid, "pressure_hpa": v.values}).dropna(subset=["pressure_hpa"]))
    return pd.concat(out, ignore_index=True)
