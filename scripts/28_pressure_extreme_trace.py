"""Where do flagged high-pressure records at quiet stations come from? Usage: 28_pressure_extreme_trace.py WORK_DIR"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
S = Path(sys.argv[1])
sys.path.insert(0, str(Path(__file__).resolve().parent))
import neighbour_reference as nb  # noqa: E402

co = pd.read_csv(S / "pool/pool_coords.csv", index_col=0)
al = pd.read_parquet(S / "pool/pool_all.parquet", columns=["station_id", "timestamp_utc", "pressure_hpa"]).dropna()
al["timestamp_utc"] = pd.to_datetime(al.timestamp_utc, utc=True)
ids = sorted(set(al.station_id) & set(co.index))
co = co.loc[ids]
ix = {s: i for i, s in enumerate(ids)}
grid = pd.date_range("2023-01-01", "2025-12-31 23:50", freq="10min", tz="UTC")
P = np.full((len(grid), len(ids)), np.nan, np.float32)
for s, g in al.groupby("station_id"):
    P[:, ix[s]] = g.set_index("timestamp_utc").pressure_hpa.sort_index().reindex(grid, method="ffill", tolerance=pd.Timedelta("14min")).values
del al
tr = np.asarray(grid < pd.Timestamp("2025-01-01", tz="UTC"))
te = ~tr
A = P - np.nanmedian(P[tr], axis=0)
a, b = np.radians(co.lat.values), np.radians(co.lon.values)
D = 2 * 6371.0088 * np.arcsin(np.sqrt(np.sin((a[:, None] - a[None, :]) / 2) ** 2 + np.cos(a[:, None]) * np.cos(a[None, :]) * np.sin((b[:, None] - b[None, :]) / 2) ** 2))
np.fill_diagonal(D, np.inf)
nn = np.argsort(D, axis=1)[:, :5]
res = np.full(A.shape, np.nan, np.float32)
for i in range(len(ids)):
    res[:, i] = A[:, i] - nb.guarded_median(A[:, nn[i]].astype(float), 8.0, 3, True)
fl = np.isfinite(res) & (np.abs(res) > 8)
share = fl.sum(0) / np.isfinite(res).sum(0)
quiet = share < 0.01
hi = np.nanquantile(A[tr], .99, axis=0)
ev = (A > hi) & np.isfinite(res) & te[:, None]
ev[:, ~quiet] = False
f = ev & fl
per = pd.Series(f.sum(0), index=ids).sort_values(ascending=False)
print("high events", int(ev.sum()), "flagged", int(f.sum()), "stations with any", int((per > 0).sum()), "stations with events", int((ev.sum(0) > 0).sum()))
print("top 1/3/5 share of flagged", [round(float(per.head(k).sum() / per.sum()), 3) for k in (1, 3, 5)])
print("flagged: median own anomaly", round(float(np.nanmedian(A[f])), 1), "median neighbour-consensus anomaly", round(float(np.nanmedian(A[f] - res[f])), 1), "median residual", round(float(np.nanmedian(res[f])), 1))
print("unflagged: median own anomaly", round(float(np.nanmedian(A[ev & ~fl])), 1), "consensus", round(float(np.nanmedian((A - res)[ev & ~fl])), 1))
ret = 1 - (f.sum(0) / np.maximum(ev.sum(0), 1))
print("median station retention", round(float(np.median(ret[ev.sum(0) > 0])), 3))
top = per.index[0]
print("excluding top station", round(float(1 - (per.sum() - per.iloc[0]) / (ev.sum() - ev[:, ix[top]].sum())), 3))
