"""Export exploratory pressure-extreme diagnostics. Usage: 28_pressure_extreme_trace.py WORK_DIR OUT_DIR

The below-1% subset uses flags over the full archive; excluding its largest
contributor is a post-hoc influence check, not an independent release test.
AVAMET civil-clock values retain the historical timestamp_utc column name.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
S = Path(sys.argv[1])
O = Path(sys.argv[2])
O.mkdir(parents=True, exist_ok=True)
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


# Export counts as well as rates; no station identifiers are released.
lo = np.nanquantile(A[tr], .01, axis=0)
valid_test = np.isfinite(res) & te[:, None]
low_events = (A < lo) & valid_test
high_events = (A > hi) & valid_test
without_top = quiet.copy()
without_top[ix[top]] = False
rows = []
for name, cols in [("all stations", np.ones(len(ids), bool)),
                   ("stations below 1% flagged", quiet),
                   ("below 1% flagged, largest high-tail contributor excluded", without_top)]:
    nl = int(low_events[:, cols].sum())
    nh = int(high_events[:, cols].sum())
    fl = int((low_events[:, cols] & (np.abs(res[:, cols]) > 8)).sum())
    fh = int((high_events[:, cols] & (np.abs(res[:, cols]) > 8)).sum())
    rows.append(dict(group=name, n_stations=int(cols.sum()),
                     low_events=nl, low_flagged=fl, low_retention=1-fl/nl,
                     high_events=nh, high_flagged=fh, high_retention=1-fh/nh,
                     exploratory_subset=name != "all stations"))
pd.DataFrame(rows).to_csv(O / "pressure_extreme_influence.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False), flush=True)
