"""Pressure-specific release tests and altitude check. Usage: 27_pressure_release_tests.py WORK_DIR OUT_DIR

1. Does pressure quarantine remove real pressure extremes (the rule acts on pressure, so gust tests cannot show this)?
2. Are flagged stations explained by altitude difference from their references?
3. Row-level recall against event-level and station-level views.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200, "display.max_rows", 60)
S, O = Path(sys.argv[1]), Path(sys.argv[2])
O.mkdir(parents=True, exist_ok=True)
R = Path(__file__).resolve().parent
sys.path.insert(0, str(R))
import neighbour_reference as nb  # noqa: E402

TAU, Q = 8.0, 3
co = pd.read_csv(S / "pool/pool_coords.csv", index_col=0)
al = pd.read_parquet(S / "pool/pool_all.parquet", columns=["station_id", "timestamp_utc", "pressure_hpa"]).dropna()
al["timestamp_utc"] = pd.to_datetime(al.timestamp_utc, utc=True)
ids = sorted(set(al.station_id) & set(co.index))
co = co.loc[ids]
ix = {s: i for i, s in enumerate(ids)}
grid = pd.date_range("2023-01-01", "2025-12-31 23:50", freq="10min", tz="UTC")
P = np.full((len(grid), len(ids)), np.nan, np.float32)
for s, g in al.groupby("station_id"):
    if s in ix:
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
    med = nb.guarded_median(A[:, nn[i]].astype(float), TAU, Q, True)
    res[:, i] = A[:, i] - med
fl = np.isfinite(res) & (np.abs(res) > TAU)
print("residuals done", flush=True)

# 1. state dependence on pressure, evaluation period, using the network median anomaly as the weather state
state = np.nanmedian(A, axis=1)
valid = np.isfinite(res) & te[:, None]
stat = pd.DataFrame({"s": ids, "share": fl.sum(0) / np.isfinite(res).sum(0)}).set_index("s")
quiet = (stat.share < 0.01).values  # stations without sustained episodes
rows = []
dec = pd.Series(pd.qcut(state[te], 10, labels=False, duplicates="drop")).fillna(-1).astype(int).to_numpy()
dfull = np.full(len(grid), -1)
dfull[te] = dec
for name, cols in (("all stations", np.ones(len(ids), bool)), ("stations below 1% flagged", quiet)):
    r = [fl[dfull == q][:, cols].sum() / valid[dfull == q][:, cols].sum() for q in range(int(dec.max()) + 1)]
    rows.append({"group": name, "spread_p90_p10": np.quantile(r, .9) - np.quantile(r, .1), "lowest_decile": r[0], "highest_decile": r[-1], "overall": fl[te][:, cols].sum() / valid[te][:, cols].sum()})
t = pd.DataFrame(rows)
t.to_csv(O / "pressure_state_dependence.csv", index=False)
print(t.round(5).to_string(index=False), flush=True)
# retention of each station's own pressure extremes, stations without sustained episodes
lo = np.nanquantile(A[tr], .01, axis=0)
hi = np.nanquantile(A[tr], .99, axis=0)
rows = []
for name, cols in (("all stations", np.ones(len(ids), bool)), ("stations below 1% flagged", quiet)):
    e_lo = (A < lo) & valid
    e_hi = (A > hi) & valid
    rows.append({"group": name, "low_events": int(e_lo[:, cols].sum()), "low_retention": 1 - (e_lo & fl)[:, cols].sum() / e_lo[:, cols].sum(), "high_events": int(e_hi[:, cols].sum()), "high_retention": 1 - (e_hi & fl)[:, cols].sum() / e_hi[:, cols].sum()})
t = pd.DataFrame(rows)
t.to_csv(O / "pressure_extreme_retention.csv", index=False)
print(t.round(4).to_string(index=False), flush=True)
# network-coherent extremes: hours when the network median is in its lowest 1%: are healthy stations flagged?
low = te & (state < np.nanquantile(state[tr], .01))
print("hours-of-network-low records", int(low.sum()), "flag rate quiet stations", fl[low][:, quiet].sum() / max(np.isfinite(res[low][:, quiet]).sum(), 1), flush=True)

# 2. altitude
alt = co.alt.values
dalt = np.abs(alt - np.median(alt[nn], axis=1))
share = stat.share.values
print("spearman flagged share vs altitude difference", round(float(pd.Series(share).corr(pd.Series(dalt), method="spearman")), 3), flush=True)
g = pd.DataFrame({"dalt": dalt, "share": share})
g["b"] = pd.cut(g.dalt, [-1, 50, 150, 300, 3000])
t = g.groupby("b", observed=True).agg(n=("share", "size"), median_share=("share", "median"), stations_over_1pct=("share", lambda x: int((x > .01).sum())), mean_share=("share", "mean"))
t.to_csv(O / "flags_by_altitude_difference.csv")
print(t.round(4).to_string(), flush=True)
# seasonal residual amplitude vs altitude difference, for quiet stations
mres = pd.DataFrame(res, index=grid, columns=ids).resample("MS").median()
amp = (mres.max() - mres.min()).values
print("monthly residual range, quiet stations, median hPa", round(float(np.nanmedian(amp[quiet])), 2), "| spearman with altitude difference", round(float(pd.Series(amp[quiet]).corr(pd.Series(dalt[quiet]), method="spearman")), 3), flush=True)
print("monthly residual range, by altitude difference (quiet)", pd.Series(amp[quiet]).groupby(pd.cut(pd.Series(dalt[quiet]), [-1, 50, 150, 300, 3000]), observed=True).median().round(2).to_dict(), flush=True)
print("CHECKSDONE")
