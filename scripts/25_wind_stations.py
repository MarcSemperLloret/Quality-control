"""Are the stations driving the wind retention loss showing real wind or instrument behaviour? Usage: wind_stations.py SCRATCH_DIR NET_DIR"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 220, "display.max_rows", 80, "display.max_columns", 30)
S, N = Path(sys.argv[1]), Path(sys.argv[2])
co = pd.read_csv(S / "pool/pool_coords.csv", index_col=0)
w = pd.read_csv(N / "wind_rule_network.csv").sort_values("flag", ascending=False)
top = list(w.station_id.head(10))
al = pd.read_parquet(S / "pool/pool_all.parquet", columns=["station_id", "timestamp_utc", "wind_gust", "wind_mean"])
al["timestamp_utc"] = pd.to_datetime(al.timestamp_utc, utc=True)
al = al.dropna(subset=["wind_gust"])
al["h"] = al.timestamp_utc.dt.floor("h")
g = al.pivot_table(index="h", columns="station_id", values="wind_gust", aggfunc="max")
ids = [s for s in g.columns if s in co.index]
c = co.loc[ids]
a, b = np.radians(c.lat.values), np.radians(c.lon.values)
D = 2 * 6371.0088 * np.arcsin(np.sqrt(np.sin((a[:, None] - a[None, :]) / 2) ** 2 + np.cos(a[:, None]) * np.cos(a[None, :]) * np.sin((b[:, None] - b[None, :]) / 2) ** 2))
np.fill_diagonal(D, np.inf)
nn = np.argsort(D, axis=1)[:, :5]
TR = pd.Timestamp("2025-01-01", tz="UTC")
rows, ratio = [], {}
for s in top:
    i = ids.index(s)
    x = g[s]
    nb = g.iloc[:, nn[i]].median(axis=1)
    u99 = x[x.index < TR].quantile(0.99)
    ev = x[x.index >= TR] >= u99
    r = (x / nb.where(nb > 5)).resample("QS").median()
    ratio[s] = r
    e = ev[ev].index
    # exceedance hours where any neighbour shows gust at or above its own TRAIN p95 within +-1 h
    nbp = g.iloc[:, nn[i]]
    p95 = nbp[nbp.index < TR].quantile(0.95)
    high = nbp.ge(p95, axis=1).any(axis=1).rolling(3, center=True, min_periods=1).max().astype(bool)
    mm = al[al.station_id.eq(s) & (al.timestamp_utc >= TR)]
    gf = (mm.wind_gust / mm.wind_mean.where(mm.wind_mean > 5)).median()
    rows.append({
        "station": s, "alt": c.alt.iloc[i], "km5": D[i][nn[i]].max(), "u99_train": u99, "exceed_hours_test": len(e),
        "expected_1pct": int(0.01 * x[x.index >= TR].notna().sum()), "share_neighbour_high": high.reindex(e).mean(),
        "hour_mode": int(pd.Series(e.hour).mode().iloc[0]) if len(e) else -1, "gust_factor": gf,
        "max_gust": x[x.index >= TR].max(), "max_neighbour_gust": nb[nb.index >= TR].max(),
    })
t = pd.DataFrame(rows)
print(t.round(2).to_string(index=False))
t.to_csv(N / "wind_top_stations.csv", index=False)
print("quarterly median gust ratio to neighbours")
q = pd.DataFrame(ratio).T.round(2)
q.columns = [f"{x.year}Q{x.quarter}" for x in q.columns]
print(q.to_string())
