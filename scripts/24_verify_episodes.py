"""Are the long network pressure episodes sensor faults, or artefacts? Usage: verify_episodes.py SCRATCH_DIR NET_DIR"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 220, "display.max_rows", 120, "display.max_columns", 30)
S, N = Path(sys.argv[1]), Path(sys.argv[2])
ep = pd.read_csv(N / "episodes.csv", parse_dates=["start", "end"])
L = ep[ep.n >= 36].copy()
st = pd.read_csv(N / "station_p4_rates.csv")
print("long episodes", len(L), "stations", L.station_id.nunique(), "in sample", int(L.in_sample.sum()))
print("concordant neighbours (m>=4, MAD<2):", int(((L.m >= 4) & (L.neigh_mad < 2)).sum()))
print("rows in long episodes", int(L.n.sum()), "of all flagged", int(ep.n.sum()))
print("abs residual quantiles", L.med_resid.abs().quantile([.1, .5, .9, .99]).round(1).to_dict())
print("hours quantiles", L.hours.quantile([.1, .5, .9]).round(1).to_dict())
print("sign share negative", round(float((L.med_resid < 0).mean()), 3))

# common-cause: starts or ends shared by several stations within one hour
for col in ("start", "end"):
    x = L.assign(h=L[col].dt.floor("h")).groupby("h").station_id.nunique().sort_values(ascending=False)
    print(col, "hours shared by >=3 stations:", int((x >= 3).sum()))
    print(x.head(10).to_string())
shared = set()
for col in ("start", "end"):
    h = L[col].dt.floor("h")
    c = L.groupby(h).station_id.transform("nunique")
    shared |= set(L.index[c >= 3])
L["shared_edge"] = L.index.isin(shared)
print("episodes with a shared edge", int(L.shared_edge.sum()), "of", len(L))

# per-station monthly residual pattern from raw series
co = pd.read_csv(S / "pool/pool_coords.csv", index_col=0)
al = pd.read_parquet(S / "pool/pool_all.parquet", columns=["station_id", "timestamp_utc", "pressure_hpa"])
al["timestamp_utc"] = pd.to_datetime(al.timestamp_utc, utc=True)
al = al.dropna()
al["h"] = al.timestamp_utc.dt.floor("h")
w = al.pivot_table(index="h", columns="station_id", values="pressure_hpa", aggfunc="mean")
del al
med = w[w.index < pd.Timestamp("2025-01-01", tz="UTC")].median()
A = w - med
ids = list(w.columns)
c = co.loc[ids]
a, b = np.radians(c.lat.values), np.radians(c.lon.values)
D = 2 * 6371.0088 * np.arcsin(np.sqrt(np.sin((a[:, None] - a[None, :]) / 2) ** 2 + np.cos(a[:, None]) * np.cos(a[None, :]) * np.sin((b[:, None] - b[None, :]) / 2) ** 2))
np.fill_diagonal(D, np.inf)
nn = np.argsort(D, axis=1)[:, :5]
top = L.groupby("station_id").n.sum().sort_values(ascending=False).head(8).index
rows = {}
for s in top:
    i = ids.index(s)
    r = A[s] - A.iloc[:, nn[i]].median(axis=1)
    rows[f"{s} ({c.alt.iloc[i]:.0f} m)"] = r.resample("QS").median()
print("quarterly median residual hPa, top stations")
print(pd.DataFrame(rows).T.round(0).to_string())

# abruptness: residual jump across episode start, for episodes with data either side
jr = []
Rs = {}
for _, e in L.sort_values("n", ascending=False).head(200).iterrows():
    s = e.station_id
    if s not in Rs:
        i = ids.index(s)
        Rs[s] = A[s] - A.iloc[:, nn[i]].median(axis=1)
    r = Rs[s]
    t = e.start.tz_convert("UTC") if e.start.tzinfo else e.start.tz_localize("UTC")
    pre = r[t - pd.Timedelta("48h"):t - pd.Timedelta("1h")].median()
    post = r[t:t + pd.Timedelta("48h")].median()
    gap = r[t - pd.Timedelta("48h"):t - pd.Timedelta("1h")].notna().sum()
    jr.append({"pre": pre, "post": post, "jump": post - pre, "pre_hours": gap})
j = pd.DataFrame(jr)
print("top-200 episodes: median |jump|", round(float(j.jump.abs().median()), 1), "share |pre|<4", round(float((j.pre.abs() < 4).mean()), 3), "share with no data before", round(float((j.pre_hours < 6).mean()), 3))

# stations flagged: what share of each station's time, and independent check against sample selection
q = st.assign(frac=(st.flag_test + st.flag_train) / (st.valid_test + st.valid_train)).sort_values("frac", ascending=False)
print("stations with >1% rows flagged", int((q.frac > 0.01).sum()), ">10%", int((q.frac > 0.1).sum()), ">50%", int((q.frac > 0.5).sum()))
print("sample stations: max flagged share", round(float(q[q.in_sample].frac.max()), 5))

# wind: concentration of flags
w2 = pd.read_csv(N / "wind_rule_network.csv").sort_values("flag", ascending=False)
w2["ret"] = 1 - w2.flag / w2.n_u99
print("wind stations", len(w2), "top10 share of flags", round(float(w2.flag.head(10).sum() / w2.flag.sum()), 3), "top50", round(float(w2.flag.head(50).sum() / w2.flag.sum()), 3))
print("stations with retention >=0.95:", int((w2.ret >= 0.95).sum()), "of", len(w2), "| median station retention", round(float(w2.ret.median()), 3))
print("retention excluding top 10 stations", round(float(1 - w2.flag.iloc[10:].sum() / w2.n_u99.iloc[10:].sum()), 3))
print(w2.head(10)[["station_id", "km_nn5", "n_u99", "flag", "ret"]].round(3).to_string(index=False))
print("corr(km_nn5, ret) spearman", round(float(w2[["km_nn5", "ret"]].corr(method="spearman").iloc[0, 1]), 3))
