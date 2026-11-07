"""Network-wide analyses for the EMS revision.

Decorrelation by distance, spatial validation on stations outside the 30-station
sample, real-fault search, and the wind rule over the whole network.
Usage: network_analysis.py SCRATCH_DIR OUT_DIR
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")
S = Path(sys.argv[1])
O = Path(sys.argv[2])
O.mkdir(parents=True, exist_ok=True)
ROOT = Path(__file__).resolve().parents[1]
RESULTS = __import__('os').environ.get('RESULTS_DIR', str(ROOT / 'results_v3'))
TAU, Q, K = 8.0, 3, 5
TOL = pd.Timedelta("14min")
TR = pd.Timestamp("2025-01-01", tz="UTC")
SEED = 20261106

co = pd.read_csv(S / "pool/pool_coords.csv", index_col=0)
sample = sorted(pd.read_csv(RESULTS + "/selected_stations.csv").station_id)
al = pd.read_parquet(S / "pool/pool_all.parquet")
al["timestamp_utc"] = pd.to_datetime(al.timestamp_utc, utc=True)
ids = sorted(set(al.station_id) & set(co.index))
co = co.loc[ids]


def dist_matrix(c):
    a, b = np.radians(c.lat.values), np.radians(c.lon.values)
    h = np.sin((a[:, None] - a[None, :]) / 2) ** 2 + np.cos(a[:, None]) * np.cos(a[None, :]) * np.sin((b[:, None] - b[None, :]) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(h))


D = dist_matrix(co)
np.fill_diagonal(D, np.inf)
ix = {s: i for i, s in enumerate(ids)}
grid = pd.date_range("2023-01-01", "2025-12-31 23:50", freq="10min", tz="UTC")


def gridded(col):
    A = np.full((len(grid), len(ids)), np.nan, np.float32)
    for s, g in al.groupby("station_id"):
        if s not in ix:
            continue
        v = g.dropna(subset=[col]).set_index("timestamp_utc")[col].sort_index()
        if len(v):
            A[:, ix[s]] = v.reindex(grid, method="ffill", tolerance=TOL).values
    return A


P, G, M = gridded("pressure_hpa"), gridded("wind_gust"), gridded("wind_mean")
del al
print("gridded", P.shape, flush=True)
tr = np.asarray(grid < TR)
te = ~tr
A = P - np.nanmedian(P[tr], axis=0)

# 1. neighbour distances
nn = np.argsort(D, axis=1)[:, :K]
dk = np.sort(D, axis=1)[:, :K]
pd.DataFrame({
    "station_id": ids, "in_sample": [s in sample for s in ids], "km_nn1": dk[:, 0], "km_nn5": dk[:, 4],
    "km_mean5": dk.mean(1), "alt": co.alt.values,
}).to_csv(O / "station_neighbour_distance.csv", index=False)

# 2. decorrelation on TRAIN, hourly
bins = [0, 5, 10, 25, 50, 100, 200, 400]
iu = np.triu_indices(len(ids), 1)
pairs = pd.DataFrame({"i": iu[0], "j": iu[1], "km": D[iu], "dalt": np.abs(co.alt.values[iu[0]] - co.alt.values[iu[1]])})
pairs = pairs[pairs.km < 400].reset_index(drop=True)


def hourly(X, how):
    d = pd.DataFrame(X[tr], index=grid[tr], columns=ids).resample("1h")
    return d.max() if how == "max" else d.mean()


for nm, X, how in (("pressure", A, "mean"), ("gust", G, "max"), ("wind_mean", M, "mean")):
    h = hourly(X, how)
    des = h - h.groupby([h.index.year, h.index.month]).transform("mean")
    for lab, hh in (("anomaly", h), ("deseason", des)):
        C = hh.corr(min_periods=1000).values
        pairs[f"r_{nm}_{lab}"] = C[pairs.i.values, pairs.j.values]
        print(nm, lab, flush=True)
    if nm == "pressure":
        rng = np.random.default_rng(SEED)
        pairs["p99"] = np.nan
        pairs["f8"] = np.nan
        V = h.values
        for b0, b1 in zip(bins[:-1], bins[1:]):
            q = pairs.index[(pairs.km >= b0) & (pairs.km < b1)]
            for t in rng.choice(q, min(400, len(q)), replace=False):
                d = np.abs(V[:, pairs.i[t]] - V[:, pairs.j[t]])
                d = d[np.isfinite(d)]
                if len(d) > 1000:
                    pairs.loc[t, "p99"] = np.quantile(d, 0.99)
                    pairs.loc[t, "f8"] = (d > TAU).mean()
pairs["bin"] = pd.cut(pairs.km, bins)
pairs.drop(columns=["i", "j"]).astype({"bin": str}).to_parquet(O / "pairs.parquet")
cols = [c for c in pairs.columns if c.startswith("r_") or c in ("p99", "f8")]
t = pairs.groupby("bin", observed=True)[cols].median()
t.insert(0, "n", pairs.groupby("bin", observed=True).size())
t.round(4).to_csv(O / "decorrelation_by_distance.csv")
print(t.round(3).to_string(), flush=True)
rows = []
for c in [x for x in pairs if x.startswith("r_")]:
    p = pairs.dropna(subset=[c])
    pp, _ = curve_fit(lambda d, a, L: a * np.exp(-d / L), p.km, p[c], p0=(0.9, 200), maxfev=5000)
    rows.append({"series": c, "r0": pp[0], "efold_km": pp[1]})
e = pd.DataFrame(rows)
e.to_csv(O / "decorrelation_efold.csv", index=False)
print(e.round(3).to_string(index=False), flush=True)
p = pairs[(pairs.km >= 10) & (pairs.km < 25)].copy()
p["alt"] = pd.cut(p.dalt, [-1, 100, 300, 3000])
t = p.groupby("alt", observed=True).agg(n=("km", "size"), r_gust=("r_gust_deseason", "median"), r_press=("r_pressure_deseason", "median"))
t.to_csv(O / "decorrelation_altitude_10_25km.csv")
print(t.round(3).to_string(), flush=True)

# 3. guarded reference for every station
sys.path.insert(0, str(ROOT / 'scripts'))
import neighbour_reference as nb
res = np.full(A.shape, np.nan, np.float32)
mcnt = np.zeros(A.shape, np.int8)
nmad = np.full(A.shape, np.nan, np.float32)
for i in range(len(ids)):
    ref = nb.reference_health_filter(A[:, nn[i]].astype(float), TAU)
    m = np.isfinite(ref).sum(1)
    med = nb._nanmedian(ref)
    r = A[:, i] - med
    r[m < Q] = np.nan
    res[:, i] = r
    mcnt[:, i] = m
    nmad[:, i] = nb._nanmedian(np.abs(ref - med[:, None]))
print("residuals done", flush=True)
ok = np.isfinite(res)
fl = ok & (np.abs(res) > TAU)
cover = [(mcnt[np.isfinite(A[:, i]), i] >= Q).mean() if np.isfinite(A[:, i]).any() else np.nan for i in range(len(ids))]
st = pd.DataFrame({
    "station_id": ids, "in_sample": [s in sample for s in ids], "km_nn5": dk[:, 4],
    "valid_test": ok[te].sum(0), "flag_test": fl[te].sum(0), "valid_train": ok[tr].sum(0), "flag_train": fl[tr].sum(0), "cover_q3": cover,
})
st["rate_test"] = st.flag_test / st.valid_test
st.to_csv(O / "station_p4_rates.csv", index=False)
print("flag rate all", fl.sum() / ok.sum(), "test", fl[te].sum() / ok[te].sum(), "stations no TEST flag", int((st.flag_test == 0).sum()), "of", len(st), "cover median", st.cover_q3.median(), flush=True)

ep = []
for i, s in enumerate(ids):
    t = np.flatnonzero(fl[:, i])
    if not len(t):
        continue
    for seg in np.split(t, np.flatnonzero(np.diff(t) > 36) + 1):
        ep.append({
            "station_id": s, "in_sample": s in sample, "start": grid[seg[0]], "end": grid[seg[-1]], "n": len(seg),
            "hours": (seg[-1] - seg[0]) / 6, "med_resid": float(np.median(res[seg, i])), "m": float(np.median(mcnt[seg, i])),
            "neigh_mad": float(np.nanmedian(nmad[seg, i])), "own_anom": float(np.nanmedian(A[seg, i])),
        })
ep = pd.DataFrame(ep).sort_values("n", ascending=False)
ep.to_csv(O / "episodes.csv", index=False)
print("episodes", len(ep), ">=36 rows", int((ep.n >= 36).sum()), "stations", ep[ep.n >= 36].station_id.nunique(), flush=True)
print(ep.head(25).round(2).to_string(index=False), flush=True)

# 4. spatial validation: injected faults on stations outside the sample
rng = np.random.default_rng(SEED)
hold = [i for i, s in enumerate(ids) if s not in sample]
rows = []
for mag in (5, 8, 10, 12, 15, 20):
    for kind, L, E in (("offset", 36, 2), ("drift", 72, 1)):
        for i in hold:
            v = np.flatnonzero(te & ok[:, i])
            if len(v) < 1000:
                continue
            for _ in range(E):
                s0 = rng.choice(v[:-L])
                w = np.arange(s0, min(s0 + L, len(grid)))
                add = mag if kind == "offset" else np.linspace(0, mag, len(w))
                g = np.isfinite(res[w, i])
                hit = np.abs(res[w, i] + add)[g] > TAU
                rows.append({"kind": kind, "mag": mag, "station": ids[i], "km_nn5": dk[i, 4], "n": int(g.sum()), "hit": int(hit.sum()), "missing": int((~g).sum()), "detected": bool(hit.any())})
iv = pd.DataFrame(rows)
iv.to_csv(O / "heldout_injection.csv", index=False)
iv["dq"] = pd.qcut(iv.km_nn5, 4, labels=["Q1 densest", "Q2", "Q3", "Q4 sparsest"])
t = iv.groupby(["kind", "mag"]).agg(events=("n", "size"), rows=("n", "sum"), hits=("hit", "sum"), missing=("missing", "sum"), event_detect=("detected", "mean"))
t["row_recall"] = t.hits / (t.rows + t.missing)
t.to_csv(O / "heldout_recall.csv")
print(t.round(3).to_string(), flush=True)
t = iv[iv.mag == 20].groupby(["kind", "dq"], observed=True).agg(rows=("n", "sum"), hits=("hit", "sum"), missing=("missing", "sum"), km=("km_nn5", "median"))
t["row_recall"] = t.hits / (t.rows + t.missing)
t.to_csv(O / "heldout_recall_by_density.csv")
print(t.round(3).to_string(), flush=True)

# 5. wind rule over the whole network
u90 = np.nanquantile(G[tr], 0.9, axis=0)
u99 = np.nanquantile(G[tr], 0.99, axis=0)
dq = np.clip(np.nanquantile((G - M)[tr], 0.995, axis=0), 30, None)
rows = []
for i, s in enumerate(ids):
    ev = te & (G[:, i] >= u99[i])
    if not ev.any():
        continue
    sup = (G[:, nn[i]] >= u90[nn[i]]).sum(1)
    f = ev & ((G[:, i] - M[:, i]) >= dq[i]) & (sup < 2)
    rows.append({"station_id": s, "in_sample": s in sample, "km_nn5": dk[i, 4], "n_u99": int(ev.sum()), "flag": int(f.sum())})
w = pd.DataFrame(rows)
w.to_csv(O / "wind_rule_network.csv", index=False)
print("u99 retention network", 1 - w.flag.sum() / w.n_u99.sum(), "sample", 1 - w[w.in_sample].flag.sum() / w[w.in_sample].n_u99.sum(), "held-out", 1 - w[~w.in_sample].flag.sum() / w[~w.in_sample].n_u99.sum(), flush=True)
w["dq"] = pd.qcut(w.km_nn5, 4, labels=["Q1 densest", "Q2", "Q3", "Q4 sparsest"])
t = w.groupby("dq", observed=True).agg(n=("n_u99", "sum"), f=("flag", "sum"), km=("km_nn5", "median"))
t["ret"] = 1 - t.f / t.n
t.to_csv(O / "wind_retention_by_density.csv")
print(t.round(3).to_string(), flush=True)
print("NETDONE")
