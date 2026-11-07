"""Summarise the sensitivity battery and rerun the neighbour-failure experiment over K, q and tau.

Usage: sensitivity_summary.py SCRATCH_DIR OUT_DIR
"""
import json
import os
import sys
import warnings
from importlib import util
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220, "display.max_rows", 120, "display.max_columns", 30)
S, O = Path(sys.argv[1]), Path(sys.argv[2])
O.mkdir(parents=True, exist_ok=True)
ROOT = Path(__file__).resolve().parents[1]
RESULTS = os.environ.get('RESULTS_DIR', str(ROOT / 'results_v3'))
rows = []
for v in sorted(os.listdir(S / "var")):
    if not (v.startswith("B") or v == "V10_ini_pool14_q3_rh"):
        continue
    r = S / "var" / v / "results_v3"
    a = pd.read_csv(r / "measurement_fault_suite/measurement_fault_suite_summary.csv")
    a = a[a.detector == "Rule + spatial evidence"].set_index("fault_type")
    d = {"variant": v, "thr": a.pressure_spatial_threshold_hpa.iloc[0]}
    for k, n in (("pressure_offset", "off"), ("pressure_drift", "drift"), ("wind_spike_burst", "spike")):
        d[n + "_rec"] = a.loc[k, "recall"]
        d[n + "_prec"] = a.loc[k, "precision"]
    c = r / "operational_risk/operational_risk_components.csv"
    if c.exists():
        m = pd.read_csv(c).set_index("policy").missed_hard_fault_rate
        d["P4_missed"] = m["P4 pressure quarantine"]
        q = pd.read_csv(r / "spatial_quarantine_archive/archive_quarantine_counts.csv").set_index(["phase", "scope"])
        ret = pd.read_csv(r / "spatial_quarantine_archive/archive_retention.csv")
        d["u99_P5"] = ret[(ret.threshold == "u99") & (ret.policy == "P5 full spatial quarantine")].retention.iloc[0]
        d["press_train"] = q.loc[("train", "global"), "pressure_quarantine"]
        d["press_test"] = q.loc[("test", "global"), "pressure_quarantine"]
        d["wind_test"] = q.loc[("test", "global"), "wind_quarantine"]
    rows.append(d)
t = pd.DataFrame(rows)
t.to_csv(O / "battery_summary.csv", index=False)
print(t.round(3).to_string(index=False), flush=True)

# which sample stations drive the P5 retention loss (V10, dirty TEST)
b = S / "var/V10_ini_pool14_q3_rh/results_v3"
p0 = pd.read_parquet(b / "products/P0/data.parquet", columns=["station_id", "timestamp_utc", "stratum", "wind_gust"])
p0["timestamp_utc"] = pd.to_datetime(p0.timestamp_utc, utc=True)
f = pd.read_parquet(b / "spatial_quarantine_archive/spatial_quarantine_flags.parquet")
f["timestamp_utc"] = pd.to_datetime(f.timestamp_utc, utc=True)
x = p0.merge(f[["station_id", "timestamp_utc", "spatial_wind_quarantine"]], on=["station_id", "timestamp_utc"], how="left")
TR = pd.Timestamp("2025-01-01", tz="UTC")
u = x[x.timestamp_utc < TR].wind_gust.groupby(x.station_id).quantile(0.99)
pool99 = x[(x.timestamp_utc < TR) & x.stratum.eq("dirty")].wind_gust.quantile(0.99)
y = x[(x.timestamp_utc >= TR) & x.stratum.eq("dirty")]
print("dirty pooled TRAIN u99", pool99, flush=True)
e = y[y.wind_gust >= pool99]
g = e.groupby("station_id").agg(n=("wind_gust", "size"), flagged=("spatial_wind_quarantine", "sum")).sort_values("flagged", ascending=False)
g["ret"] = 1 - g.flagged / g.n
print(g.round(3).to_string(), "\nretention", 1 - g.flagged.sum() / g.n.sum(), flush=True)
g.to_csv(O / "wind_loss_by_sample_station.csv")

# neighbour-failure experiment over K, q, tau with aligned data and the filtered pool
sp = util.spec_from_file_location("nf", ROOT / "scripts/20_neighbor_failure_experiment.py")
nf = util.module_from_spec(sp)
sys.modules["nf"] = nf
sp.loader.exec_module(nf)
sample = sorted(pd.read_csv(RESULTS + "/selected_stations.csv").station_id)
al = pd.read_parquet(S / "pool/pool_all.parquet", columns=["station_id", "timestamp_utc", "pressure_hpa"])
al["timestamp_utc"] = pd.to_datetime(al.timestamp_utc, utc=True)
al = al.dropna()
grid = pd.date_range("2023-01-01", "2025-12-31 23:50", freq="10min", tz="UTC")


def aligned(keep):
    o = []
    for s, g_ in al[al.station_id.isin(keep)].groupby("station_id"):
        v = g_.set_index("timestamp_utc").pressure_hpa.sort_index()
        o.append(pd.DataFrame({"station_id": s, "timestamp_utc": grid, "pressure_hpa": v.reindex(grid, method="ffill", tolerance=pd.Timedelta("14min")).values}).dropna(subset=["pressure_hpa"]))
    return pd.concat(o, ignore_index=True)


cfg = [("K3", "pool_K3", 3, 8.0), ("K5", "pool", 3, 8.0), ("K7", "pool_K7", 3, 8.0), ("K10", "pool_K10", 3, 8.0),
       ("q2", "pool", 2, 8.0), ("q4", "pool", 4, 8.0), ("q5", "pool", 5, 8.0), ("tau5", "pool", 3, 5.0), ("tau12", "pool", 3, 12.0)]
base = {}
out = []
for name, pool, q, tau in cfg:
    ne = json.loads((S / pool / "neighbors.json").read_text())
    keep = set(sample) | {n for v in ne.values() for n in v}
    if pool not in base:
        base[pool] = aligned(keep)
    df = base[pool]
    nf.QUORUM, nf.RESIDUAL_FLOOR_HPA = q, tau
    nf.load_network = lambda *a, df=df: (df, pd.DataFrame(index=sorted(keep)))
    nf.neighbor_sets = lambda c_, ne=ne, **k: ne
    mh, mg, bm, sm = nf.run_network("AVAMET", None, None)
    mh, mg = pd.DataFrame(mh), pd.DataFrame(mg)
    for _, r in mg[mg.failure_type.isin(["dead_zero", "offset"])].iterrows():
        out.append({"cfg": name, "fail": r.failure_type, "d": r.d_failed, "naive": r.naive_false_quarantine_rate, "guard": r.guarded_false_quarantine_rate, "refhealth": r.refhealth_false_quarantine_rate})
    print(name, "thr", sm["threshold_hpa"], "m<=2", round(sm["frac_tolerance0_m_le2"], 5), "m>=q", round(float(mh[mh.m >= q].fraction.sum()), 4), flush=True)
n = pd.DataFrame(out)
n.to_csv(O / "neighbor_failure_sensitivity.csv", index=False)
print(n[n.d <= 2].pivot_table(index="cfg", columns=["fail", "d"], values=["naive", "refhealth"]).round(4).to_string(), flush=True)
print("SENSDONE")
