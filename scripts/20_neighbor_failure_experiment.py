"""Controlled neighbour-failure experiment: false isolation in spatial QC actions.

Thesis. A spatial-redundancy quarantine action decides whether a target sensor
is faulty by comparing it with the median of its neighbours. This is a median
voter, and a median voter over m references tolerates only floor((m-1)/2) failed
references. When a nearby or operationally coupled neighbour dies (reads ~0 hPa) its value contaminates
the reference; once the dead fraction reaches the voter's tolerance the action
falsely quarantines a *healthy* target -- a false isolation that propagates a
reliability cost (spurious maintenance / loss of a working sensor).

We quantify this on two real networks by taking genuinely healthy target/time
points, injecting d failed neighbours, and measuring the false-quarantine rate of
the NAIVE action (raw neighbour median), a hard-invalid GUARDED action, and a
reference-health guarded action that adds a leave-one-neighbour-out MAD screen.
Dead/sentinel failures are the primary experiment; in-range neighbour offsets
test whether this additional diagnostic recovers any residual hazard.
We also report the empirical distribution of the effective neighbour count m,
which is what places sparse networks in the fragile regime.

Outputs: CSVs + a two-panel figure + a LaTeX summary table.
"""
from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

TRAIN_END = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
TEST_END = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
K_NEIGHBORS = 5
SUPPORT_MIN_NEIGHBORS = 2     # minimum references to attempt any action
QUORUM = 3                    # guarded action requires >=3 *healthy* references
REF_HEALTH_MIN_PEERS = 3      # peers needed for leave-one-neighbour-out screening
REF_HEALTH_MAD_Z = 3.0
PRESSURE_FLOOR_HPA = 500.0    # below this a pressure reading is physically dead/sentinel
PRESSURE_CEIL_HPA = 1100.0
RESIDUAL_FLOOR_HPA = 8.0
KILL_VALUE_HPA = 0.0          # realistic dead-sensor reading observed in AVAMET
OFFSET_HPA = 15.0             # in-range neighbour bias (> RESIDUAL_FLOOR_HPA); invisible to hard-invalid guard
MAX_D = 3
RNG_SEED = 20260628
FAILURE_TYPES = ("dead_zero", "offset", "stuck_plausible")
N_COMBO_REPLICATES = 1


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def load_network(name, product_path, raw_parquet):
    df = pd.read_parquet(product_path, columns=None)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df[df["timestamp_utc"] < TEST_END]
    if "lat" in df.columns and "lon" in df.columns:
        coords = df.groupby("station_id")[["lat", "lon"]].median()
    else:
        import duckdb
        sids = sorted(df["station_id"].unique())
        q = ",".join("'" + s + "'" for s in sids)
        coords = (
            duckdb.connect()
            .execute(
                f"SELECT station_id, median(lat) lat, median(lon) lon "
                f"FROM read_parquet('{Path(raw_parquet).as_posix()}') "
                f"WHERE station_id IN ({q}) GROUP BY station_id"
            )
            .df()
            .set_index("station_id")
        )
    return df[["station_id", "timestamp_utc", "pressure_hpa"]], coords


def neighbor_sets(coords, k=K_NEIGHBORS):
    out = {}
    for s in coords.index:
        d = sorted(
            (haversine_km(coords.loc[s, "lat"], coords.loc[s, "lon"], coords.loc[o, "lat"], coords.loc[o, "lon"]), o)
            for o in coords.index
            if o != s
        )
        out[s] = [o for _, o in d[:k]]
    return out


def valid_pressure(series):
    """Physically valid pressure (dead/sentinel zeros and out-of-range -> NaN)."""
    return series.where((series > PRESSURE_FLOOR_HPA) & (series < PRESSURE_CEIL_HPA))


def nanmedian_no_warn(values, axis=1):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(values, axis=axis)


def reference_health_filter(ref, threshold):
    """Drop in-range reference outliers using neighbour-only robust evidence."""
    ref = np.asarray(ref, dtype=float)
    out = ref.copy()
    finite = np.isfinite(ref)
    for j in range(ref.shape[1]):
        peers = ref.copy()
        peers[:, j] = np.nan
        peer_count = np.isfinite(peers).sum(axis=1)
        peer_med = nanmedian_no_warn(peers, axis=1)
        peer_dev = np.abs(peers - peer_med[:, None])
        peer_mad = nanmedian_no_warn(peer_dev, axis=1)
        cutoff = np.maximum(threshold, REF_HEALTH_MAD_Z * 1.4826 * peer_mad)
        bad = finite[:, j] & (peer_count >= REF_HEALTH_MIN_PEERS) & (np.abs(ref[:, j] - peer_med) > cutoff)
        out[bad, j] = np.nan
    return out


def run_network(name, product_path, raw_parquet):
    rng = np.random.default_rng(RNG_SEED + (0 if name == "AVAMET" else 1000))
    df, coords = load_network(name, product_path, raw_parquet)
    neighbors = neighbor_sets(coords)

    df = df.assign(pressure_valid=valid_pressure(df["pressure_hpa"]))
    train_median = df[df["timestamp_utc"] < TRAIN_END].groupby("station_id")["pressure_valid"].median()

    wide = df.pivot(index="timestamp_utc", columns="station_id", values="pressure_valid")
    anom = wide.subtract(train_median, axis="columns")

    # Residual threshold from TRAIN guarded references (consistent with fixed pipeline).
    train_anom = anom[anom.index < TRAIN_END]
    resids = []
    for s, ns in neighbors.items():
        vn = [n for n in ns if n in train_anom.columns]
        if len(vn) < SUPPORT_MIN_NEIGHBORS or s not in train_anom.columns:
            continue
        t = train_anom[s]
        ne = train_anom[vn]
        ok = t.notna() & (ne.notna().sum(axis=1) >= SUPPORT_MIN_NEIGHBORS)
        if ok.any():
            r = (t[ok] - ne[ok].median(axis=1)).abs().to_numpy()
            resids.append(r[np.isfinite(r)])
    threshold = max(RESIDUAL_FLOOR_HPA, float(np.quantile(np.concatenate(resids), 0.995))) if resids else RESIDUAL_FLOOR_HPA

    test_anom = anom[anom.index >= TRAIN_END]

    by_m_rows, marginal_rows, m_hist = [], [], []
    # accumulate marginal counts across stations
    marg_naive = {(ft, d): [0, 0] for ft in FAILURE_TYPES for d in range(1, MAX_D + 1)}
    marg_guard = {(ft, d): [0, 0] for ft in FAILURE_TYPES for d in range(1, MAX_D + 1)}
    marg_refhealth = {(ft, d): [0, 0] for ft in FAILURE_TYPES for d in range(1, MAX_D + 1)}
    bym = {}  # (failure_type,m,d) -> [naive_flag, guard_flag, refhealth_flag, total]
    m_counter = {}

    for s, ns in neighbors.items():
        vn = [n for n in ns if n in test_anom.columns]
        if len(vn) < SUPPORT_MIN_NEIGHBORS or s not in test_anom.columns:
            continue
        target = test_anom[s].to_numpy()                       # [T]
        ne = test_anom[vn].to_numpy()                          # [T, m<=5]
        kill_anom = np.array([KILL_VALUE_HPA - float(train_median.get(n, np.nan)) for n in vn])  # [m]

        alive = np.isfinite(ne)                                # [T,m]
        m_t = alive.sum(axis=1)                                # [T]
        base_med = nanmedian_no_warn(np.where(alive, ne, np.nan), axis=1)
        with np.errstate(invalid="ignore"):
            base_resid = np.abs(target - base_med)
        healthy = np.isfinite(target) & (m_t >= SUPPORT_MIN_NEIGHBORS) & (base_resid < threshold)

        # effective neighbour-count distribution over healthy target-time
        for mm in np.unique(m_t[healthy]):
            m_counter[int(mm)] = m_counter.get(int(mm), 0) + int(((m_t == mm) & healthy).sum())

        for d in range(1, MAX_D + 1):
            if len(vn) < d:
                continue
            for _rep in range(N_COMBO_REPLICATES):
                random_scores = rng.random(alive.shape)
                random_scores = np.where(alive, random_scores, np.inf)
                chosen = np.argpartition(random_scores, kth=d - 1, axis=1)[:, :d]
                kill_mask = np.zeros_like(alive, dtype=bool)
                rows = np.arange(alive.shape[0])[:, None]
                kill_mask[rows, chosen] = np.isfinite(np.take_along_axis(random_scores, chosen, axis=1))
                sel = healthy & (kill_mask.sum(axis=1) == d)
                if not sel.any():
                    continue

                alive_anom = np.where(alive, ne, np.nan)
                for failure_type in FAILURE_TYPES:
                    if failure_type == "dead_zero":
                        # out-of-range sentinel; caught by the hard-invalid guard
                        injected_vals = np.broadcast_to(kill_anom, ne.shape)
                        injected_invalid = True
                    elif failure_type == "offset":
                        # in-range bias above the residual threshold; invisible to the guard
                        injected_vals = alive_anom + OFFSET_HPA
                        injected_invalid = False
                    elif failure_type == "stuck_plausible":
                        # frozen at a plausible in-range constant (TRAIN median => ~0 anomaly)
                        injected_vals = np.zeros_like(ne)
                        injected_invalid = False
                    else:
                        raise ValueError(failure_type)

                    # NAIVE: contaminated values enter the reference unchanged.
                    naive_ref = np.where(kill_mask, injected_vals, alive_anom)
                    naive_med = nanmedian_no_warn(naive_ref, axis=1)
                    naive_flag = (np.abs(target - naive_med) > threshold) & sel

                    # GUARDED: only HARD-INVALID injected values are excluded; in-range
                    # reference faults are invisible to the guard and remain in the vote.
                    if injected_invalid:
                        guard_ref = np.where(kill_mask, np.nan, alive_anom)
                    else:
                        guard_ref = naive_ref
                    remaining = np.isfinite(guard_ref).sum(axis=1)
                    guard_med = nanmedian_no_warn(guard_ref, axis=1)
                    with np.errstate(invalid="ignore"):
                        guard_flag = (remaining >= QUORUM) & (np.abs(target - guard_med) > threshold) & sel

                    refhealth_ref = reference_health_filter(guard_ref, threshold)
                    refhealth_remaining = np.isfinite(refhealth_ref).sum(axis=1)
                    refhealth_med = nanmedian_no_warn(refhealth_ref, axis=1)
                    with np.errstate(invalid="ignore"):
                        refhealth_flag = (
                            (refhealth_remaining >= QUORUM)
                            & (np.abs(target - refhealth_med) > threshold)
                            & sel
                        )

                    marg_naive[(failure_type, d)][0] += int(naive_flag.sum())
                    marg_naive[(failure_type, d)][1] += int(sel.sum())
                    marg_guard[(failure_type, d)][0] += int(guard_flag.sum())
                    marg_guard[(failure_type, d)][1] += int(sel.sum())
                    marg_refhealth[(failure_type, d)][0] += int(refhealth_flag.sum())
                    marg_refhealth[(failure_type, d)][1] += int(sel.sum())
                    for mm in np.unique(m_t[sel]):
                        cell = (failure_type, int(mm), d)
                        msel = sel & (m_t == mm)
                        acc = bym.get(cell, [0, 0, 0, 0])
                        acc[0] += int((naive_flag & msel).sum())
                        acc[1] += int((guard_flag & msel).sum())
                        acc[2] += int((refhealth_flag & msel).sum())
                        acc[3] += int(msel.sum())
                        bym[cell] = acc

    total_healthy = sum(m_counter.values())
    for mm in sorted(m_counter):
        m_hist.append({"network": name, "m": mm, "count": m_counter[mm], "fraction": m_counter[mm] / total_healthy})
    for failure_type in FAILURE_TYPES:
        for d in range(1, MAX_D + 1):
            n_f, n_t = marg_naive[(failure_type, d)]
            g_f, g_t = marg_guard[(failure_type, d)]
            h_f, h_t = marg_refhealth[(failure_type, d)]
            marginal_rows.append({
                "network": name, "failure_type": failure_type, "d_failed": d,
                "n_trials": n_t,
                "naive_false_quarantine_rate": (n_f / n_t) if n_t else np.nan,
                "guarded_false_quarantine_rate": (g_f / g_t) if g_t else np.nan,
                "refhealth_false_quarantine_rate": (h_f / h_t) if h_t else np.nan,
            })
    for (failure_type, mm, d), acc in sorted(bym.items()):
        by_m_rows.append({
            "network": name, "failure_type": failure_type, "m_neighbors": mm, "d_failed": d, "n_trials": acc[3],
            "naive_false_quarantine_rate": acc[0] / acc[3] if acc[3] else np.nan,
            "guarded_false_quarantine_rate": acc[1] / acc[3] if acc[3] else np.nan,
            "refhealth_false_quarantine_rate": acc[2] / acc[3] if acc[3] else np.nan,
        })

    tol0 = sum(c for m, c in m_counter.items() if m <= 2) / total_healthy if total_healthy else np.nan
    summary = {
        "network": name, "threshold_hpa": threshold, "n_healthy_points": total_healthy,
        "median_m": float(np.median(np.repeat(list(m_counter.keys()), list(m_counter.values())))) if total_healthy else np.nan,
        "frac_tolerance0_m_le2": tol0,
    }
    return pd.DataFrame(m_hist), pd.DataFrame(marginal_rows), pd.DataFrame(by_m_rows), summary


def make_figure(m_hist, marginal, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nets = list(m_hist["network"].unique())
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    # Panel A: marginal false-quarantine vs d, by reference-failure mode, on the
    # sparse network (AVAMET). The hard-invalid guard collapses sentinel failures;
    # the reference-health screen tests recoverability of in-range offsets.
    modes = [("dead_zero", "sentinel", "C0"), ("offset", "offset $+15$ hPa", "C3"),
             ("stuck_plausible", "stuck (plausible)", "C2")]
    net0 = "AVAMET"
    for ft, label, color in modes:
        sub = marginal[(marginal["network"] == net0) & (marginal["failure_type"] == ft)].sort_values("d_failed")
        ax[0].plot(sub["d_failed"], sub["naive_false_quarantine_rate"], "o-", color=color, label=f"{label} (naive)")
        ax[0].plot(sub["d_failed"], sub["guarded_false_quarantine_rate"], "s--", color=color, label=f"{label} (guarded)")
        ax[0].plot(sub["d_failed"], sub["refhealth_false_quarantine_rate"], "^:", color=color, label=f"{label} (ref-health)")
    ax[0].set_xlabel("failed neighbours injected $d$")
    ax[0].set_ylabel("false-quarantine rate (healthy targets)")
    ax[0].set_title(f"(a) False isolation by failure mode ({net0}, sparse)")
    ax[0].set_xticks(sorted(marginal["d_failed"].unique()))
    ax[0].legend(fontsize=6.5, ncol=1)
    ax[0].grid(alpha=0.3)
    # Panel B: distribution of effective neighbour count m
    width = 0.38
    ms = sorted(m_hist["m"].unique())
    for i, net in enumerate(nets):
        sub = m_hist[m_hist["network"] == net].set_index("m").reindex(ms).fillna(0)
        ax[1].bar(np.array(ms) + (i - 0.5) * width, sub["fraction"], width=width, label=net)
    ax[1].set_xlabel("effective neighbours available $m$")
    ax[1].set_ylabel("fraction of healthy station-time")
    ax[1].set_title("(b) Redundancy depth by network")
    ax[1].set_xticks(ms)
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    fig.savefig(str(out_png).replace(".png", ".pdf"))
    plt.close(fig)


def write_table(marginal, summary_df, out_tex):
    nets = list(marginal["network"].unique())
    mode_labels = [
        ("dead_zero", "Sentinel (dead/out-of-range)"),
        ("offset", "In-range offset ($+15$\\,hPa)"),
        ("stuck_plausible", "Stuck (plausible constant)"),
    ]
    lines = ["\\begin{tabular}{ll" + "rrr" * len(nets) + "}", "\\toprule"]
    lines.append("& & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{n}}}" for n in nets) + " \\\\")
    lines.append("Reference-failure mode & $d$ & " + " & ".join("Naive & Guard v1 & Ref-health" for _ in nets) + " \\\\")
    lines.append("\\midrule")
    for ft, label in mode_labels:
        sub = marginal[marginal["failure_type"] == ft]
        ds = sorted(sub["d_failed"].unique())
        for i, d in enumerate(ds):
            cells = []
            for n in nets:
                r = sub[(sub["network"] == n) & (sub["d_failed"] == d)].iloc[0]
                cells.append(
                    f"{r['naive_false_quarantine_rate']:.3f} & "
                    f"{r['guarded_false_quarantine_rate']:.3f} & "
                    f"{r['refhealth_false_quarantine_rate']:.3f}"
                )
            name_cell = label if i == 0 else ""
            lines.append(f"{name_cell} & {d} & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    Path(out_tex).write_text("\n".join(lines), encoding="utf-8")


def main():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=root / "runs_additional" / "neighbor_failure")
    ap.add_argument("--article-dir", type=Path, default=root / "article")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.article_dir.mkdir(parents=True, exist_ok=True)

    networks = [
        ("AVAMET", root / "runs_additional" / "BASELINE" / "results_v3" / "products" / "P0" / "data.parquet", root.parent / "avamet_all.parquet"),
        ("DWD", root / "runs_additional" / "DWD" / "results_v3" / "products" / "P0" / "data.parquet", None),
    ]
    m_hist_all, marg_all, bym_all, summaries = [], [], [], []
    for name, prod, raw in networks:
        print(f"running {name} ...")
        mh, mg, bm, summ = run_network(name, prod, raw)
        m_hist_all.append(mh); marg_all.append(mg); bym_all.append(bm); summaries.append(summ)
        print(f"  threshold={summ['threshold_hpa']:.2f} hPa, healthy points={summ['n_healthy_points']:,}, "
              f"median m={summ['median_m']:.1f}, tol-0 (m<=2) frac={summ['frac_tolerance0_m_le2']:.3f}")

    m_hist = pd.concat(m_hist_all, ignore_index=True)
    marginal = pd.concat(marg_all, ignore_index=True)
    by_m = pd.concat(bym_all, ignore_index=True)
    summary_df = pd.DataFrame(summaries)

    m_hist.to_csv(args.out_dir / "m_distribution.csv", index=False)
    marginal.to_csv(args.out_dir / "neighbor_failure_marginal.csv", index=False)
    by_m.to_csv(args.out_dir / "neighbor_failure_by_m.csv", index=False)
    summary_df.to_csv(args.out_dir / "neighbor_failure_summary.csv", index=False)
    make_figure(m_hist, marginal, args.article_dir / "F_neighbor_failure.png")
    write_table(marginal, summary_df, args.article_dir / "tab_neighbor_failure.tex")

    print("\n=== MARGINAL false-quarantine rate ===")
    print(marginal.to_string(index=False))
    print("\n=== summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nWrote outputs to {args.out_dir} and figure/table to {args.article_dir}")


if __name__ == "__main__":
    main()
