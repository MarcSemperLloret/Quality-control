from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


TRAIN_START = pd.Timestamp("2023-01-01 00:00:00", tz="UTC")
TRAIN_END = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
TEST_START = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
TEST_END = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    paper2_dir = repo_root
    parser = argparse.ArgumentParser(
        description=(
            "External sensor-domain transfer check using DWD 1-minute precipitation. "
            "This is a protocol-portability diagnostic, not a transfer of AVAMET P3 thresholds."
        )
    )
    parser.add_argument(
        "--dwd-selected-stations",
        type=Path,
        default=repo_root / "data" / "dwd_rr_1min" / "dwd_external_selected_stations.csv",
    )
    parser.add_argument(
        "--dwd-normalized-dir",
        type=Path,
        default=repo_root / "data" / "dwd_rr_1min" / "normalized",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=paper2_dir / "runs_additional" / "BASELINE" / "results_v3" / "dwd_external_transfer",
    )
    parser.add_argument("--article-dir", type=Path, default=paper2_dir / "article")
    parser.add_argument("--n-stations", type=int, default=30)
    parser.add_argument("--min-coverage", type=float, default=0.95)
    parser.add_argument("--mcar-seed", type=int, default=20260519)
    return parser.parse_args()


def wilson_ci(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return (np.nan, np.nan)
    p = successes / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denom
    half = z * math.sqrt((p * (1.0 - p) / total) + (z * z / (4.0 * total * total))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def build_minute_arrays(intervals: pd.DataFrame, t0_sec: int, t1_sec: int) -> tuple[np.ndarray, np.ndarray]:
    n_min = int((t1_sec - t0_sec) // 60)
    mass = np.zeros(n_min, dtype=np.float64)
    known = np.zeros(n_min, dtype=bool)
    if intervals.empty:
        return mass, known

    s = intervals["start_sec"].to_numpy(dtype=np.int64)
    e = intervals["end_sec"].to_numpy(dtype=np.int64)
    p = intervals["precip_mm"].to_numpy(dtype=np.float64)
    mask = np.isfinite(p) & (e > s)
    if not np.any(mask):
        return mass, known

    s = np.clip(s[mask], t0_sec, t1_sec)
    e = np.clip(e[mask], t0_sec, t1_sec)
    p = p[mask]
    mask2 = e > s
    if not np.any(mask2):
        return mass, known
    s = s[mask2]
    e = e[mask2]
    p = p[mask2]

    i0 = ((s - t0_sec) // 60).astype(np.int64)
    i1 = ((e - t0_sec) // 60).astype(np.int64)

    for a, b, mm in zip(i0, i1, p):
        aa = int(a)
        bb = int(b)
        if bb <= aa:
            continue
        dur = bb - aa
        known[aa:bb] = True
        if mm != 0:
            mass[aa:bb] += float(mm) / dur

    return mass, known


def hourly_from_station(station_id: str, parquet_path: Path) -> pd.DataFrame:
    t0 = int(TRAIN_START.timestamp())
    t1 = int(TEST_END.timestamp())
    intervals = pd.read_parquet(parquet_path, columns=["st", "start_sec", "end_sec", "precip_mm"])
    intervals = intervals[(intervals["end_sec"] > t0) & (intervals["start_sec"] < t1)].copy()
    mass_min, known_min = build_minute_arrays(intervals, t0, t1)
    n_hours = int(len(mass_min) // 60)
    mass_min = mass_min[: n_hours * 60]
    known_min = known_min[: n_hours * 60]

    precip_h = mass_min.reshape(n_hours, 60).sum(axis=1)
    coverage = known_min.reshape(n_hours, 60).sum(axis=1).astype(np.float64) / 60.0
    peak_mm_h = mass_min.reshape(n_hours, 60).max(axis=1) * 60.0
    timestamps = pd.date_range(TRAIN_START, periods=n_hours, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "station_id": station_id,
            "timestamp_utc": timestamps,
            "precip_h_mm": precip_h,
            "coverage": coverage,
            "peak_1min_mm_h": peak_mm_h,
        }
    )


def load_dwd_hourly(selected_path: Path, normalized_dir: Path, n_stations: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = pd.read_csv(selected_path, dtype={"st": str})
    selected["st"] = selected["st"].astype(str).str.zfill(5)
    selected = selected.head(n_stations).copy()

    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for st in selected["st"]:
        path = normalized_dir / f"{st}.parquet"
        if not path.exists():
            missing.append(st)
            continue
        frames.append(hourly_from_station(st, path))
    if missing:
        raise FileNotFoundError(f"Missing normalized DWD station parquet files: {missing}")
    if not frames:
        raise RuntimeError("No DWD stations loaded.")
    return pd.concat(frames, ignore_index=True), selected


def mnar_spread_wet(values: pd.Series, removed: pd.Series) -> float:
    domain = values.notna() & removed.notna() & (values > 0)
    if int(domain.sum()) < 20:
        return float("nan")
    vals = values.loc[domain]
    rem = removed.loc[domain].astype(float)
    try:
        bins = pd.qcut(vals.rank(method="first"), q=10, labels=False, duplicates="drop")
    except ValueError:
        return float("nan")
    rates = rem.groupby(bins).mean()
    if rates.empty:
        return float("nan")
    return float(rates.max() - rates.min())


def policy_metrics(hourly: pd.DataFrame, min_coverage: float, mcar_seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    train = hourly[(hourly["timestamp_utc"] >= TRAIN_START) & (hourly["timestamp_utc"] < TRAIN_END)].copy()
    test = hourly[(hourly["timestamp_utc"] >= TEST_START) & (hourly["timestamp_utc"] < TEST_END)].copy()

    train_available = (train["coverage"] >= min_coverage) & np.isfinite(train["precip_h_mm"])
    train_wet = train_available & (train["precip_h_mm"] > 0)
    if int(train_wet.sum()) < 100:
        raise RuntimeError("Insufficient wet-hour TRAIN data for DWD external check.")

    u95 = float(train.loc[train_wet, "precip_h_mm"].quantile(0.95))
    u99 = float(train.loc[train_wet, "precip_h_mm"].quantile(0.99))
    u999 = float(train.loc[train_wet, "precip_h_mm"].quantile(0.999))
    peak_u999 = float(train.loc[train_available, "peak_1min_mm_h"].quantile(0.999))
    gross_hard = max(100.0, 3.0 * u999)
    peak_hard = max(300.0, 3.0 * peak_u999)

    available = (test["coverage"] >= min_coverage) & np.isfinite(test["precip_h_mm"])
    wet = available & (test["precip_h_mm"] > 0)
    exceed_u95 = available & (test["precip_h_mm"] >= u95)
    exceed_u99 = available & (test["precip_h_mm"] >= u99)

    physical_fail = (
        available
        & (
            (test["precip_h_mm"] < 0)
            | (test["precip_h_mm"] > gross_hard)
            | (test["peak_1min_mm_h"] > peak_hard)
        )
    )
    tail_delete = physical_fail | (available & (test["precip_h_mm"] >= u99))
    extra_delete_n = int(tail_delete.sum() - physical_fail.sum())
    rng = np.random.default_rng(int(mcar_seed))
    mcar = physical_fail.to_numpy(dtype=bool).copy()
    candidate_idx = np.flatnonzero((available & (~physical_fail)).to_numpy(dtype=bool))
    if extra_delete_n > 0:
        if extra_delete_n > len(candidate_idx):
            raise RuntimeError("MCAR deletion count exceeds available candidate rows.")
        picked = rng.choice(candidate_idx, size=extra_delete_n, replace=False)
        mcar[picked] = True

    policies = {
        "Raw DWD archive": np.zeros(len(test), dtype=bool),
        "Conservative physical QC": physical_fail.to_numpy(dtype=bool),
        "Matched MCAR deletion": mcar,
        "Tail-censor stressor": tail_delete.to_numpy(dtype=bool),
    }

    rows: list[dict[str, object]] = []
    total_hours = len(test)
    for policy, delete_arr in policies.items():
        delete = pd.Series(delete_arr, index=test.index)
        kept_available = available & (~delete)
        kept_wet = wet & (~delete)
        kept_u95 = exceed_u95 & (~delete)
        kept_u99 = exceed_u99 & (~delete)
        u95_n = int(exceed_u95.sum())
        u99_n = int(exceed_u99.sum())
        u95_k = int(kept_u95.sum())
        u99_k = int(kept_u99.sum())
        u95_lo, u95_hi = wilson_ci(u95_k, u95_n)
        u99_lo, u99_hi = wilson_ci(u99_k, u99_n)
        rows.append(
            {
                "policy": policy,
                "coverage_pct": 100.0 * float(kept_available.sum()) / float(total_hours),
                "wet_coverage_pct": 100.0 * float(kept_wet.sum()) / max(float(wet.sum()), 1.0),
                "u95_events": u95_n,
                "u95_retention": float(u95_k) / max(float(u95_n), 1.0),
                "u95_ci95_low": u95_lo,
                "u95_ci95_high": u95_hi,
                "u99_events": u99_n,
                "u99_retention": float(u99_k) / max(float(u99_n), 1.0),
                "u99_ci95_low": u99_lo,
                "u99_ci95_high": u99_hi,
                "mnar_wet_spread": mnar_spread_wet(test["precip_h_mm"], delete),
            }
        )

    diagnostics = {
        "train_wet_u95_mm": u95,
        "train_wet_u99_mm": u99,
        "train_wet_u999_mm": u999,
        "hard_hourly_threshold_mm": gross_hard,
        "hard_peak_1min_threshold_mm_h": peak_hard,
        "test_total_station_hours": float(total_hours),
        "test_available_station_hours": float(available.sum()),
        "test_wet_station_hours": float(wet.sum()),
        "mcar_seed": float(mcar_seed),
        "matched_random_deletions": float(extra_delete_n),
    }
    return pd.DataFrame(rows), diagnostics


def fmt_pct_ci(p: float, lo: float, hi: float) -> str:
    return f"{100.0 * p:.2f} [{100.0 * lo:.2f}, {100.0 * hi:.2f}]"


def write_tex_table(metrics: pd.DataFrame, out_path: Path) -> None:
    order = ["Raw DWD archive", "Conservative physical QC", "Matched MCAR deletion", "Tail-censor stressor"]
    metrics = metrics.set_index("policy").loc[order].reset_index()
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Policy & Coverage (\%) & Wet coverage (\%) & Wet $u95$ retention [95\% CI] & Wet $u99$ retention [95\% CI] & Wet state-dep. spread \\",
        r"\midrule",
    ]
    for _, row in metrics.iterrows():
        lines.append(
            "{} & {:.2f} & {:.2f} & {} & {} & {:.3f} \\\\".format(
                row["policy"],
                float(row["coverage_pct"]),
                float(row["wet_coverage_pct"]),
                fmt_pct_ci(float(row["u95_retention"]), float(row["u95_ci95_low"]), float(row["u95_ci95_high"])),
                fmt_pct_ci(float(row["u99_retention"]), float(row["u99_ci95_low"]), float(row["u99_ci95_high"])),
                float(row["mnar_wet_spread"]),
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.article_dir.mkdir(parents=True, exist_ok=True)

    hourly, selected = load_dwd_hourly(
        selected_path=args.dwd_selected_stations,
        normalized_dir=args.dwd_normalized_dir,
        n_stations=int(args.n_stations),
    )
    metrics, diagnostics = policy_metrics(
        hourly,
        min_coverage=float(args.min_coverage),
        mcar_seed=int(args.mcar_seed),
    )

    hourly_out = args.results_dir / "dwd_external_hourly_metrics.parquet"
    selected_out = args.results_dir / "dwd_external_selected_stations.csv"
    metrics_out = args.results_dir / "dwd_external_policy_metrics.csv"
    diag_out = args.results_dir / "dwd_external_diagnostics.json"
    tex_out = args.results_dir / "tab_dwd_external_transfer.tex"

    hourly.to_parquet(hourly_out, index=False)
    selected.to_csv(selected_out, index=False)
    metrics.to_csv(metrics_out, index=False)
    pd.Series(diagnostics).to_json(diag_out, indent=2)
    write_tex_table(metrics, tex_out)

    (args.article_dir / "tab_dwd_external_transfer.tex").write_text(tex_out.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"Wrote {metrics_out}")
    print(f"Wrote {tex_out}")
    print(f"Wrote {args.article_dir / 'tab_dwd_external_transfer.tex'}")


if __name__ == "__main__":
    main()
