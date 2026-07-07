from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


TRAIN_END = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
TEST_END = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
K_NEIGHBORS = 5
SUPPORT_MIN_NEIGHBORS = 2
PRESSURE_RESIDUAL_FLOOR_HPA = 8.0


@dataclass(frozen=True)
class TimedStep:
    name: str
    seconds: float


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Archive-wide spatial quarantine and direct-deletion experiments."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=root / "runs_additional" / "BASELINE" / "results_v3",
    )
    parser.add_argument(
        "--raw-parquet",
        type=Path,
        default=root.parent / "avamet_all.parquet",
    )
    parser.add_argument("--article-dir", type=Path, default=root / "article")
    return parser.parse_args()


def timed(steps: list[TimedStep], name: str, start: float) -> None:
    steps.append(TimedStep(name, time.perf_counter() - start))


def mask_invalid_pressure(p0: pd.DataFrame) -> pd.Series:
    """Pressure with hard-invalid values masked to NaN.

    A spatial-consistency reference must never be built from values that the
    local QC rules already mark as hard FAIL (zero/out-of-range). Including a
    dead sensor reading 0 hPa in the neighbour reference drags the neighbour
    median by ~1000 hPa and falsely flags healthy stations whose only fault is
    a dead neighbour. We therefore exclude those values from both the target
    anomaly and the neighbour reference; hard faults remain handled by the
    local FAIL deletion rule, not by the spatial action.
    """
    invalid = pd.Series(False, index=p0.index)
    for col in ("pressure_fail_zero", "pressure_fail_out_range_after_norm"):
        if col in p0.columns:
            invalid = invalid | p0[col].fillna(0).astype(bool)
    return p0["pressure_hpa"].where(~invalid)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0088
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * radius_km * math.asin(math.sqrt(a))


def load_product(results_dir: Path, product: str) -> pd.DataFrame:
    df = pd.read_parquet(results_dir / "products" / product / "data.parquet", engine="pyarrow")
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df.sort_values(["station_id", "timestamp_utc"]).reset_index(drop=True)


def load_coordinates(raw_parquet: Path, station_ids: list[str]) -> pd.DataFrame:
    quoted = ",".join("'" + sid.replace("'", "''") + "'" for sid in station_ids)
    query = f"""
        SELECT
            station_id,
            median(lat) AS lat,
            median(lon) AS lon,
            median(alt) AS alt
        FROM read_parquet('{raw_parquet.as_posix()}')
        WHERE station_id IN ({quoted})
        GROUP BY station_id
        ORDER BY station_id
    """
    coords = duckdb.connect().execute(query).df()
    if len(coords) != len(station_ids):
        missing = sorted(set(station_ids) - set(coords["station_id"]))
        raise RuntimeError(f"Missing coordinates for stations: {missing}")
    return coords


def nearest_neighbors(coords: pd.DataFrame, k: int = K_NEIGHBORS) -> dict[str, list[str]]:
    rows = coords.set_index("station_id")
    out: dict[str, list[str]] = {}
    for sid, row in rows.iterrows():
        distances: list[tuple[float, str]] = []
        for other, other_row in rows.iterrows():
            if sid == other:
                continue
            distances.append(
                (
                    haversine_km(float(row["lat"]), float(row["lon"]), float(other_row["lat"]), float(other_row["lon"])),
                    str(other),
                )
            )
        distances.sort(key=lambda item: item[0])
        out[str(sid)] = [other for _, other in distances[:k]]
    return out


def train_thresholds(p0: pd.DataFrame, neighbors: dict[str, list[str]]) -> tuple[pd.DataFrame, dict[str, float], float]:
    train = p0[p0["timestamp_utc"] < TRAIN_END].copy()
    train["pressure_hpa"] = mask_invalid_pressure(train)
    wind_thresholds = pd.DataFrame(
        {
            "wind_u90": train.groupby("station_id")["wind_gust"].quantile(0.90),
            "wind_u99": train.groupby("station_id")["wind_gust"].quantile(0.99),
            "wind_delta_q995": (train["wind_gust"] - train["wind_mean"]).groupby(train["station_id"]).quantile(0.995),
        }
    )
    wind_thresholds["wind_delta_q995"] = wind_thresholds["wind_delta_q995"].clip(lower=30.0)

    pressure_medians = train.groupby("station_id")["pressure_hpa"].median().to_dict()
    wide_pressure = train.pivot(index="timestamp_utc", columns="station_id", values="pressure_hpa")
    pressure_anom = wide_pressure.subtract(pd.Series(pressure_medians), axis="columns")
    residuals: list[np.ndarray] = []
    for station_id, nids in neighbors.items():
        valid_nids = [n for n in nids if n in pressure_anom.columns]
        if len(valid_nids) < SUPPORT_MIN_NEIGHBORS or station_id not in pressure_anom.columns:
            continue
        target = pressure_anom[station_id]
        neigh = pressure_anom[valid_nids]
        valid = target.notna() & (neigh.notna().sum(axis=1) >= SUPPORT_MIN_NEIGHBORS)
        if valid.any():
            residual = (target[valid] - neigh[valid].median(axis=1)).abs().to_numpy(dtype=float)
            residuals.append(residual[np.isfinite(residual)])
    if residuals:
        pressure_threshold = max(PRESSURE_RESIDUAL_FLOOR_HPA, float(np.quantile(np.concatenate(residuals), 0.995)))
    else:
        pressure_threshold = PRESSURE_RESIDUAL_FLOOR_HPA
    return wind_thresholds, pressure_medians, pressure_threshold


def spatial_quarantine_flags(
    p0: pd.DataFrame,
    wind_thresholds: pd.DataFrame,
    pressure_medians: dict[str, float],
    pressure_threshold: float,
    neighbors: dict[str, list[str]],
) -> pd.DataFrame:
    p0 = p0.assign(pressure_hpa=mask_invalid_pressure(p0))
    wide_pressure = p0.pivot(index="timestamp_utc", columns="station_id", values="pressure_hpa")
    wide_wind = p0.pivot(index="timestamp_utc", columns="station_id", values="wind_gust")
    wide_wind_mean = p0.pivot(index="timestamp_utc", columns="station_id", values="wind_mean")
    pressure_anom = wide_pressure.subtract(pd.Series(pressure_medians), axis="columns")
    wind_high = wide_wind.ge(wind_thresholds["wind_u90"], axis="columns")

    flags = pd.DataFrame(
        {
            "station_id": p0["station_id"].to_numpy(),
            "timestamp_utc": p0["timestamp_utc"].to_numpy(),
            "spatial_pressure_quarantine": False,
            "spatial_wind_quarantine": False,
        }
    )

    for station_id, nids in neighbors.items():
        if station_id not in wide_pressure.columns:
            continue
        valid_nids = [n for n in nids if n in wide_pressure.columns]
        if len(valid_nids) < SUPPORT_MIN_NEIGHBORS:
            continue
        part_idx = p0.index[p0["station_id"].eq(station_id)]
        part_times = p0.loc[part_idx, "timestamp_utc"]

        target_pressure = pressure_anom[station_id]
        neigh_pressure = pressure_anom[valid_nids]
        valid_pressure = target_pressure.notna() & (neigh_pressure.notna().sum(axis=1) >= SUPPORT_MIN_NEIGHBORS)
        pressure_resid = (target_pressure - neigh_pressure.median(axis=1)).abs()
        pressure_times = set(pressure_resid[valid_pressure & pressure_resid.gt(pressure_threshold)].index)
        if pressure_times:
            flags.loc[part_idx[part_times.isin(pressure_times).to_numpy()], "spatial_pressure_quarantine"] = True

        target_wind = wide_wind[station_id]
        target_delta = target_wind - wide_wind_mean[station_id]
        support_n = wind_high[valid_nids].sum(axis=1)
        wind_times = set(
            target_wind[
                target_wind.ge(float(wind_thresholds.loc[station_id, "wind_u99"]))
                & target_delta.ge(float(wind_thresholds.loc[station_id, "wind_delta_q995"]))
                & support_n.lt(SUPPORT_MIN_NEIGHBORS)
            ].index
        )
        if wind_times:
            flags.loc[part_idx[part_times.isin(wind_times).to_numpy()], "spatial_wind_quarantine"] = True

    flags["spatial_any_quarantine"] = flags["spatial_pressure_quarantine"] | flags["spatial_wind_quarantine"]
    return flags


def phase_mask(df: pd.DataFrame, phase: str) -> pd.Series:
    if phase == "train":
        return df["timestamp_utc"] < TRAIN_END
    if phase == "test":
        return (df["timestamp_utc"] >= TRAIN_END) & (df["timestamp_utc"] < TEST_END)
    raise ValueError(phase)


def scope_mask(df: pd.DataFrame, scope: str) -> pd.Series:
    if scope == "global":
        return pd.Series(True, index=df.index)
    return df["stratum"].eq(scope)


def unavailable_mask(product_df: pd.DataFrame, flags: pd.DataFrame | None, policy: str, variable: str) -> pd.Series:
    missing = product_df[variable].isna()
    if policy in {"P4 pressure spatial quarantine", "P4 pressure spatial direct deletion"}:
        if variable == "pressure_hpa":
            return missing | flags["spatial_pressure_quarantine"]
    if policy in {"P5 full spatial quarantine", "P6 full spatial direct deletion"}:
        if variable == "pressure_hpa":
            return missing | flags["spatial_pressure_quarantine"]
        if variable in {"wind_gust", "wind_mean"}:
            return missing | flags["spatial_wind_quarantine"]
    return missing


def missingness_metrics(
    p0: pd.DataFrame,
    products: dict[str, pd.DataFrame],
    flags: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    variables = ["wind_gust", "pressure_hpa"]
    scopes = ["global", "clean", "medium", "dirty"]
    phases = ["train", "test"]
    for policy, df in products.items():
        for phase in phases:
            pm = phase_mask(p0, phase)
            for scope in scopes:
                sm = scope_mask(p0, scope)
                mask = pm & sm
                if not mask.any():
                    continue
                for variable in variables:
                    raw_missing = p0.loc[mask, variable].isna()
                    unavailable = unavailable_mask(df.loc[mask].reset_index(drop=True), flags.loc[mask].reset_index(drop=True), policy, variable)
                    rows.append(
                        {
                            "policy": policy,
                            "phase": phase,
                            "scope": scope,
                            "variable": variable,
                            "n_total": int(mask.sum()),
                            "missing_rate_p0": float(raw_missing.mean()),
                            "unavailable_rate_policy": float(unavailable.mean()),
                            "additional_unavailable": float(unavailable.mean() - raw_missing.mean()),
                        }
                    )
    return pd.DataFrame(rows)


def mnar_spread_metrics(p0: pd.DataFrame, products: dict[str, pd.DataFrame], flags: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scopes = ["global", "clean", "medium", "dirty"]
    for policy, df in products.items():
        for phase in ["train", "test"]:
            pm = phase_mask(p0, phase)
            for scope in scopes:
                mask = pm & scope_mask(p0, scope) & p0["wind_gust"].notna()
                part = p0.loc[mask, ["wind_gust"]].copy()
                if len(part) < 100:
                    continue
                try:
                    part["decile"] = pd.qcut(part["wind_gust"], 10, labels=False, duplicates="drop") + 1
                except ValueError:
                    continue
                unavailable = unavailable_mask(
                    df.loc[mask].reset_index(drop=True),
                    flags.loc[mask].reset_index(drop=True),
                    policy,
                    "wind_gust",
                ).to_numpy(dtype=bool)
                tmp = part.reset_index(drop=True)
                tmp["unavailable"] = unavailable
                rates = tmp.groupby("decile")["unavailable"].mean()
                # State-dependence spread per Methods: robust inter-decile range
                # Q0.90 - Q0.10 of the removal-by-decile curve (matches metrics_mnar
                # / the main MNAR table), not the max-min full range, so a single
                # pathological decile does not dominate the diagnostic.
                rows.append(
                    {
                        "policy": policy,
                        "phase": phase,
                        "scope": scope,
                        "state_wind_spread": float(rates.quantile(0.90) - rates.quantile(0.10)),
                        "max_decile_unavailable": float(rates.max()),
                        "min_decile_unavailable": float(rates.min()),
                    }
                )
    return pd.DataFrame(rows)


def raw_exceedance_retention(p0: pd.DataFrame, products: dict[str, pd.DataFrame], flags: pd.DataFrame) -> pd.DataFrame:
    train = p0[p0["timestamp_utc"] < TRAIN_END]
    thresholds = train.groupby("station_id")["wind_gust"].quantile([0.95, 0.99]).unstack()
    thresholds.columns = ["u95", "u99"]
    rows: list[dict[str, object]] = []
    mask_test_dirty = phase_mask(p0, "test") & p0["stratum"].eq("dirty") & p0["wind_gust"].notna()
    base = p0.loc[mask_test_dirty, ["station_id", "wind_gust"]].copy()
    for q in ["u95", "u99"]:
        thr = base["station_id"].map(thresholds[q])
        event_mask = base["wind_gust"].ge(thr).to_numpy()
        idx = base.index[event_mask]
        for policy, df in products.items():
            unavailable = unavailable_mask(
                df.loc[idx].reset_index(drop=True),
                flags.loc[idx].reset_index(drop=True),
                policy,
                "wind_gust",
            )
            retained = (~unavailable).to_numpy(dtype=bool)
            rows.append(
                {
                    "policy": policy,
                    "threshold": q,
                    "n_raw_events": int(len(idx)),
                    "retained": int(retained.sum()),
                    "retention": float(retained.mean()) if len(retained) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def quarantine_counts(p0: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for phase in ["train", "test"]:
        pm = phase_mask(p0, phase)
        for scope in ["global", "clean", "medium", "dirty"]:
            mask = pm & scope_mask(p0, scope)
            rows.append(
                {
                    "phase": phase,
                    "scope": scope,
                    "n_total": int(mask.sum()),
                    "pressure_quarantine": int(flags.loc[mask, "spatial_pressure_quarantine"].sum()),
                    "wind_quarantine": int(flags.loc[mask, "spatial_wind_quarantine"].sum()),
                    "any_quarantine": int(flags.loc[mask, "spatial_any_quarantine"].sum()),
                    "pressure_rate": float(flags.loc[mask, "spatial_pressure_quarantine"].mean()),
                    "wind_rate": float(flags.loc[mask, "spatial_wind_quarantine"].mean()),
                    "any_rate": float(flags.loc[mask, "spatial_any_quarantine"].mean()),
                }
            )
    return pd.DataFrame(rows)


def select_real_episodes(p0: pd.DataFrame, flags: pd.DataFrame, coords: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    df = p0.join(flags[["spatial_pressure_quarantine", "spatial_wind_quarantine", "spatial_any_quarantine"]])
    test = df[phase_mask(df, "test") & df["spatial_any_quarantine"]].copy()
    test["kind"] = np.select(
        [test["spatial_pressure_quarantine"], test["spatial_wind_quarantine"]],
        ["pressure spatial anomaly", "wind isolated high-state"],
        default="combined spatial anomaly",
    )
    test["date"] = test["timestamp_utc"].dt.date.astype(str)
    grouped = (
        test.groupby(["station_id", "date", "kind"])
        .agg(
            start_time=("timestamp_utc", "min"),
            end_time=("timestamp_utc", "max"),
            n_rows=("timestamp_utc", "size"),
            max_wind_gust=("wind_gust", "max"),
            min_pressure=("pressure_hpa", "min"),
            max_pressure=("pressure_hpa", "max"),
        )
        .reset_index()
    )
    grouped["duration_min"] = (grouped["end_time"] - grouped["start_time"]).dt.total_seconds() / 60.0 + 10.0
    grouped["classification"] = np.where(
        grouped["kind"].eq("pressure spatial anomaly"),
        "probable sensor fault",
        "indeterminate isolated event",
    )
    grouped = grouped.sort_values(["kind", "n_rows", "max_wind_gust"], ascending=[True, False, False])
    diverse_rows = []
    per_station: dict[str, int] = {}
    per_kind: dict[str, int] = {}
    for _, row in grouped.iterrows():
        station_id = str(row["station_id"])
        kind = str(row["kind"])
        if per_station.get(station_id, 0) >= 3:
            continue
        if per_kind.get(kind, 0) >= 12:
            continue
        diverse_rows.append(row)
        per_station[station_id] = per_station.get(station_id, 0) + 1
        per_kind[kind] = per_kind.get(kind, 0) + 1
        if len(diverse_rows) >= 20:
            break
    grouped = pd.DataFrame(diverse_rows)
    grouped.to_csv(out_dir / "real_episode_candidates.csv", index=False)
    return grouped


def write_tables(
    article_dir: Path,
    metrics: pd.DataFrame,
    mnar: pd.DataFrame,
    retention: pd.DataFrame,
    counts: pd.DataFrame,
    episodes: pd.DataFrame,
    timing: pd.DataFrame,
) -> None:
    main_policies = [
        "Raw",
        "Delete-all-flags",
        "Constraint-selected P3",
        "P4 pressure spatial quarantine",
        "P5 full spatial quarantine",
        "P6 full spatial direct deletion",
    ]
    miss = metrics[
        (metrics["phase"].eq("test"))
        & (metrics["scope"].eq("dirty"))
        & (metrics["variable"].eq("wind_gust"))
        & (metrics["policy"].isin(main_policies))
    ].set_index("policy")
    mn = mnar[(mnar["phase"].eq("test")) & (mnar["scope"].eq("dirty"))].set_index("policy")
    ret = retention[retention["threshold"].eq("u99")].set_index("policy")
    ct = counts[(counts["phase"].eq("test")) & (counts["scope"].eq("dirty"))].iloc[0]
    lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Policy & Wind add. unavail. & State-dep. spread & u99 retention & Pressure add. unavail. \\\\",
        "\\midrule",
    ]
    for policy in main_policies:
        if policy not in miss.index:
            continue
        pressure = metrics[
            (metrics["phase"].eq("test"))
            & (metrics["scope"].eq("dirty"))
            & (metrics["variable"].eq("pressure_hpa"))
            & (metrics["policy"].eq(policy))
        ].iloc[0]
        lines.append(
            f"{policy} & {miss.loc[policy, 'additional_unavailable']:.6f} & "
            f"{mn.loc[policy, 'state_wind_spread']:.6f} & "
            f"{ret.loc[policy, 'retention']:.4f} & "
            f"{pressure['additional_unavailable']:.6f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (article_dir / "tab_spatial_quarantine_archive.tex").write_text("\n".join(lines), encoding="utf-8")

    ep = episodes.head(10).copy()
    lines = [
        "\\begin{tabular}{llllrr}",
        "\\toprule",
        "Station & Start UTC & Kind & Classification & Rows & Max gust \\\\",
        "\\midrule",
    ]
    for _, row in ep.iterrows():
        max_gust = "--" if pd.isna(row["max_wind_gust"]) else f"{row['max_wind_gust']:.1f}"
        lines.append(
            f"{row['station_id']} & {pd.Timestamp(row['start_time']).strftime('%Y-%m-%d %H:%M')} & "
            f"{row['kind']} & {row['classification']} & {int(row['n_rows'])} & {max_gust} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (article_dir / "tab_real_episode_candidates.tex").write_text("\n".join(lines), encoding="utf-8")

    t = timing.copy()
    lines = ["\\begin{tabular}{lr}", "\\toprule", "Step & Seconds \\\\", "\\midrule"]
    for _, row in t.iterrows():
        lines.append(f"{row['step']} & {row['seconds']:.2f} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (article_dir / "tab_computational_feasibility.tex").write_text("\n".join(lines), encoding="utf-8")


def write_policy_manifests(results_dir: Path, flags: pd.DataFrame, pressure_threshold: float, timings: list[TimedStep]) -> None:
    out_dir = results_dir / "spatial_quarantine_archive"
    out_dir.mkdir(parents=True, exist_ok=True)
    flags.to_parquet(out_dir / "spatial_quarantine_flags.parquet", index=False)
    manifest = {
        "policies": {
            "P4 pressure spatial quarantine": "P3 local rule actions plus spatial pressure quarantine flags; quarantined observations remain auditable.",
            "P4 pressure spatial direct deletion": "Same pressure triggers as P4, but treated as direct deletion in automatic-availability metrics.",
            "P5 full spatial quarantine": "P3 plus pressure and strict isolated-wind spatial quarantine; retained as a sensitivity because wind quarantine affects real high-state retention.",
            "P6 full spatial direct deletion": "Same full spatial triggers as P5, but treated as direct deletion in automatic-availability metrics.",
        },
        "pressure_residual_threshold_hpa": pressure_threshold,
        "k_neighbors": K_NEIGHBORS,
        "support_min_neighbors": SUPPORT_MIN_NEIGHBORS,
        "timings": [{"step": step.name, "seconds": step.seconds} for step in timings],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    results_dir: Path = args.results_dir
    out_dir = results_dir / "spatial_quarantine_archive"
    out_dir.mkdir(parents=True, exist_ok=True)
    args.article_dir.mkdir(parents=True, exist_ok=True)
    timings: list[TimedStep] = []

    t0 = time.perf_counter()
    p0 = load_product(results_dir, "P0")
    p1 = load_product(results_dir, "P1")
    p3 = load_product(results_dir, "P3")
    p0 = p0[p0["timestamp_utc"] < TEST_END].reset_index(drop=True)
    p1 = p1[p1["timestamp_utc"] < TEST_END].reset_index(drop=True)
    p3 = p3[p3["timestamp_utc"] < TEST_END].reset_index(drop=True)
    timed(timings, "load products", t0)

    t0 = time.perf_counter()
    station_ids = sorted(p0["station_id"].unique().tolist())
    coords = load_coordinates(args.raw_parquet, station_ids)
    neighbors = nearest_neighbors(coords)
    wind_thresholds, pressure_medians, pressure_threshold = train_thresholds(p0, neighbors)
    timed(timings, "train spatial thresholds", t0)

    t0 = time.perf_counter()
    flags = spatial_quarantine_flags(p0, wind_thresholds, pressure_medians, pressure_threshold, neighbors)
    timed(timings, "compute archive spatial flags", t0)

    products = {
        "Raw": p0,
        "Delete-all-flags": p1,
        "Constraint-selected P3": p3,
        "P4 pressure spatial quarantine": p3,
        "P4 pressure spatial direct deletion": p3,
        "P5 full spatial quarantine": p3,
        "P6 full spatial direct deletion": p3,
    }

    t0 = time.perf_counter()
    metrics = missingness_metrics(p0, products, flags)
    mnar = mnar_spread_metrics(p0, products, flags)
    retention = raw_exceedance_retention(p0, products, flags)
    counts = quarantine_counts(p0, flags)
    timed(timings, "compute archive metrics", t0)

    t0 = time.perf_counter()
    episodes = select_real_episodes(p0, flags, coords, out_dir)
    timed(timings, "select real episodes", t0)

    timing_df = pd.DataFrame([{"step": s.name, "seconds": s.seconds} for s in timings])
    metrics.to_csv(out_dir / "archive_missingness.csv", index=False)
    mnar.to_csv(out_dir / "archive_mnar.csv", index=False)
    retention.to_csv(out_dir / "archive_retention.csv", index=False)
    counts.to_csv(out_dir / "archive_quarantine_counts.csv", index=False)
    timing_df.to_csv(out_dir / "computational_timing.csv", index=False)
    write_policy_manifests(results_dir, flags, pressure_threshold, timings)
    write_tables(args.article_dir, metrics, mnar, retention, counts, episodes, timing_df)

    print(f"Wrote {out_dir / 'archive_missingness.csv'}")
    print(f"Wrote {out_dir / 'archive_mnar.csv'}")
    print(f"Wrote {out_dir / 'archive_retention.csv'}")
    print(f"Wrote {out_dir / 'archive_quarantine_counts.csv'}")
    print(f"Wrote {out_dir / 'real_episode_candidates.csv'}")
    print(f"Wrote {args.article_dir / 'tab_spatial_quarantine_archive.tex'}")
    print(f"Wrote {args.article_dir / 'tab_real_episode_candidates.tex'}")
    print(f"Wrote {args.article_dir / 'tab_computational_feasibility.tex'}")
    print(f"Pressure spatial residual threshold hPa: {pressure_threshold:.3f}")


if __name__ == "__main__":
    main()
