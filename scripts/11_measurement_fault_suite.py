from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import duckdb
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

matplotlib.use("Agg")


TRAIN_END = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
TEST_END = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
SEED = 20260519
K_NEIGHBORS = 5
SUPPORT_MIN_NEIGHBORS = 2
PRESSURE_RESIDUAL_FLOOR_HPA = 8.0


@dataclass(frozen=True)
class FaultSpec:
    key: str
    label: str
    variable_family: str
    length_rows: int
    events_per_station: int
    spatial_family: str


FAULTS = [
    FaultSpec(
        key="pressure_stuck",
        label="Pressure stuck",
        variable_family="pressure",
        length_rows=36,
        events_per_station=2,
        spatial_family="pressure",
    ),
    FaultSpec(
        key="pressure_unit_scale",
        label="Pressure unit-scale",
        variable_family="pressure",
        length_rows=12,
        events_per_station=2,
        spatial_family="pressure",
    ),
    FaultSpec(
        key="wind_incoherence",
        label="Wind cross-channel incoherence",
        variable_family="wind",
        length_rows=3,
        events_per_station=2,
        spatial_family="wind",
    ),
    FaultSpec(
        key="pressure_offset",
        label="Pressure offset",
        variable_family="pressure",
        length_rows=36,
        events_per_station=2,
        spatial_family="pressure",
    ),
    FaultSpec(
        key="pressure_drift",
        label="Pressure drift",
        variable_family="pressure",
        length_rows=72,
        events_per_station=1,
        spatial_family="pressure",
    ),
    FaultSpec(
        key="wind_spike_burst",
        label="Wind spike/burst",
        variable_family="wind",
        length_rows=3,
        events_per_station=3,
        spatial_family="wind",
    ),
]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    paper_dir = repo_root
    parser = argparse.ArgumentParser(
        description="Controlled multi-class measurement-fault suite for paper 2."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=paper_dir / "runs_additional" / "BASELINE" / "results_v3",
    )
    parser.add_argument("--raw-parquet", type=Path, default=repo_root / "avamet_all.parquet")
    parser.add_argument("--article-dir", type=Path, default=paper_dir / "article")
    return parser.parse_args()


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0088
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * radius_km * math.asin(math.sqrt(a))


def wilson_ci(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return (np.nan, np.nan)
    p = successes / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denom
    half = z * math.sqrt((p * (1.0 - p) / total) + (z * z / (4.0 * total * total))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_p0(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "products" / "P0" / "data.parquet"
    df = pd.read_parquet(
        path,
        columns=[
            "station_id",
            "timestamp_utc",
            "stratum",
            "pressure_hpa",
            "wind_mean",
            "wind_gust",
            "wind_dir",
        ],
        engine="pyarrow",
    )
    df = df.reset_index(drop=False).rename(columns={"index": "_row_id"})
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df


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


def mask_invalid_pressure(df: pd.DataFrame) -> pd.Series:
    """Pressure with hard-invalid (zero/out-of-range FAIL) values masked to NaN.

    Hard faults are handled by the local FAIL rule; they must not enter the
    spatial neighbour reference, where a dead sensor reading 0 hPa would drag
    the neighbour median by ~1000 hPa and falsely flag healthy stations.
    """
    invalid = pd.Series(False, index=df.index)
    for col in ("pressure_fail_zero", "pressure_fail_out_range_after_norm"):
        if col in df.columns:
            invalid = invalid | df[col].fillna(0).astype(bool)
    return df["pressure_hpa"].where(~invalid)


def pressure_residual_thresholds(p0: pd.DataFrame, neighbors: dict[str, list[str]]) -> tuple[dict[str, float], float]:
    train = p0[p0["timestamp_utc"] < TRAIN_END].copy()
    train["pressure_hpa"] = mask_invalid_pressure(train)
    med = train.groupby("station_id")["pressure_hpa"].median()
    wide = train.pivot(index="timestamp_utc", columns="station_id", values="pressure_hpa")
    anomalies = wide.subtract(med, axis="columns")
    residuals: list[np.ndarray] = []
    for station_id, nids in neighbors.items():
        valid_nids = [n for n in nids if n in anomalies.columns]
        neigh = anomalies[valid_nids]
        target = anomalies[station_id]
        valid = target.notna() & (neigh.notna().sum(axis=1) >= SUPPORT_MIN_NEIGHBORS)
        if valid.any():
            residual = (target[valid] - neigh[valid].median(axis=1)).abs().to_numpy(dtype=float)
            residuals.append(residual[np.isfinite(residual)])
    if not residuals:
        threshold = PRESSURE_RESIDUAL_FLOOR_HPA
    else:
        threshold = max(PRESSURE_RESIDUAL_FLOOR_HPA, float(np.quantile(np.concatenate(residuals), 0.995)))
    return med.to_dict(), threshold


def choose_events(test_dirty: pd.DataFrame, spec: FaultSpec, rng: np.random.Generator) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    context = 6
    for station_id, part in test_dirty.groupby("station_id", sort=True):
        part = part.sort_values("timestamp_utc").reset_index(drop=True)
        value_cols = ["pressure_hpa"] if spec.variable_family == "pressure" else ["wind_gust", "wind_mean"]
        valid_values = part[value_cols].notna().all(axis=1).to_numpy()
        ts = part["timestamp_utc"].to_numpy()
        candidates: list[int] = []
        for start in range(0, len(part) - spec.length_rows - 1):
            stop = start + spec.length_rows
            if not valid_values[start:stop].all():
                continue
            diffs_min = np.diff(ts[start:stop]).astype("timedelta64[m]").astype(float)
            if len(diffs_min) and np.nanmax(diffs_min) > 20.0:
                continue
            candidates.append(start)
        rng.shuffle(candidates)
        chosen: list[int] = []
        for start in candidates:
            if all(abs(start - other) > spec.length_rows + 2 * context for other in chosen):
                chosen.append(start)
            if len(chosen) >= spec.events_per_station:
                break
        for event_i, start in enumerate(chosen):
            stop = start + spec.length_rows
            rows.append(
                {
                    "fault_type": spec.key,
                    "fault_label": spec.label,
                    "station_id": station_id,
                    "event_id": f"{spec.key}_{station_id}_{event_i:02d}",
                    "start_pos": int(start),
                    "stop_pos": int(stop),
                    "start_time": part.loc[start, "timestamp_utc"],
                    "stop_time": part.loc[stop - 1, "timestamp_utc"],
                }
            )
    return pd.DataFrame(rows)


def apply_faults(
    test_dirty: pd.DataFrame,
    events: pd.DataFrame,
    spec: FaultSpec,
    thresholds: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    faulted = test_dirty.sort_values(["station_id", "timestamp_utc"]).copy()
    faulted["_injected"] = False
    faulted["_event_id"] = ""
    faulted["_fault_type"] = ""
    faulted = faulted.reset_index(drop=True)
    label_rows: list[dict[str, object]] = []

    for _, event in events.iterrows():
        station_id = str(event["station_id"])
        idx = faulted.index[faulted["station_id"].eq(station_id)].to_numpy()
        start = int(event["start_pos"])
        stop = int(event["stop_pos"])
        locs = idx[start:stop]
        if len(locs) != stop - start:
            continue
        original_pressure = faulted.loc[locs, "pressure_hpa"].to_numpy(dtype=float)
        original_gust = faulted.loc[locs, "wind_gust"].to_numpy(dtype=float)
        original_mean = faulted.loc[locs, "wind_mean"].to_numpy(dtype=float)

        if spec.key == "pressure_stuck":
            base = float(original_pressure[0])
            faulted.loc[locs, "pressure_hpa"] = base + rng.uniform(-0.01, 0.01, size=len(locs))
        elif spec.key == "pressure_unit_scale":
            faulted.loc[locs, "pressure_hpa"] = original_pressure * 0.750062
        elif spec.key == "wind_incoherence":
            faulted.loc[locs, "wind_mean"] = original_gust + 10.0
        elif spec.key == "pressure_offset":
            faulted.loc[locs, "pressure_hpa"] = original_pressure + 20.0
        elif spec.key == "pressure_drift":
            faulted.loc[locs, "pressure_hpa"] = original_pressure + np.linspace(0.0, 20.0, len(locs))
        elif spec.key == "wind_spike_burst":
            u99 = float(thresholds.loc[station_id, "wind_u99"])
            faulted.loc[locs, "wind_gust"] = np.maximum(original_gust + 60.0, u99 + 30.0)
            faulted.loc[locs, "wind_mean"] = original_mean
        else:
            raise ValueError(spec.key)

        faulted.loc[locs, "_injected"] = True
        faulted.loc[locs, "_event_id"] = str(event["event_id"])
        faulted.loc[locs, "_fault_type"] = spec.key
        for loc in locs:
            label_rows.append(
                {
                    "fault_type": spec.key,
                    "fault_label": spec.label,
                    "event_id": str(event["event_id"]),
                    "station_id": station_id,
                    "_row_id": int(faulted.loc[loc, "_row_id"]),
                    "timestamp_utc": faulted.loc[loc, "timestamp_utc"],
                    "is_injected": True,
                    "start_time": event["start_time"],
                }
            )

    context_rows: list[dict[str, object]] = []
    context = 6
    for _, event in events.iterrows():
        station_id = str(event["station_id"])
        idx = faulted.index[faulted["station_id"].eq(station_id)].to_numpy()
        start = max(0, int(event["start_pos"]) - context)
        stop = min(len(idx), int(event["stop_pos"]) + context)
        for loc in idx[start:stop]:
            if bool(faulted.loc[loc, "_injected"]):
                continue
            context_rows.append(
                {
                    "fault_type": spec.key,
                    "fault_label": spec.label,
                    "event_id": str(event["event_id"]),
                    "station_id": station_id,
                    "_row_id": int(faulted.loc[loc, "_row_id"]),
                    "timestamp_utc": faulted.loc[loc, "timestamp_utc"],
                    "is_injected": False,
                    "start_time": event["start_time"],
                }
            )

    labels = pd.DataFrame(label_rows + context_rows).drop_duplicates(["fault_type", "_row_id"])
    return faulted, labels


def compute_stuck_flags(df: pd.DataFrame, T_hours: float = 3.0, tol_hpa: float = 0.05) -> pd.Series:
    out = pd.Series(False, index=df.index)
    for _, part in df.sort_values(["station_id", "timestamp_utc"]).groupby("station_id", sort=False):
        vals = part["pressure_hpa"].to_numpy(dtype=float)
        ts = part["timestamp_utc"].to_numpy()
        idx = part.index.to_numpy()
        if len(part) < 2:
            continue
        start = 0
        for i in range(1, len(part)):
            gap_min = (pd.Timestamp(ts[i]) - pd.Timestamp(ts[i - 1])).total_seconds() / 60.0
            same = np.isfinite(vals[i]) and np.isfinite(vals[i - 1]) and abs(vals[i] - vals[i - 1]) <= tol_hpa
            contiguous = gap_min <= 20.0
            if not (same and contiguous):
                if i - start > 1:
                    duration_h = (
                        (pd.Timestamp(ts[i - 1]) - pd.Timestamp(ts[start])).total_seconds() / 60.0 + 10.0
                    ) / 60.0
                    if duration_h >= T_hours:
                        out.loc[idx[start:i]] = True
                start = i
        if len(part) - start > 1:
            duration_h = (
                (pd.Timestamp(ts[-1]) - pd.Timestamp(ts[start])).total_seconds() / 60.0 + 10.0
            ) / 60.0
            if duration_h >= T_hours:
                out.loc[idx[start:]] = True
    return out


def rule_flags(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    pressure = df["pressure_hpa"]
    wind_mean = df["wind_mean"]
    wind_gust = df["wind_gust"]
    wind_dir = df["wind_dir"]
    fail = (
        pressure.eq(0)
        | pressure.lt(850)
        | pressure.gt(1100)
        | wind_gust.lt(0)
        | wind_mean.lt(0)
        | (wind_mean.gt(wind_gust) & wind_mean.notna() & wind_gust.notna())
        | wind_dir.eq(999)
        | wind_dir.lt(0)
        | wind_dir.gt(360)
    ).fillna(False)
    suspect = compute_stuck_flags(df)
    return fail.astype(bool), suspect.astype(bool)


def spatial_flags(
    df: pd.DataFrame,
    thresholds: pd.DataFrame,
    pressure_medians: dict[str, float],
    pressure_residual_threshold: float,
    neighbors: dict[str, list[str]],
) -> tuple[pd.Series, pd.Series]:
    df = df.assign(pressure_hpa=mask_invalid_pressure(df))
    wide_pressure = df.pivot(index="timestamp_utc", columns="station_id", values="pressure_hpa")
    wide_wind = df.pivot(index="timestamp_utc", columns="station_id", values="wind_gust")
    wide_wind_mean = df.pivot(index="timestamp_utc", columns="station_id", values="wind_mean")
    pressure_anom = wide_pressure.subtract(pd.Series(pressure_medians), axis="columns")
    wind_high = wide_wind.ge(thresholds["wind_u90"], axis="columns")

    pressure_flag = pd.Series(False, index=df.index)
    wind_flag = pd.Series(False, index=df.index)
    index_map = df.set_index(["timestamp_utc", "station_id"]).index

    for station_id, nids in neighbors.items():
        if station_id not in wide_pressure.columns:
            continue
        valid_nids = [n for n in nids if n in wide_pressure.columns]
        if len(valid_nids) < SUPPORT_MIN_NEIGHBORS:
            continue
        target_pressure = pressure_anom[station_id]
        neigh_pressure = pressure_anom[valid_nids]
        valid_pressure = target_pressure.notna() & (neigh_pressure.notna().sum(axis=1) >= SUPPORT_MIN_NEIGHBORS)
        pressure_resid = (target_pressure - neigh_pressure.median(axis=1)).abs()
        pressure_times = pressure_resid[valid_pressure & pressure_resid.gt(pressure_residual_threshold)].index

        target_wind = wide_wind[station_id]
        target_delta = target_wind - wide_wind_mean[station_id]
        support_n = wind_high[valid_nids].sum(axis=1)
        wind_times = target_wind[
            target_wind.ge(float(thresholds.loc[station_id, "wind_u99"]))
            & target_delta.ge(float(thresholds.loc[station_id, "wind_delta_q995"]))
            & support_n.lt(SUPPORT_MIN_NEIGHBORS)
        ].index

        pressure_keys = set((t, station_id) for t in pressure_times)
        wind_keys = set((t, station_id) for t in wind_times)
        if pressure_keys:
            pressure_flag.loc[[key in pressure_keys for key in index_map]] = True
        if wind_keys:
            wind_flag.loc[[key in wind_keys for key in index_map]] = True

    return pressure_flag.astype(bool), wind_flag.astype(bool)


def metrics_for_detector(
    labels: pd.DataFrame,
    flag_by_row: pd.Series,
    events: pd.DataFrame,
    detector_name: str,
) -> dict[str, float | str | int]:
    bench = labels.merge(
        flag_by_row.rename("flag").reset_index().rename(columns={"index": "_row_id"}),
        on="_row_id",
        how="left",
    )
    bench["flag"] = bench["flag"].fillna(False).astype(bool)
    y = bench["is_injected"].astype(bool)
    pred = bench["flag"]
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    recall_n = tp + fn
    precision_n = tp + fp
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall_low, recall_high = wilson_ci(tp, recall_n)
    precision_low, precision_high = wilson_ci(tp, precision_n)

    latencies: list[float] = []
    for _, event in events.iterrows():
        event_rows = bench[(bench["event_id"] == event["event_id"]) & bench["is_injected"]].sort_values("timestamp_utc")
        flagged = event_rows[event_rows["flag"]]
        if flagged.empty:
            continue
        latency_min = (
            pd.Timestamp(flagged["timestamp_utc"].iloc[0]) - pd.Timestamp(event["start_time"])
        ).total_seconds() / 60.0
        latencies.append(float(latency_min))

    return {
        "detector": detector_name,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "recall": float(recall),
        "recall_ci95_low": recall_low,
        "recall_ci95_high": recall_high,
        "recall_n": int(recall_n),
        "precision": float(precision),
        "precision_ci95_low": precision_low,
        "precision_ci95_high": precision_high,
        "precision_n": int(precision_n),
        "median_latency_min": float(np.median(latencies)) if latencies else np.nan,
        "events_detected": int(len(latencies)),
        "events_total": int(len(events)),
    }


def write_tex_table(summary: pd.DataFrame, policy_context: pd.DataFrame, path: Path) -> None:
    def fmt_recall(sub: pd.DataFrame, detector: str) -> str:
        row = sub.loc[detector]
        return f"{row['recall']:.2f} [{row['recall_ci95_low']:.2f}, {row['recall_ci95_high']:.2f}]"

    rows = []
    for fault in FAULTS:
        sub = summary[summary["fault_type"].eq(fault.key)].set_index("detector")
        rows.append(
            {
                "Fault": fault.label,
                "Rule recall": fmt_recall(sub, "Rule evidence"),
                "Rule+spatial recall": fmt_recall(sub, "Rule + spatial evidence"),
                "P3 deletion": fmt_recall(sub, "P3 deletion action"),
                "Delete-all action": fmt_recall(sub, "Aggressive deletion action"),
                "Latency": sub.loc["Rule + spatial evidence", "median_latency_min"],
            }
        )
    table = pd.DataFrame(rows)
    lines = [
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Injected fault & Rule ev. recall [95\\% CI] & Rule+spatial ev. recall [95\\% CI] & P3 deletion recall [95\\% CI] & Delete-all action recall [95\\% CI] & Latency (min) \\\\",
        "\\midrule",
    ]
    for _, row in table.iterrows():
        latency = "--" if pd.isna(row["Latency"]) else f"{row['Latency']:.0f}"
        lines.append(
            f"{row['Fault']} & {row['Rule recall']} & {row['Rule+spatial recall']} & "
            f"{row['P3 deletion']} & {row['Delete-all action']} & {latency} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])

    p3_clean = float(policy_context.loc[policy_context["policy"].eq("P3"), "false_rejection_clean_test"].iloc[0])
    aggr_clean = float(policy_context.loc[policy_context["policy"].eq("Aggressive censoring"), "false_rejection_clean_test"].iloc[0])
    lines.append(
        f"% Clean TEST false rejection proxies: P3={p3_clean:.6f}, delete-all={aggr_clean:.6f}"
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def make_figure(summary: pd.DataFrame, path_pdf: Path, path_png: Path) -> None:
    plot = summary[summary["detector"].isin(["Rule evidence", "Rule + spatial evidence"])].copy()
    fault_order = [fault.key for fault in FAULTS]
    labels = {fault.key: fault.label for fault in FAULTS}
    x = np.arange(len(fault_order))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8.8, 4.1))
    for i, detector in enumerate(["Rule evidence", "Rule + spatial evidence"]):
        sub = plot[plot["detector"].eq(detector)].set_index("fault_type").reindex(fault_order)
        values = sub["recall"].to_numpy(dtype=float)
        bars = ax.bar(
            x + (i - 0.5) * width,
            values,
            width=width,
            label=detector,
            color="#5b8cc0" if detector == "Rule evidence" else "#009e73",
            edgecolor="black",
            linewidth=0.5,
        )
        for bar, value in zip(bars, values, strict=False):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                min(1.02, value + 0.025),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_ylim(0, 1.10)
    ax.set_ylabel("Row-level evidence recall")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[key] for key in fault_order], rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.17), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(path_pdf)
    fig.savefig(path_png, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_dir: Path = args.results_dir
    out_dir = results_dir / "measurement_fault_suite"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    p0 = load_p0(results_dir)
    p0 = p0[p0["timestamp_utc"] < TEST_END].copy()
    station_ids = sorted(p0["station_id"].unique().tolist())
    coords = load_coordinates(args.raw_parquet, station_ids)
    neighbors = nearest_neighbors(coords)
    pressure_medians, pressure_residual_threshold = pressure_residual_thresholds(p0, neighbors)

    train = p0[p0["timestamp_utc"] < TRAIN_END].copy()
    thresholds = pd.DataFrame(
        {
            "wind_u90": train.groupby("station_id")["wind_gust"].quantile(0.90),
            "wind_u99": train.groupby("station_id")["wind_gust"].quantile(0.99),
            "wind_delta_q995": (train["wind_gust"] - train["wind_mean"]).groupby(train["station_id"]).quantile(0.995),
        }
    )
    thresholds["wind_delta_q995"] = thresholds["wind_delta_q995"].clip(lower=30.0)

    test_dirty = p0[(p0["timestamp_utc"] >= TRAIN_END) & (p0["stratum"].eq("dirty"))].copy()
    test_dirty = test_dirty.sort_values(["station_id", "timestamp_utc"]).reset_index(drop=True)

    summary_rows: list[dict[str, object]] = []
    all_events: list[pd.DataFrame] = []
    all_labels: list[pd.DataFrame] = []

    for spec in FAULTS:
        events = choose_events(test_dirty, spec, rng)
        if events.empty:
            raise RuntimeError(f"No events selected for {spec.key}")
        faulted, labels = apply_faults(test_dirty, events, spec, thresholds, rng)
        fail, suspect = rule_flags(faulted)
        pressure_spatial, wind_spatial = spatial_flags(
            faulted,
            thresholds,
            pressure_medians,
            pressure_residual_threshold,
            neighbors,
        )
        spatial = pressure_spatial if spec.spatial_family == "pressure" else wind_spatial

        flag_map = {
            "Rule evidence": fail | suspect,
            "Rule + spatial evidence": fail | suspect | spatial,
            "P3 deletion action": fail,
            "Aggressive deletion action": fail | suspect,
        }
        for detector, flags in flag_map.items():
            by_row = pd.Series(flags.to_numpy(dtype=bool), index=faulted["_row_id"].astype(int).to_numpy())
            row = metrics_for_detector(labels, by_row, events, detector)
            row.update(
                {
                    "fault_type": spec.key,
                    "fault_label": spec.label,
                    "n_injected_rows": int(labels["is_injected"].sum()),
                    "n_benchmark_rows": int(len(labels)),
                    "pressure_spatial_threshold_hpa": float(pressure_residual_threshold),
                }
            )
            summary_rows.append(row)

        all_events.append(events)
        all_labels.append(labels)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "measurement_fault_suite_summary.csv", index=False)
    pd.concat(all_events, ignore_index=True).to_csv(out_dir / "measurement_fault_suite_events.csv", index=False)
    pd.concat(all_labels, ignore_index=True).to_csv(out_dir / "measurement_fault_suite_labels.csv", index=False)

    missing = pd.read_csv(results_dir / "metrics_missingness.csv")
    context_rows = []
    for product, label in [("P3", "P3"), ("P1", "Aggressive censoring"), ("P4", "Fixed domain-rule QC")]:
        row = missing.query(
            "product == @product and phase == 'test' and scope == 'clean' and variable == 'wind_gust'"
        )
        context_rows.append(
            {
                "policy": label,
                "false_rejection_clean_test": float(row["missing_additional"].iloc[0]) if not row.empty else np.nan,
            }
        )
    policy_context = pd.DataFrame(context_rows)
    policy_context.to_csv(out_dir / "measurement_fault_suite_policy_context.csv", index=False)

    tex_path = out_dir / "tab_measurement_fault_suite.tex"
    write_tex_table(summary, policy_context, tex_path)
    fig_pdf = out_dir / "F8_measurement_fault_suite.pdf"
    fig_png = out_dir / "F8_measurement_fault_suite.png"
    make_figure(summary, fig_pdf, fig_png)

    article_dir: Path = args.article_dir
    article_dir.mkdir(parents=True, exist_ok=True)
    (article_dir / "tab_measurement_fault_suite.tex").write_text(tex_path.read_text(encoding="utf-8"), encoding="utf-8")
    (article_dir / "F8_measurement_fault_suite.pdf").write_bytes(fig_pdf.read_bytes())
    print(f"Wrote {out_dir / 'measurement_fault_suite_summary.csv'}")
    print(f"Wrote {article_dir / 'tab_measurement_fault_suite.tex'}")
    print(f"Wrote {article_dir / 'F8_measurement_fault_suite.pdf'}")
    print(f"Pressure spatial residual threshold hPa: {pressure_residual_threshold:.3f}")


if __name__ == "__main__":
    main()
