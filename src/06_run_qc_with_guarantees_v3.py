from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")


START_UTC = "2023-01-01 00:00:00"
END_UTC = "2026-01-01 00:00:00"
TRAIN_START = "2023-01-01 00:00:00"
TRAIN_END = "2025-01-01 00:00:00"
TEST_START = "2025-01-01 00:00:00"
TEST_END = "2026-01-01 00:00:00"

B_SET = (0.05, 0.10)
EPSILON = 0.005
TAU = 0.10
K_TOP = 20
SEED_GLOBAL = 20260218
SEED_STATION_SPLIT = 20260218
SEED_MODEL = 20260218
SEED_SYNTHETIC = 20260218
UTILITY_V3_LAMBDA = 0.25
UTILITY_V3_ETA = 0.05
BENCHMARK_RECALL_MIN = 0.95
BENCHMARK_PRECISION_MIN = 0.90
BENCHMARK_STATIONS_DIRTY = 5
BENCHMARK_DURATIONS_H = (3, 6, 12, 24)
BENCHMARK_INJECTION_TOL_HPA = 0.05
C6_ENABLED = True
C2V2_SPREAD_RATIO_R = 10.0
C2V2_MISSING_RATIO_M = 10.0
C2V2_REMOVED_KEPT_DELTA_MIN_TRAIN = 0.05
C2V2_REMOVED_KEPT_DELTA_MIN_TEST = 0.02
C2V2_SPREAD_MIN = 0.01

T_GRID = (3, 6, 12, 24, 48)
TOL_GRID = (0.05, 0.10, 0.20)
POLICY_GRID = ("KEEP", "DROP")

T_REF = 3
TOL_REF = 0.10


@dataclass
class Paths:
    root: Path
    results: Path
    figures: Path
    downstream: Path
    synthetic: Path
    benchmark: Path


def _json_hash(payload: dict[str, Any]) -> str:
    b = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(f"ABORT: {msg}")


def _ensure_dirs(base: Path) -> Paths:
    results = base / "results_v3"
    figures = results / "figures"
    downstream = results / "downstream"
    synthetic = results / "synthetic"
    benchmark = results / "benchmark_stuck"
    for p in [results, figures, downstream, synthetic, benchmark, results / "products"]:
        p.mkdir(parents=True, exist_ok=True)
    for name in ["P0", "P1", "P2", "P3", "P4", "P5"]:
        (results / "products" / name).mkdir(parents=True, exist_ok=True)
    return Paths(root=base, results=results, figures=figures, downstream=downstream, synthetic=synthetic, benchmark=benchmark)


def _budget_suffix(B: float) -> str:
    return f"{int(round(float(B) * 1000.0)):03d}"


def _budget_best_filename(B: float) -> str:
    return f"budget_best_B{_budget_suffix(B)}_v3.json"


def _write_preregister(paths: Paths, gamma_value: float) -> None:
    payload = {
        "version": "v3",
        "START": START_UTC,
        "END": END_UTC,
        "TRAIN": [TRAIN_START, TRAIN_END],
        "TEST": [TEST_START, TEST_END],
        "B_SET": list(B_SET),
        "epsilon": EPSILON,
        "tau": TAU,
        "k_top": K_TOP,
        "seeds": {
            "seed_global": SEED_GLOBAL,
            "seed_station_split": SEED_STATION_SPLIT,
            "seed_model": SEED_MODEL,
            "seed_synthetic": SEED_SYNTHETIC,
        },
        "grids": {
            "T_hours": list(T_GRID),
            "tol_hpa": list(TOL_GRID),
            "policy_suspect": list(POLICY_GRID),
            "config_ref": {"T_ref": T_REF, "tol_ref": TOL_REF},
            "downstream_alpha_grid": [0.01, 0.1, 1.0, 10.0, 100.0],
        },
        "domain_rule_baseline": {
            "product": "P5",
            "policy": "domain_rule_qc",
            "description": "Fixed, non-optimized rule-based meteorological QC baseline.",
            "drop_fail": True,
            "drop_suspect": True,
            "unit_censor": False,
            "T_hours": T_REF,
            "tol_hpa": TOL_REF,
        },
        "utility_v3": {
            "definition": "utility_v3 = -state_wind_spread - lambda*missing_additional - eta*suspect_rate",
            "lambda": UTILITY_V3_LAMBDA,
            "eta": UTILITY_V3_ETA,
            "selection_scope": "dirty",
            "selection_phase": "train",
        },
        "benchmark_stuck": {
            "enabled": bool(C6_ENABLED),
            "seed_synthetic": SEED_SYNTHETIC,
            "dirty_station_count": BENCHMARK_STATIONS_DIRTY,
            "durations_hours": list(BENCHMARK_DURATIONS_H),
            "injection_tol_hpa": BENCHMARK_INJECTION_TOL_HPA,
            "constraints": {
                "recall_min": BENCHMARK_RECALL_MIN,
                "precision_min": BENCHMARK_PRECISION_MIN,
            },
        },
        "quality_weights": {
            "q_definition": {"FAIL": 0.0, "SUSPECT": 0.5, "PASS": 1.0},
            "fields": ["q_wind_gust", "q_pressure"],
        },
        "downstream_cost_sensitive": {
            "formula": "score = RMSE_operational + gamma*(1-coverage_operational)",
            "gamma": gamma_value,
            "gamma_source": "P0 TRAIN ridge RMSE_operational",
            "evaluation_scope_for_C5": "global",
        },
        "c2v2_thresholds": {
            "spread_ratio_r": C2V2_SPREAD_RATIO_R,
            "missing_ratio_m": C2V2_MISSING_RATIO_M,
            "removed_vs_kept_delta_min_train": C2V2_REMOVED_KEPT_DELTA_MIN_TRAIN,
            "removed_vs_kept_delta_min_test": C2V2_REMOVED_KEPT_DELTA_MIN_TEST,
            "spread_min": C2V2_SPREAD_MIN,
            "required_subchecks": ["C2v2.1", "C2v2.2"],
        },
        "fixed_rules": {
            "dedup_key": ["station_id", "timestamp_utc"],
            "availability_threshold": 0.80,
            "anti_concentration_max_share": 0.45,
            "ci_config_ref": {"T_ref": T_REF, "tol_ref": TOL_REF},
        },
    }
    (paths.results / "preregister_v3.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_dedup_table(con: duckdb.DuckDBPyConnection, input_parquet: str) -> None:
    con.execute("DROP TABLE IF EXISTS raw_window")
    con.execute("DROP TABLE IF EXISTS dedup_window")

    q_raw = f"""
        CREATE TABLE raw_window AS
        SELECT
          ROW_NUMBER() OVER () AS ingest_order,
          station_id,
          COALESCE(TRY_CAST(json_extract_string(payload_json, '$."data ini"') AS TIMESTAMP), TRY_CAST(obs_time AS TIMESTAMP)) AS timestamp_utc,
          TRY_CAST(REPLACE(NULLIF(TRIM(json_extract_string(payload_json, '$."pres mit_mit"')), 'None'), ',', '.') AS DOUBLE) AS pressure_raw,
          TRY_CAST(REPLACE(NULLIF(TRIM(json_extract_string(payload_json, '$."vent vel_mit"')), 'None'), ',', '.') AS DOUBLE) AS wind_mean_raw,
          TRY_CAST(REPLACE(NULLIF(TRIM(json_extract_string(payload_json, '$."vent vel_max"')), 'None'), ',', '.') AS DOUBLE) AS wind_gust_raw,
          TRY_CAST(REPLACE(NULLIF(TRIM(json_extract_string(payload_json, '$."vent gra_mit"')), 'None'), ',', '.') AS DOUBLE) AS wind_dir_raw
        FROM read_parquet('{input_parquet}')
        WHERE COALESCE(TRY_CAST(json_extract_string(payload_json, '$."data ini"') AS TIMESTAMP), TRY_CAST(obs_time AS TIMESTAMP)) >= TIMESTAMP '{START_UTC}'
          AND COALESCE(TRY_CAST(json_extract_string(payload_json, '$."data ini"') AS TIMESTAMP), TRY_CAST(obs_time AS TIMESTAMP)) < TIMESTAMP '{END_UTC}'
          AND station_id IS NOT NULL
    """
    con.execute(q_raw)

    q_dedup = """
        CREATE TABLE dedup_window AS
        SELECT
          station_id,
          timestamp_utc,
          COUNT(*) AS dup_count,

          COUNT(*) AS dup_audit_pressure_n_in,
          COUNT(pressure_raw) AS dup_audit_pressure_n_nonnull,
          arg_max(pressure_raw, CASE WHEN pressure_raw IS NOT NULL THEN ingest_order ELSE NULL END) AS pressure_raw,

          COUNT(*) AS dup_audit_wind_mean_n_in,
          COUNT(wind_mean_raw) AS dup_audit_wind_mean_n_nonnull,
          AVG(wind_mean_raw) AS wind_mean,

          COUNT(*) AS dup_audit_wind_gust_n_in,
          COUNT(wind_gust_raw) AS dup_audit_wind_gust_n_nonnull,
          MAX(wind_gust_raw) AS wind_gust,

          COUNT(*) AS dup_audit_wind_dir_n_in,
          COUNT(wind_dir_raw) AS dup_audit_wind_dir_n_nonnull,
          arg_max(wind_dir_raw, CASE WHEN wind_dir_raw IS NOT NULL THEN ingest_order ELSE NULL END) AS wind_dir
        FROM raw_window
        WHERE timestamp_utc IS NOT NULL
        GROUP BY station_id, timestamp_utc
    """
    con.execute(q_dedup)


def _station_availability(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    q = """
        SELECT
          station_id,
          COUNT(*) AS n_total,
          COUNT(pressure_raw) AS n_pressure,
          COUNT(wind_mean) AS n_wind_mean,
          COUNT(wind_gust) AS n_wind_gust,
          COUNT(wind_dir) AS n_wind_dir,
          COUNT(CASE WHEN timestamp_utc >= TIMESTAMP '2023-01-01 00:00:00' AND timestamp_utc < TIMESTAMP '2025-01-01 00:00:00' THEN 1 END) AS n_train,
          COUNT(CASE WHEN timestamp_utc >= TIMESTAMP '2025-01-01 00:00:00' AND timestamp_utc < TIMESTAMP '2026-01-01 00:00:00' THEN 1 END) AS n_test
        FROM dedup_window
        GROUP BY station_id
    """
    av = con.execute(q).df()
    for col in ["n_pressure", "n_wind_mean", "n_wind_gust"]:
        av[f"availability_{col}"] = av[col] / av["n_total"]
    av["availability_ratio"] = av[["availability_n_pressure", "availability_n_wind_mean", "availability_n_wind_gust"]].min(axis=1)
    return av


def _unit_station_train(con: duckdb.DuckDBPyConnection, eligible_ids: list[str]) -> pd.DataFrame:
    _require(len(eligible_ids) > 0, "No eligible stations to classify pressure units.")
    con.register("eligible_ids", pd.DataFrame({"station_id": eligible_ids}))
    q = """
        SELECT
          d.station_id,
          QUANTILE_CONT(d.pressure_raw, 0.05) FILTER (WHERE d.pressure_raw IS NOT NULL) AS p5,
          QUANTILE_CONT(d.pressure_raw, 0.50) FILTER (WHERE d.pressure_raw IS NOT NULL) AS p50,
          QUANTILE_CONT(d.pressure_raw, 0.95) FILTER (WHERE d.pressure_raw IS NOT NULL) AS p95
        FROM dedup_window d
        INNER JOIN eligible_ids e USING (station_id)
        WHERE d.timestamp_utc >= TIMESTAMP '2023-01-01 00:00:00'
          AND d.timestamp_utc < TIMESTAMP '2025-01-01 00:00:00'
        GROUP BY d.station_id
    """
    u = con.execute(q).df()

    def classify(row: pd.Series) -> str:
        p5, p50, p95 = row["p5"], row["p50"], row["p95"]
        if pd.notna(p50) and pd.notna(p95) and (80000 <= p50 <= 120000) and (p95 > 90000):
            return "Pa"
        if pd.notna(p5) and pd.notna(p50) and pd.notna(p95) and (600 <= p50 <= 820) and (p95 < 900) and (p5 > 400):
            return "mmHg"
        return "hPa"

    u["flag_unit_station"] = u.apply(classify, axis=1)
    return u[["station_id", "flag_unit_station", "p5", "p50", "p95"]].copy()


def _build_canonical_eligible(con: duckdb.DuckDBPyConnection, eligible: pd.DataFrame, units: pd.DataFrame) -> None:
    con.register("eligible_ids", eligible[["station_id"]])
    con.register("unit_map", units[["station_id", "flag_unit_station"]])

    con.execute("DROP TABLE IF EXISTS canonical_eligible")
    q = """
        CREATE TABLE canonical_eligible AS
        SELECT
          d.station_id,
          d.timestamp_utc,

          d.pressure_raw,
          CASE
            WHEN u.flag_unit_station = 'Pa' THEN d.pressure_raw / 100.0
            WHEN u.flag_unit_station = 'mmHg' THEN d.pressure_raw * 1.3332239
            ELSE d.pressure_raw
          END AS pressure_hpa,

          u.flag_unit_station,
          CASE
            WHEN u.flag_unit_station <> 'hPa' AND d.pressure_raw IS NOT NULL THEN 1
            ELSE 0
          END AS flag_unit_converted,

          d.wind_mean,
          d.wind_gust,
          d.wind_dir,

          d.dup_count,
          d.dup_audit_pressure_n_in,
          d.dup_audit_pressure_n_nonnull,
          d.dup_audit_wind_mean_n_in,
          d.dup_audit_wind_mean_n_nonnull,
          d.dup_audit_wind_gust_n_in,
          d.dup_audit_wind_gust_n_nonnull,
          d.dup_audit_wind_dir_n_in,
          d.dup_audit_wind_dir_n_nonnull,

          CASE
            WHEN d.pressure_raw IS NOT NULL AND
                 (
                   CASE
                     WHEN u.flag_unit_station = 'Pa' THEN d.pressure_raw / 100.0
                     WHEN u.flag_unit_station = 'mmHg' THEN d.pressure_raw * 1.3332239
                     ELSE d.pressure_raw
                   END
                 ) < 850 THEN 1
            WHEN d.pressure_raw IS NOT NULL AND
                 (
                   CASE
                     WHEN u.flag_unit_station = 'Pa' THEN d.pressure_raw / 100.0
                     WHEN u.flag_unit_station = 'mmHg' THEN d.pressure_raw * 1.3332239
                     ELSE d.pressure_raw
                   END
                 ) > 1100 THEN 1
            ELSE 0
          END AS pressure_fail_out_range_after_norm,

          CASE WHEN d.pressure_raw = 0 THEN 1 ELSE 0 END AS pressure_fail_zero,

          CASE WHEN d.wind_mean IS NOT NULL AND d.wind_gust IS NOT NULL AND d.wind_mean > d.wind_gust + 1e-9 THEN 1 ELSE 0 END AS fail_wind_incoherent,
          CASE WHEN (d.wind_mean IS NOT NULL AND d.wind_mean < 0) OR (d.wind_gust IS NOT NULL AND d.wind_gust < 0) THEN 1 ELSE 0 END AS fail_wind_negative,
          CASE WHEN d.wind_dir = 999 THEN 1 ELSE 0 END AS fail_wind_dir_999,
          CASE WHEN d.wind_dir IS NOT NULL AND (d.wind_dir < 0 OR d.wind_dir > 360) THEN 1 ELSE 0 END AS fail_wind_dir_range
        FROM dedup_window d
        INNER JOIN eligible_ids e USING (station_id)
        INNER JOIN unit_map u USING (station_id)
    """
    con.execute(q)


def _write_phaseA_outputs(con: duckdb.DuckDBPyConnection, paths: Paths) -> None:
    p0_path = paths.results / "P0_raw_canon.parquet"
    flags_path = paths.results / "flags_base.parquet"

    con.execute(
        f"""
        COPY (
          SELECT
            station_id,
            timestamp_utc AS timestamp_utc,
            pressure_raw,
            pressure_hpa,
            flag_unit_station,
            flag_unit_converted,
            wind_mean,
            wind_gust,
            wind_dir,
            dup_count,
            dup_audit_pressure_n_in,
            dup_audit_pressure_n_nonnull,
            dup_audit_wind_mean_n_in,
            dup_audit_wind_mean_n_nonnull,
            dup_audit_wind_gust_n_in,
            dup_audit_wind_gust_n_nonnull,
            dup_audit_wind_dir_n_in,
            dup_audit_wind_dir_n_nonnull,
            pressure_fail_out_range_after_norm,
            pressure_fail_zero
          FROM canonical_eligible
        ) TO '{p0_path.as_posix()}' (FORMAT PARQUET)
        """
    )

    con.execute(
        f"""
        COPY (
          SELECT
            station_id,
            timestamp_utc AS timestamp_utc,
            fail_wind_incoherent,
            fail_wind_negative,
            fail_wind_dir_999,
            fail_wind_dir_range,
            pressure_fail_out_range_after_norm,
            pressure_fail_zero,
            flag_unit_station,
            flag_unit_converted
          FROM canonical_eligible
        ) TO '{flags_path.as_posix()}' (FORMAT PARQUET)
        """
    )


def _compute_stuck_counts_train_sql(con: duckdb.DuckDBPyConnection, tol_hpa: float, T_hours: int) -> pd.DataFrame:
    q = f"""
        WITH train_rows AS (
          SELECT station_id, timestamp_utc, pressure_hpa
          FROM canonical_eligible
          WHERE timestamp_utc >= TIMESTAMP '{TRAIN_START}'
            AND timestamp_utc < TIMESTAMP '{TRAIN_END}'
            AND pressure_hpa IS NOT NULL
        ),
        deltas AS (
          SELECT
            station_id,
            DATE_DIFF('second', LAG(timestamp_utc) OVER (PARTITION BY station_id ORDER BY timestamp_utc), timestamp_utc) AS delta_s
          FROM train_rows
        ),
        med AS (
          SELECT
            station_id,
            COALESCE(QUANTILE_CONT(delta_s, 0.5) FILTER (WHERE delta_s IS NOT NULL AND delta_s > 0), 600.0) AS med_delta_s
          FROM deltas
          GROUP BY station_id
        ),
        seq AS (
          SELECT
            t.station_id,
            t.timestamp_utc,
            t.pressure_hpa,
            m.med_delta_s,
            LAG(t.timestamp_utc) OVER (PARTITION BY t.station_id ORDER BY t.timestamp_utc) AS prev_ts,
            LAG(t.pressure_hpa) OVER (PARTITION BY t.station_id ORDER BY t.timestamp_utc) AS prev_pressure
          FROM train_rows t
          INNER JOIN med m USING (station_id)
        ),
        run_marks AS (
          SELECT
            *,
            CASE
              WHEN prev_ts IS NULL THEN 1
              WHEN DATE_DIFF('second', prev_ts, timestamp_utc) > 2.0 * med_delta_s THEN 1
              WHEN ABS(pressure_hpa - prev_pressure) > {tol_hpa} THEN 1
              ELSE 0
            END AS run_break
          FROM seq
        ),
        run_ids AS (
          SELECT
            *,
            SUM(run_break) OVER (PARTITION BY station_id ORDER BY timestamp_utc ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS run_id
          FROM run_marks
        ),
        run_stats AS (
          SELECT
            station_id,
            run_id,
            MIN(timestamp_utc) AS min_ts,
            MAX(timestamp_utc) AS max_ts,
            DATE_DIFF('second', MIN(timestamp_utc), MAX(timestamp_utc)) / 3600.0 AS run_hours
          FROM run_ids
          GROUP BY station_id, run_id
        ),
        stuck_rows AS (
          SELECT r.station_id, r.timestamp_utc
          FROM run_ids r
          INNER JOIN run_stats s USING (station_id, run_id)
          WHERE s.run_hours >= {T_hours}
        )
        SELECT station_id, COUNT(*) AS suspect_pressure_stuck_count
        FROM stuck_rows
        GROUP BY station_id
    """
    return con.execute(q).df()


def _compute_ci_and_select(
    con: duckdb.DuckDBPyConnection,
    eligible: pd.DataFrame,
    units: pd.DataFrame,
    paths: Paths,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    stuck_ref = _compute_stuck_counts_train_sql(con, tol_hpa=TOL_REF, T_hours=T_REF)

    q_train_counts = f"""
        SELECT
          station_id,
          COUNT(*) FILTER (WHERE (pressure_hpa IS NOT NULL OR wind_gust IS NOT NULL)) AS N_nonnull_rows_pressure_or_windgust,
          SUM(pressure_fail_zero + pressure_fail_out_range_after_norm + fail_wind_incoherent + fail_wind_negative) AS hard_count,
          SUM(flag_unit_converted) AS converted_count
        FROM canonical_eligible
        WHERE timestamp_utc >= TIMESTAMP '{TRAIN_START}'
          AND timestamp_utc < TIMESTAMP '{TRAIN_END}'
        GROUP BY station_id
    """
    tr = con.execute(q_train_counts).df()
    ci = tr.merge(stuck_ref, on="station_id", how="left")
    ci["suspect_pressure_stuck_count"] = ci["suspect_pressure_stuck_count"].fillna(0.0)
    ci["soft_count"] = ci["suspect_pressure_stuck_count"] + ci["converted_count"]

    _require((ci["N_nonnull_rows_pressure_or_windgust"] > 0).all(), "Some stations have zero denominator for CI_total.")

    ci["ci_total"] = (
        10000.0 * ci["hard_count"] / ci["N_nonnull_rows_pressure_or_windgust"]
        + 0.5 * 10000.0 * ci["soft_count"] / ci["N_nonnull_rows_pressure_or_windgust"]
    )

    # Global CI_total and excl_top1 by hard_count.
    top1_station = ci.sort_values(["hard_count", "station_id"], ascending=[False, True]).iloc[0]["station_id"]
    glob_hard = float(ci["hard_count"].sum())
    glob_soft = float(ci["soft_count"].sum())
    glob_n = float(ci["N_nonnull_rows_pressure_or_windgust"].sum())
    ci_global = 10000.0 * glob_hard / max(1.0, glob_n) + 0.5 * 10000.0 * glob_soft / max(1.0, glob_n)

    ci_ex = ci[ci["station_id"] != top1_station].copy()
    ex_hard = float(ci_ex["hard_count"].sum())
    ex_soft = float(ci_ex["soft_count"].sum())
    ex_n = float(ci_ex["N_nonnull_rows_pressure_or_windgust"].sum())
    ci_excl_top1 = 10000.0 * ex_hard / max(1.0, ex_n) + 0.5 * 10000.0 * ex_soft / max(1.0, ex_n)

    ci = ci.sort_values(["ci_total", "station_id"], ascending=[True, True]).reset_index(drop=True)
    ci["pct_rank"] = 100.0 * (np.arange(1, ci.shape[0] + 1) / ci.shape[0])

    clean_pool = ci[(ci["pct_rank"] >= 10.0) & (ci["pct_rank"] <= 30.0)].copy()
    medium_pool = ci[(ci["pct_rank"] >= 40.0) & (ci["pct_rank"] <= 60.0)].copy()
    dirty_pool = ci[(ci["pct_rank"] >= 80.0) & (ci["pct_rank"] <= 99.0)].copy()

    _require(clean_pool.shape[0] >= 10, "Not enough stations in CLEAN percentile band 10-30.")
    _require(medium_pool.shape[0] >= 10, "Not enough stations in MEDIUM percentile band 40-60.")
    _require(dirty_pool.shape[0] >= 10, "Not enough stations in DIRTY percentile band 80-99.")

    clean_sel = clean_pool.sort_values(["ci_total", "station_id"], ascending=[True, True]).head(10).copy()
    medium_sel = medium_pool.sort_values(["ci_total", "station_id"], ascending=[True, True]).head(10).copy()
    dirty_sel = dirty_pool.sort_values(["ci_total", "station_id"], ascending=[False, True]).head(10).copy()

    case_rows: list[dict[str, Any]] = []
    dirty_candidates = dirty_pool.sort_values(["ci_total", "station_id"], ascending=[False, True]).copy()

    def max_share(df_dirty: pd.DataFrame) -> float:
        total = float(df_dirty["hard_count"].sum())
        if total <= 0:
            return 0.0
        return float(df_dirty["hard_count"].max() / total)

    while max_share(dirty_sel) > 0.45:
        offender = dirty_sel.sort_values(["hard_count", "station_id"], ascending=[False, True]).iloc[0]
        offender_id = str(offender["station_id"])

        replacement_pool = dirty_candidates[
            (~dirty_candidates["station_id"].isin(dirty_sel["station_id"]))
            & (~dirty_candidates["station_id"].isin([r["station_id"] for r in case_rows]))
            & (dirty_candidates["station_id"] != offender_id)
        ].copy()

        replacement = None
        for _, cand in replacement_pool.iterrows():
            trial = dirty_sel[dirty_sel["station_id"] != offender_id].copy()
            trial = pd.concat([trial, cand.to_frame().T], ignore_index=True)
            if max_share(trial) <= 0.45:
                replacement = cand
                dirty_sel = trial
                break

        case_rows.append(
            {
                "station_id": offender_id,
                "reason": "anti_concentration_hard_share_gt_0.45",
                "hard_count": float(offender["hard_count"]),
                "ci_total": float(offender["ci_total"]),
                "replacement_station_id": None if replacement is None else str(replacement["station_id"]),
            }
        )

        _require(replacement is not None, "Anti-concentration failed: no replacement candidate available in DIRTY percentile band.")

    clean_sel["stratum"] = "clean"
    medium_sel["stratum"] = "medium"
    dirty_sel["stratum"] = "dirty"
    selected = pd.concat([clean_sel, medium_sel, dirty_sel], ignore_index=True)
    selected = selected.sort_values(["stratum", "station_id"]).reset_index(drop=True)

    _require(selected.shape[0] == 30, f"Selected station count is {selected.shape[0]}, expected 30.")
    _require((selected.groupby("stratum").size().to_dict().get("clean", 0) == 10), "Selected CLEAN != 10.")
    _require((selected.groupby("stratum").size().to_dict().get("medium", 0) == 10), "Selected MEDIUM != 10.")
    _require((selected.groupby("stratum").size().to_dict().get("dirty", 0) == 10), "Selected DIRTY != 10.")

    # Save outputs phase C
    selected.to_csv(paths.results / "selected_stations.csv", index=False)
    pd.DataFrame(case_rows).to_csv(paths.results / "case_stations.csv", index=False)

    dirty_total_hard = float(selected[selected["stratum"] == "dirty"]["hard_count"].sum())
    if dirty_total_hard > 0:
        dirty_share = (
            selected[selected["stratum"] == "dirty"]["hard_count"].max() / dirty_total_hard
        )
    else:
        dirty_share = 0.0

    report = {
        "ci_total_global": ci_global,
        "ci_total_excl_top1": ci_excl_top1,
        "top1_hard_station_global": str(top1_station),
        "percentile_pools": {
            "clean_n": int(clean_pool.shape[0]),
            "medium_n": int(medium_pool.shape[0]),
            "dirty_n": int(dirty_pool.shape[0]),
        },
        "selected_counts": selected.groupby("stratum").size().to_dict(),
        "dirty_max_hard_share": float(dirty_share),
        "anti_concentration_pass": bool(dirty_share <= 0.45),
    }
    (paths.results / "selection_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    return selected, pd.DataFrame(case_rows), report


def _load_selected_canonical(con: duckdb.DuckDBPyConnection, selected: pd.DataFrame) -> pd.DataFrame:
    con.register("sel_ids", selected[["station_id"]])
    q = """
        SELECT c.*
        FROM canonical_eligible c
        INNER JOIN sel_ids s USING (station_id)
        ORDER BY c.station_id, c.timestamp_utc
    """
    df = con.execute(q).df()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    stratum_map = dict(zip(selected["station_id"], selected["stratum"], strict=False))
    df["stratum"] = df["station_id"].map(stratum_map)
    return df


def _compute_station_med_delta_train(df: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    train_mask = (df["timestamp_utc"] >= pd.Timestamp(TRAIN_START, tz="UTC")) & (df["timestamp_utc"] < pd.Timestamp(TRAIN_END, tz="UTC"))
    for station, st in df[train_mask].groupby("station_id", sort=False):
        ts = st["timestamp_utc"].sort_values()
        d = ts.diff().dt.total_seconds().dropna()
        d = d[d > 0]
        if d.empty:
            out[str(station)] = 600.0
        else:
            out[str(station)] = float(np.nanmedian(d.to_numpy(dtype=float)))
    return out


def _compute_stuck_flags(df: pd.DataFrame, T_hours: int, tol_hpa: float, med_delta_train_s: dict[str, float]) -> pd.Series:
    out = pd.Series(False, index=df.index)
    for station, idx in df.groupby("station_id", sort=False).groups.items():
        i = np.array(list(idx), dtype=int)
        st = df.loc[i, ["timestamp_utc", "pressure_hpa"]].copy().sort_values("timestamp_utc")
        med = float(med_delta_train_s.get(str(station), 600.0))

        prev_ts = st["timestamp_utc"].shift(1)
        prev_p = st["pressure_hpa"].shift(1)
        dt = (st["timestamp_utc"] - prev_ts).dt.total_seconds()

        both_ok = st["pressure_hpa"].notna() & prev_p.notna()
        break_gap = prev_ts.isna() | (dt > (2.0 * med))
        break_missing = ~both_ok
        break_tol = both_ok & (st["pressure_hpa"].sub(prev_p).abs() > tol_hpa)
        run_break = break_gap | break_missing | break_tol

        run_id = run_break.cumsum()
        tmp = st.copy()
        tmp["run_id"] = run_id.to_numpy()
        grp = tmp.groupby("run_id", sort=False)
        run_min = grp["timestamp_utc"].transform("min")
        run_max = grp["timestamp_utc"].transform("max")
        run_hours = (run_max - run_min).dt.total_seconds().div(3600.0)
        stuck = st["pressure_hpa"].notna() & (run_hours >= T_hours)

        out.loc[st.index] = stuck.to_numpy()
    return out


def _add_stuck_grid(df: pd.DataFrame, med_delta_train_s: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    for T in T_GRID:
        for tol in TOL_GRID:
            col = f"suspect_stuck_T{T}_tol{tol:.2f}"
            out[col] = _compute_stuck_flags(out, T_hours=T, tol_hpa=float(tol), med_delta_train_s=med_delta_train_s)
    return out


def _build_stuck_benchmark(
    paths: Paths,
    base: pd.DataFrame,
    selected: pd.DataFrame,
    med_delta_train_s: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = base.copy()
    rng = np.random.default_rng(SEED_SYNTHETIC)
    dirty = sorted(selected[selected["stratum"] == "dirty"]["station_id"].tolist())
    _require(len(dirty) >= BENCHMARK_STATIONS_DIRTY, "Not enough DIRTY stations for stuck benchmark.")
    chosen = sorted(rng.choice(np.array(dirty, dtype=object), size=BENCHMARK_STATIONS_DIRTY, replace=False).tolist())

    phase_windows = {
        "train": (pd.Timestamp(TRAIN_START, tz="UTC"), pd.Timestamp(TRAIN_END, tz="UTC")),
        "test": (pd.Timestamp(TEST_START, tz="UTC"), pd.Timestamp(TEST_END, tz="UTC")),
    }

    events: list[dict[str, Any]] = []
    bench_rows_idx: set[int] = set()
    injected_idx: set[int] = set()
    event_records: list[dict[str, Any]] = []

    for station in chosen:
        st_all = work[work["station_id"] == station].sort_values("timestamp_utc").copy()
        _require(not st_all.empty, f"Station {station} has no rows in base for benchmark.")
        med_s = float(med_delta_train_s.get(station, 600.0))
        for phase, (t0, t1) in phase_windows.items():
            st = st_all[(st_all["timestamp_utc"] >= t0) & (st_all["timestamp_utc"] < t1)].copy()
            _require(not st.empty, f"No rows for benchmark station={station}, phase={phase}.")
            st = st.reset_index().rename(columns={"index": "base_index"})
            used_local: set[int] = set()

            for dur_h in BENCHMARK_DURATIONS_H:
                candidates = list(range(st.shape[0]))
                rng.shuffle(candidates)
                found = None

                for pos in candidates:
                    if pos in used_local:
                        continue
                    row0 = st.iloc[pos]
                    if pd.isna(row0["pressure_hpa"]):
                        continue
                    if int(row0["pressure_fail_zero"]) == 1 or int(row0["pressure_fail_out_range_after_norm"]) == 1:
                        continue

                    run_positions = [pos]
                    k = pos
                    while True:
                        k_next = k + 1
                        if k_next >= st.shape[0]:
                            break
                        dt = (st.iloc[k_next]["timestamp_utc"] - st.iloc[k]["timestamp_utc"]).total_seconds()
                        if dt > (2.0 * med_s):
                            break
                        if pd.isna(st.iloc[k_next]["pressure_hpa"]):
                            break
                        if int(st.iloc[k_next]["pressure_fail_zero"]) == 1 or int(st.iloc[k_next]["pressure_fail_out_range_after_norm"]) == 1:
                            break
                        run_positions.append(k_next)
                        k = k_next
                        run_hours = (st.iloc[k]["timestamp_utc"] - st.iloc[pos]["timestamp_utc"]).total_seconds() / 3600.0
                        if run_hours >= dur_h:
                            break

                    run_hours = (st.iloc[run_positions[-1]]["timestamp_utc"] - st.iloc[run_positions[0]]["timestamp_utc"]).total_seconds() / 3600.0
                    if run_hours < dur_h:
                        continue
                    if any((rp in used_local) for rp in run_positions):
                        continue

                    found = run_positions
                    break

                _require(found is not None, f"Benchmark injection failed for station={station}, phase={phase}, dur={dur_h}h.")

                event_id = len(events)
                start_ts = st.iloc[found[0]]["timestamp_utc"]
                end_ts = st.iloc[found[-1]]["timestamp_utc"]
                base_indices = st.iloc[found]["base_index"].astype(int).tolist()
                base_value = float(st.iloc[found[0]]["pressure_hpa"])

                for ii, bi in enumerate(base_indices):
                    jitter = ((ii % 3) - 1) * 0.005
                    work.loc[bi, "pressure_hpa"] = base_value + jitter
                    injected_idx.add(int(bi))
                    event_records.append(
                        {
                            "station_id": station,
                            "timestamp_utc": work.loc[bi, "timestamp_utc"],
                            "event_id": int(event_id),
                            "phase": phase,
                            "duration_h": int(dur_h),
                            "stuck_injected": 1,
                        }
                    )

                # Include +/-1h context around injected event for precision denominator.
                t_start_ctx = start_ts - pd.Timedelta(hours=1)
                t_end_ctx = end_ts + pd.Timedelta(hours=1)
                ctx = st[(st["timestamp_utc"] >= t_start_ctx) & (st["timestamp_utc"] <= t_end_ctx)]
                bench_rows_idx.update(ctx["base_index"].astype(int).tolist())

                used_local.update(found)
                events.append(
                    {
                        "event_id": int(event_id),
                        "station_id": station,
                        "phase": phase,
                        "duration_h": int(dur_h),
                        "start_ts": str(start_ts),
                        "end_ts": str(end_ts),
                        "n_rows": int(len(base_indices)),
                    }
                )

    _require(len(bench_rows_idx) > 0, "Benchmark rows are empty after injection.")
    bench_df = work.loc[sorted(bench_rows_idx)].copy().sort_values(["station_id", "timestamp_utc"]).reset_index(drop=True)
    bench_df["stuck_injected"] = 0

    injected_key = set((int(i)) for i in injected_idx)
    # Recover original base index by merge key.
    key = work[["station_id", "timestamp_utc"]].reset_index().rename(columns={"index": "base_index"})
    bench_df = bench_df.merge(key, on=["station_id", "timestamp_utc"], how="left", validate="one_to_one")
    bench_df["stuck_injected"] = bench_df["base_index"].astype(int).isin(injected_key).astype(int)

    labels = bench_df[["station_id", "timestamp_utc", "base_index", "stuck_injected"]].copy()
    if event_records:
        ev = pd.DataFrame(event_records)
        ev["timestamp_utc"] = pd.to_datetime(ev["timestamp_utc"], utc=True)
        labels = labels.merge(ev, on=["station_id", "timestamp_utc", "stuck_injected"], how="left")
    labels["event_id"] = labels["event_id"].fillna(-1).astype(int)
    labels["phase"] = labels["phase"].fillna("none")
    labels["duration_h"] = labels["duration_h"].fillna(0).astype(int)
    labels.to_parquet(paths.benchmark / "benchmark_labels.parquet", index=False)

    manifest = {
        "seed_synthetic": SEED_SYNTHETIC,
        "selected_dirty_stations": chosen,
        "durations_hours": list(BENCHMARK_DURATIONS_H),
        "injection_tol_hpa": BENCHMARK_INJECTION_TOL_HPA,
        "events_n": len(events),
        "events": events,
    }
    (paths.benchmark / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return bench_df, labels


def _evaluate_stuck_benchmark_configs(
    paths: Paths,
    bench_df: pd.DataFrame,
    med_delta_train_s: dict[str, float],
) -> pd.DataFrame:
    rows = []
    gt = bench_df["stuck_injected"].astype(bool).to_numpy()
    for cfg in _config_grid():
        T = int(cfg["T_hours"])
        tol = float(cfg["tol_hpa"])
        pred = _compute_stuck_flags(bench_df, T_hours=T, tol_hpa=tol, med_delta_train_s=med_delta_train_s).to_numpy(dtype=bool)
        tp = int(np.sum(pred & gt))
        fn = int(np.sum((~pred) & gt))
        fp = int(np.sum(pred & (~gt)))
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else np.nan
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rows.append(
            {
                "config_key": _config_key(cfg),
                "T_hours": T,
                "tol_hpa": tol,
                "policy_suspect": str(cfg["policy_suspect"]),
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "recall_stuck": recall,
                "precision_stuck": precision,
                "constraint_recall": bool((not np.isnan(recall)) and (recall >= BENCHMARK_RECALL_MIN)),
                "constraint_precision": bool((not np.isnan(precision)) and (precision >= BENCHMARK_PRECISION_MIN)),
            }
        )
    perf = pd.DataFrame(rows)
    perf["constraint_benchmark"] = perf["constraint_recall"] & perf["constraint_precision"]
    perf.to_csv(paths.benchmark / "benchmark_perf_by_config.csv", index=False)
    return perf


def _make_figure_f8(paths: Paths, bench_perf: pd.DataFrame) -> None:
    f8 = paths.figures / "F8_benchmark_detector_power.pdf"
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sub = bench_perf[bench_perf["policy_suspect"] == "DROP"].copy()
    if sub.empty:
        sub = bench_perf[bench_perf["policy_suspect"] == "KEEP"].copy()
    if sub.empty:
        ax[0].text(0.5, 0.5, "No benchmark perf data", ha="center")
        ax[1].text(0.5, 0.5, "No benchmark perf data", ha="center")
    else:
        piv_r = sub.pivot_table(index="T_hours", columns="tol_hpa", values="recall_stuck", aggfunc="mean")
        piv_p = sub.pivot_table(index="T_hours", columns="tol_hpa", values="precision_stuck", aggfunc="mean")
        im0 = ax[0].imshow(piv_r.to_numpy(dtype=float), aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        im1 = ax[1].imshow(piv_p.to_numpy(dtype=float), aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
        ax[0].set_xticks(np.arange(len(piv_r.columns)), [f"{c:.2f}" for c in piv_r.columns])
        ax[1].set_xticks(np.arange(len(piv_p.columns)), [f"{c:.2f}" for c in piv_p.columns])
        ax[0].set_yticks(np.arange(len(piv_r.index)), [str(int(v)) for v in piv_r.index])
        ax[1].set_yticks(np.arange(len(piv_p.index)), [str(int(v)) for v in piv_p.index])
        ax[0].set_xlabel("tol_hpa")
        ax[1].set_xlabel("tol_hpa")
        ax[0].set_ylabel("T_hours")
        ax[1].set_ylabel("T_hours")
        ax[0].set_title("Recall stuck")
        ax[1].set_title("Precision stuck")
        fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
        fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
    fig.suptitle("F8 Benchmark Detector Power")
    plt.tight_layout()
    plt.savefig(f8)
    plt.close()


def _phase_mask(df: pd.DataFrame, phase: str) -> pd.Series:
    if phase == "train":
        return (df["timestamp_utc"] >= pd.Timestamp(TRAIN_START, tz="UTC")) & (df["timestamp_utc"] < pd.Timestamp(TRAIN_END, tz="UTC"))
    if phase == "test":
        return (df["timestamp_utc"] >= pd.Timestamp(TEST_START, tz="UTC")) & (df["timestamp_utc"] < pd.Timestamp(TEST_END, tz="UTC"))
    raise ValueError(phase)


def _scope_mask(df: pd.DataFrame, scope: str) -> pd.Series:
    if scope == "global":
        return pd.Series(True, index=df.index)
    return df["stratum"] == scope


def _base_fail_row(df: pd.DataFrame) -> pd.Series:
    return (
        (df["pressure_fail_zero"] == 1)
        | (df["pressure_fail_out_range_after_norm"] == 1)
        | (df["fail_wind_incoherent"] == 1)
        | (df["fail_wind_negative"] == 1)
    )


def _build_product_keep_mask(df: pd.DataFrame, product: str, config: dict[str, Any] | None = None) -> pd.Series:
    fail_any = _base_fail_row(df)
    if product == "P0":
        return pd.Series(True, index=df.index)

    if product == "P1":
        stuck_col = f"suspect_stuck_T{T_REF}_tol{TOL_REF:.2f}"
        suspect = df[stuck_col]
        unit_censor = df["flag_unit_converted"] == 1
        return ~(fail_any | suspect | unit_censor)

    if product == "P2":
        return ~fail_any

    if product == "P3":
        _require(config is not None, "P3 needs selected config.")
        T = int(config["T_hours"])
        tol = float(config["tol_hpa"])
        pol = str(config["policy_suspect"])
        stuck_col = f"suspect_stuck_T{T}_tol{tol:.2f}"
        if pol == "DROP":
            return ~(fail_any | df[stuck_col])
        return ~fail_any

    if product == "P4":
        # Option B baseline: drop FAIL + SUSPECT, with convert+verify, no unit censorship.
        stuck_col = f"suspect_stuck_T{T_REF}_tol{TOL_REF:.2f}"
        return ~(fail_any | df[stuck_col])

    if product == "P5":
        # Fixed domain-rule QC baseline: physical/internal checks + conventional persistence.
        stuck_col = f"suspect_stuck_T{T_REF}_tol{TOL_REF:.2f}"
        return ~(fail_any | df[stuck_col])

    raise ValueError(product)


def _build_product_suspect_mask(df: pd.DataFrame, product: str, config: dict[str, Any] | None = None) -> pd.Series:
    if product in ["P0", "P1", "P2", "P4", "P5"]:
        return df[f"suspect_stuck_T{T_REF}_tol{TOL_REF:.2f}"].astype(bool)
    if product == "P3":
        _require(config is not None, "P3 needs selected config for suspect mask.")
        T = int(config["T_hours"])
        tol = float(config["tol_hpa"])
        return df[f"suspect_stuck_T{T}_tol{tol:.2f}"].astype(bool)
    raise ValueError(product)


def _apply_product(
    df: pd.DataFrame,
    product: str,
    keep_mask: pd.Series,
    suspect_mask: pd.Series,
) -> pd.DataFrame:
    out = df[[
        "station_id",
        "timestamp_utc",
        "stratum",
        "pressure_hpa",
        "wind_mean",
        "wind_gust",
        "wind_dir",
        "pressure_raw",
        "flag_unit_station",
        "flag_unit_converted",
        "pressure_fail_out_range_after_norm",
        "pressure_fail_zero",
        "fail_wind_incoherent",
        "fail_wind_negative",
        "fail_wind_dir_999",
        "fail_wind_dir_range",
    ]].copy()
    out["product"] = product
    out["suspect_pressure_stuck"] = suspect_mask.astype(bool).to_numpy()

    fail_wind = ((out["fail_wind_incoherent"] == 1) | (out["fail_wind_negative"] == 1)).to_numpy()
    fail_pressure = ((out["pressure_fail_zero"] == 1) | (out["pressure_fail_out_range_after_norm"] == 1)).to_numpy()
    suspect_arr = out["suspect_pressure_stuck"].to_numpy(dtype=bool)

    out["q_wind_gust"] = np.where(fail_wind, 0.0, np.where(suspect_arr, 0.5, 1.0))
    out["q_pressure"] = np.where(fail_pressure, 0.0, np.where(suspect_arr, 0.5, 1.0))

    out.loc[~keep_mask, "pressure_hpa"] = np.nan
    out.loc[~keep_mask, "wind_gust"] = np.nan

    bad_dir = (out["fail_wind_dir_999"] == 1) | (out["fail_wind_dir_range"] == 1)
    out.loc[bad_dir, "wind_dir"] = np.nan
    return out


def _station_topk_metrics(df_phase: pd.DataFrame, k_top: int) -> pd.DataFrame:
    rows = []
    for station, st in df_phase.groupby("station_id", sort=True):
        st = st.sort_values("timestamp_utc")
        p0_top = st.dropna(subset=["wind_gust_P0"]).nlargest(k_top, "wind_gust_P0")
        pr_top = st.dropna(subset=["wind_gust_PR"]).nlargest(k_top, "wind_gust_PR")

        p0_set = set(p0_top["timestamp_utc"].tolist())
        pr_set = set(pr_top["timestamp_utc"].tolist())
        overlap = len(p0_set & pr_set) / max(1, k_top)

        coh_cond = (
            (st["fail_wind_incoherent"] == 0)
            & (st["fail_wind_negative"] == 0)
            & (
                st["pressure_hpa"].isna()
                | ((st["pressure_fail_zero"] == 0) & (st["pressure_fail_out_range_after_norm"] == 0))
            )
        )
        p0_coh = float(coh_cond.loc[p0_top.index].sum() / max(1, k_top))
        pr_coh = float(coh_cond.loc[pr_top.index].sum() / max(1, k_top))

        rows.append(
            {
                "station_id": station,
                "stratum": str(st["stratum"].iloc[0]),
                "topk_overlap": overlap,
                "coh_p0": p0_coh,
                "coh_pr": pr_coh,
                "coh_gain": pr_coh - p0_coh,
            }
        )
    return pd.DataFrame(rows)


def _state_wind_spread(df_phase: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    base = df_phase[df_phase["wind_gust_P0"].notna()].copy()
    if base.empty:
        return np.nan, pd.DataFrame(columns=["decile", "removal_rate"])

    base["decile"] = pd.qcut(base["wind_gust_P0"], 10, labels=False, duplicates="drop")
    base["decile"] = base["decile"].astype(int) + 1
    base["removed"] = base["wind_gust_PR"].isna()

    dec = base.groupby("decile", as_index=False).agg(removal_rate=("removed", "mean"), n=("removed", "size"))
    spread = float(np.nanquantile(dec["removal_rate"].to_numpy(dtype=float), 0.90) - np.nanquantile(dec["removal_rate"].to_numpy(dtype=float), 0.10))
    return spread, dec


def _evaluate_product_scope_phase(
    base: pd.DataFrame,
    keep_mask: pd.Series,
    suspect_mask: pd.Series,
    phase: str,
    scope: str,
    k_top: int,
) -> dict[str, Any]:
    mask = _phase_mask(base, phase) & _scope_mask(base, scope)
    df = base.loc[mask].copy()
    if df.empty:
        return {
            "phase": phase,
            "scope": scope,
            "n_total": 0,
            "missing_rate_p0": np.nan,
            "missing_rate_product": np.nan,
            "missing_additional": np.nan,
            "p99_common_p0": np.nan,
            "p99_common_product": np.nan,
            "p999_common_p0": np.nan,
            "p999_common_product": np.nan,
            "delta_common_p99": np.nan,
            "delta_common_p999": np.nan,
            "p99_oper_p0": np.nan,
            "p99_oper_product": np.nan,
            "p999_oper_p0": np.nan,
            "p999_oper_product": np.nan,
            "delta_oper_p99": np.nan,
            "delta_oper_p999": np.nan,
            "state_wind_spread": np.nan,
            "suspect_rate": np.nan,
            "utility_v3": np.nan,
            "topk_overlap_median": np.nan,
            "topk_coh_gain_median": np.nan,
        }, pd.DataFrame(), pd.DataFrame()

    df["wind_gust_P0"] = df["wind_gust"]
    df["wind_gust_PR"] = np.where(keep_mask.loc[df.index], df["wind_gust"], np.nan)
    df["suspect_active"] = suspect_mask.loc[df.index].astype(bool)

    n_total = int(df.shape[0])
    miss_p0 = float(df["wind_gust_P0"].isna().mean())
    miss_pr = float(df["wind_gust_PR"].isna().mean())

    common = df["wind_gust_P0"].notna() & df["wind_gust_PR"].notna()
    oper = df["wind_gust_PR"].notna()

    def q(v: pd.Series, quant: float) -> float:
        x = v.dropna().to_numpy(dtype=float)
        if x.size == 0:
            return np.nan
        return float(np.nanquantile(x, quant))

    p99_c_p0 = q(df.loc[common, "wind_gust_P0"], 0.99)
    p99_c_pr = q(df.loc[common, "wind_gust_PR"], 0.99)
    p999_c_p0 = q(df.loc[common, "wind_gust_P0"], 0.999)
    p999_c_pr = q(df.loc[common, "wind_gust_PR"], 0.999)

    p99_o_p0 = q(df.loc[df["wind_gust_P0"].notna(), "wind_gust_P0"], 0.99)
    p99_o_pr = q(df.loc[oper, "wind_gust_PR"], 0.99)
    p999_o_p0 = q(df.loc[df["wind_gust_P0"].notna(), "wind_gust_P0"], 0.999)
    p999_o_pr = q(df.loc[oper, "wind_gust_PR"], 0.999)

    def delta(a: float, b: float) -> float:
        if np.isnan(a) or np.isnan(b) or a == 0:
            return np.nan
        return float((b - a) / a)

    topk_station = _station_topk_metrics(df, k_top=k_top)
    spread, dec = _state_wind_spread(df)

    utility_v3 = np.nan
    ov_med = np.nan
    coh_gain_med = np.nan
    missing_additional = miss_pr - miss_p0
    suspect_rate = float(df["suspect_active"].mean()) if n_total > 0 else np.nan
    if not np.isnan(spread) and not np.isnan(missing_additional) and not np.isnan(suspect_rate):
        utility_v3 = float(
            (-1.0 * spread)
            - (UTILITY_V3_LAMBDA * missing_additional)
            - (UTILITY_V3_ETA * suspect_rate)
        )
    if not topk_station.empty:
        ov_med = float(np.nanmedian(topk_station["topk_overlap"].to_numpy(dtype=float)))
        coh_gain_med = float(np.nanmedian(topk_station["coh_gain"].to_numpy(dtype=float)))

    summary = {
        "phase": phase,
        "scope": scope,
        "n_total": n_total,
        "missing_rate_p0": miss_p0,
        "missing_rate_product": miss_pr,
        "missing_additional": missing_additional,
        "p99_common_p0": p99_c_p0,
        "p99_common_product": p99_c_pr,
        "p999_common_p0": p999_c_p0,
        "p999_common_product": p999_c_pr,
        "delta_common_p99": delta(p99_c_p0, p99_c_pr),
        "delta_common_p999": delta(p999_c_p0, p999_c_pr),
        "p99_oper_p0": p99_o_p0,
        "p99_oper_product": p99_o_pr,
        "p999_oper_p0": p999_o_p0,
        "p999_oper_product": p999_o_pr,
        "delta_oper_p99": delta(p99_o_p0, p99_o_pr),
        "delta_oper_p999": delta(p999_o_p0, p999_o_pr),
        "state_wind_spread": spread,
        "suspect_rate": suspect_rate,
        "utility_v3": utility_v3,
        "topk_overlap_median": ov_med,
        "topk_coh_gain_median": coh_gain_med,
    }
    dec["phase"] = phase
    dec["scope"] = scope
    return summary, topk_station, dec


def _config_grid() -> list[dict[str, Any]]:
    out = []
    for T, tol, pol in itertools.product(T_GRID, TOL_GRID, POLICY_GRID):
        out.append({"T_hours": int(T), "tol_hpa": float(tol), "policy_suspect": str(pol)})
    return out


def _config_key(cfg: dict[str, Any]) -> str:
    return f"T{int(cfg['T_hours'])}_tol{float(cfg['tol_hpa']):.2f}_{cfg['policy_suspect']}"


def _stratified_station_folds(selected: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED_STATION_SPLIT)
    rows = []
    for s in ["clean", "medium", "dirty"]:
        st = sorted(selected[selected["stratum"] == s]["station_id"].tolist())
        arr = np.array(st, dtype=object)
        rng.shuffle(arr)
        for i, station in enumerate(arr.tolist()):
            rows.append({"station_id": station, "stratum": s, "fold": int(i % 3)})
    return pd.DataFrame(rows)


def _search_budget_for_station_subset(
    base: pd.DataFrame,
    station_ids: set[str],
    B: float,
    phase: str,
    benchmark_perf_map: dict[str, dict[str, float]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    train_sub = base[base["station_id"].isin(station_ids)].copy()
    _require(not train_sub.empty, f"No rows for budget search in phase={phase}.")

    rows = []
    best: dict[str, Any] | None = None

    for cfg in _config_grid():
        keep = _build_product_keep_mask(train_sub, "P3", config=cfg)
        suspect = _build_product_suspect_mask(train_sub, "P3", config=cfg)
        summ_train, topk_station_train, _ = _evaluate_product_scope_phase(
            train_sub,
            keep_mask=keep,
            suspect_mask=suspect,
            phase="train",
            scope="dirty",
            k_top=K_TOP,
        )
        summ_test, topk_station_test, _ = _evaluate_product_scope_phase(
            train_sub,
            keep_mask=keep,
            suspect_mask=suspect,
            phase="test",
            scope="dirty",
            k_top=K_TOP,
        )

        row = {
            "config_key": _config_key(cfg),
            "T_hours": int(cfg["T_hours"]),
            "tol_hpa": float(cfg["tol_hpa"]),
            "policy_suspect": str(cfg["policy_suspect"]),
            "B": float(B),
            "phase": "train",
            "scope": "dirty",
            # convenience aliases (train values)
            "missing_additional": summ_train["missing_additional"],
            "delta_common_p99": summ_train["delta_common_p99"],
            "delta_common_p999": summ_train["delta_common_p999"],
            "state_wind_spread": summ_train["state_wind_spread"],
            "suspect_rate": summ_train["suspect_rate"],
            "utility_v3": summ_train["utility_v3"],
            "topk_overlap_median": summ_train["topk_overlap_median"],
            "topk_coh_gain_median": summ_train["topk_coh_gain_median"],
            # full train metrics
            "missing_additional_train": summ_train["missing_additional"],
            "delta_common_p99_train": summ_train["delta_common_p99"],
            "delta_common_p999_train": summ_train["delta_common_p999"],
            "state_wind_spread_train": summ_train["state_wind_spread"],
            "suspect_rate_train": summ_train["suspect_rate"],
            "utility_v3_train": summ_train["utility_v3"],
            "topk_overlap_median_train": summ_train["topk_overlap_median"],
            "topk_coh_gain_median_train": summ_train["topk_coh_gain_median"],
            "n_total_train": summ_train["n_total"],
            # full test metrics (same stations, out-of-sample in time)
            "missing_additional_test": summ_test["missing_additional"],
            "delta_common_p99_test": summ_test["delta_common_p99"],
            "delta_common_p999_test": summ_test["delta_common_p999"],
            "state_wind_spread_test": summ_test["state_wind_spread"],
            "suspect_rate_test": summ_test["suspect_rate"],
            "utility_v3_test": summ_test["utility_v3"],
            "topk_overlap_median_test": summ_test["topk_overlap_median"],
            "topk_coh_gain_median_test": summ_test["topk_coh_gain_median"],
            "n_total_test": summ_test["n_total"],
            "n_topk_stations_train": int(topk_station_train.shape[0]) if not topk_station_train.empty else 0,
            "n_topk_stations_test": int(topk_station_test.shape[0]) if not topk_station_test.empty else 0,
        }
        bench = None if benchmark_perf_map is None else benchmark_perf_map.get(row["config_key"])
        row["recall_stuck"] = np.nan if bench is None else float(bench.get("recall_stuck", np.nan))
        row["precision_stuck"] = np.nan if bench is None else float(bench.get("precision_stuck", np.nan))
        row["constraint_missing"] = bool((not np.isnan(row["missing_additional"])) and (row["missing_additional"] <= B))
        row["constraint_common"] = bool(
            (not np.isnan(row["delta_common_p99"]))
            and (not np.isnan(row["delta_common_p999"]))
            and (abs(row["delta_common_p99"]) <= EPSILON)
            and (abs(row["delta_common_p999"]) <= EPSILON)
        )
        row["constraint_mnar"] = bool((not np.isnan(row["state_wind_spread"])) and (row["state_wind_spread"] <= TAU))
        if C6_ENABLED:
            row["constraint_benchmark_recall"] = bool((not np.isnan(row["recall_stuck"])) and (row["recall_stuck"] >= BENCHMARK_RECALL_MIN))
            row["constraint_benchmark_precision"] = bool((not np.isnan(row["precision_stuck"])) and (row["precision_stuck"] >= BENCHMARK_PRECISION_MIN))
            row["constraint_benchmark"] = bool(row["constraint_benchmark_recall"] and row["constraint_benchmark_precision"])
        else:
            row["constraint_benchmark_recall"] = True
            row["constraint_benchmark_precision"] = True
            row["constraint_benchmark"] = True
        row["compliant"] = bool(
            row["constraint_missing"]
            and row["constraint_common"]
            and row["constraint_mnar"]
            and row["constraint_benchmark"]
        )
        rows.append(row)

    grid = pd.DataFrame(rows)
    for col, rnk in [("utility_v3_train", "rank_train"), ("utility_v3_test", "rank_test")]:
        ordered = grid.sort_values(
            [col, "T_hours", "tol_hpa", "policy_suspect"],
            ascending=[False, True, True, True],
            na_position="last",
        ).index.tolist()
        rank_map = {idx: i + 1 for i, idx in enumerate(ordered)}
        grid[rnk] = grid.index.map(rank_map).astype(int)

    feasible = grid[grid["compliant"]].copy()
    if not feasible.empty:
        feasible = feasible.sort_values(
            ["utility_v3_train", "T_hours", "tol_hpa", "policy_suspect"],
            ascending=[False, True, True, True],
        )
        pick = feasible.iloc[0].to_dict()
        best = {}
        for k, v in pick.items():
            if isinstance(v, (np.integer,)):
                best[k] = int(v)
            elif isinstance(v, (np.floating,)):
                best[k] = float(v)
            else:
                best[k] = v
        best["B"] = float(B)

    return grid, best


def _write_product_data(paths: Paths, product: str, data: pd.DataFrame, manifest: dict[str, Any]) -> None:
    prod_dir = paths.results / "products" / product
    data_path = prod_dir / "data.parquet"
    man_path = prod_dir / "manifest.json"
    data.to_parquet(data_path, index=False)

    payload = {
        **manifest,
        "product": product,
        "config_hash": _json_hash(manifest),
    }
    man_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _aggregate_metrics(
    base: pd.DataFrame,
    product_data: dict[str, pd.DataFrame],
    paths: Paths,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tails_rows = []
    miss_rows = []
    mnar_rows = []
    topk_rows = []
    removed_rows = []
    decile_rows = []
    suspect_rows = []

    for product, dfp in product_data.items():
        # align with base index using merge on key
        merged = base[[
            "station_id",
            "timestamp_utc",
            "stratum",
            "pressure_hpa",
            "wind_gust",
            "wind_mean",
            "wind_dir",
            "pressure_fail_zero",
            "pressure_fail_out_range_after_norm",
            "fail_wind_incoherent",
            "fail_wind_negative",
        ]].copy()
        merged = merged.merge(
            dfp[[
                "station_id",
                "timestamp_utc",
                "pressure_hpa",
                "wind_gust",
                "wind_dir",
                "suspect_pressure_stuck",
                "q_wind_gust",
                "q_pressure",
            ]].rename(
                columns={
                    "pressure_hpa": "pressure_hpa_PR",
                    "wind_gust": "wind_gust_PR",
                    "wind_dir": "wind_dir_PR",
                }
            ),
            on=["station_id", "timestamp_utc"],
            how="left",
            validate="one_to_one",
        )
        merged["wind_gust_P0"] = merged["wind_gust"]
        merged["pressure_hpa_P0"] = merged["pressure_hpa"]

        for phase in ["train", "test"]:
            for scope in ["global", "clean", "medium", "dirty"]:
                mask = _phase_mask(merged, phase) & _scope_mask(merged, scope)
                sub = merged.loc[mask].copy()
                if sub.empty:
                    continue

                suspect_rate = float(sub["suspect_pressure_stuck"].fillna(False).astype(bool).mean())
                by_station_sus = (
                    sub.assign(_sus=sub["suspect_pressure_stuck"].fillna(False).astype(bool))
                    .groupby("station_id", as_index=False)["_sus"]
                    .sum()
                    .rename(columns={"_sus": "suspect_count"})
                )
                total_sus = float(by_station_sus["suspect_count"].sum())
                top1_share = float(by_station_sus["suspect_count"].max() / total_sus) if total_sus > 0 else 0.0
                suspect_rows.append(
                    {
                        "product": product,
                        "phase": phase,
                        "scope": scope,
                        "suspect_rate": suspect_rate,
                        "top1_suspect_share": top1_share,
                        "suspect_count_total": int(total_sus),
                        "n_rows": int(sub.shape[0]),
                    }
                )

                # wind tails
                for var, p0_col, pr_col in [
                    ("wind_gust", "wind_gust_P0", "wind_gust_PR"),
                    ("pressure_hpa", "pressure_hpa_P0", "pressure_hpa_PR"),
                ]:
                    common = sub[p0_col].notna() & sub[pr_col].notna()
                    oper = sub[pr_col].notna()

                    def q(s: pd.Series, qq: float) -> float:
                        x = s.dropna().to_numpy(dtype=float)
                        if x.size == 0:
                            return np.nan
                        return float(np.nanquantile(x, qq))

                    p95_c0 = q(sub.loc[common, p0_col], 0.95)
                    p95_cp = q(sub.loc[common, pr_col], 0.95)
                    p99_c0 = q(sub.loc[common, p0_col], 0.99)
                    p99_cp = q(sub.loc[common, pr_col], 0.99)
                    p999_c0 = q(sub.loc[common, p0_col], 0.999)
                    p999_cp = q(sub.loc[common, pr_col], 0.999)

                    p95_o0 = q(sub.loc[sub[p0_col].notna(), p0_col], 0.95)
                    p95_op = q(sub.loc[oper, pr_col], 0.95)
                    p99_o0 = q(sub.loc[sub[p0_col].notna(), p0_col], 0.99)
                    p99_op = q(sub.loc[oper, pr_col], 0.99)
                    p999_o0 = q(sub.loc[sub[p0_col].notna(), p0_col], 0.999)
                    p999_op = q(sub.loc[oper, pr_col], 0.999)

                    def d(a: float, b: float) -> float:
                        if np.isnan(a) or np.isnan(b) or a == 0:
                            return np.nan
                        return float((b - a) / a)

                    tails_rows.append(
                        {
                            "product": product,
                            "phase": phase,
                            "scope": scope,
                            "variable": var,
                            "p95_common_p0": p95_c0,
                            "p95_common_product": p95_cp,
                            "p99_common_p0": p99_c0,
                            "p99_common_product": p99_cp,
                            "p999_common_p0": p999_c0,
                            "p999_common_product": p999_cp,
                            "delta_common_p95": d(p95_c0, p95_cp),
                            "delta_common_p99": d(p99_c0, p99_cp),
                            "delta_common_p999": d(p999_c0, p999_cp),
                            "p95_oper_p0": p95_o0,
                            "p95_oper_product": p95_op,
                            "p99_oper_p0": p99_o0,
                            "p99_oper_product": p99_op,
                            "p999_oper_p0": p999_o0,
                            "p999_oper_product": p999_op,
                            "delta_oper_p95": d(p95_o0, p95_op),
                            "delta_oper_p99": d(p99_o0, p99_op),
                            "delta_oper_p999": d(p999_o0, p999_op),
                        }
                    )

                    miss_rows.append(
                        {
                            "product": product,
                            "phase": phase,
                            "scope": scope,
                            "variable": var,
                            "missing_rate_p0": float(sub[p0_col].isna().mean()),
                            "missing_rate_product": float(sub[pr_col].isna().mean()),
                            "missing_additional": float(sub[pr_col].isna().mean() - sub[p0_col].isna().mean()),
                        }
                    )

                # MNAR only for wind_gust
                sub_w = sub[sub["wind_gust_P0"].notna()].copy()
                if not sub_w.empty:
                    sub_w["decile"] = pd.qcut(sub_w["wind_gust_P0"], 10, labels=False, duplicates="drop")
                    sub_w["decile"] = sub_w["decile"].astype(int) + 1
                    sub_w["removed"] = sub_w["wind_gust_PR"].isna()
                    dec = sub_w.groupby("decile", as_index=False).agg(removal_rate=("removed", "mean"), n=("removed", "size"))
                    spread = float(np.nanquantile(dec["removal_rate"], 0.90) - np.nanquantile(dec["removal_rate"], 0.10))
                    mnar_rows.append(
                        {
                            "product": product,
                            "phase": phase,
                            "scope": scope,
                            "state_wind_spread": spread,
                            "n_base": int(sub_w.shape[0]),
                            "n_deciles": int(dec.shape[0]),
                        }
                    )
                    for _, r in dec.iterrows():
                        decile_rows.append(
                            {
                                "product": product,
                                "phase": phase,
                                "scope": scope,
                                "state_wind_spread": spread,
                                "decile": int(r["decile"]),
                                "removal_rate": float(r["removal_rate"]),
                                "n": int(r["n"]),
                            }
                        )

                    removed_vals = sub_w.loc[sub_w["removed"], "wind_gust_P0"]
                    kept_vals = sub_w.loc[~sub_w["removed"], "wind_gust_P0"]

                    def rq(s: pd.Series, qq: float) -> float:
                        x = s.dropna().to_numpy(dtype=float)
                        if x.size == 0:
                            return np.nan
                        return float(np.nanquantile(x, qq))

                    p99_removed = rq(removed_vals, 0.99)
                    p99_kept = rq(kept_vals, 0.99)
                    if np.isnan(p99_removed) or np.isnan(p99_kept) or p99_kept == 0:
                        p99_gap_rel = np.nan
                    else:
                        p99_gap_rel = float((p99_kept - p99_removed) / p99_kept)

                    removed_rows.append(
                        {
                            "product": product,
                            "phase": phase,
                            "scope": scope,
                            "n_base": int(sub_w.shape[0]),
                            "n_removed": int(removed_vals.shape[0]),
                            "n_kept": int(kept_vals.shape[0]),
                            "removed_rate": float(sub_w["removed"].mean()),
                            "p50_removed": rq(removed_vals, 0.50),
                            "p90_removed": rq(removed_vals, 0.90),
                            "p99_removed": p99_removed,
                            "p50_kept": rq(kept_vals, 0.50),
                            "p90_kept": rq(kept_vals, 0.90),
                            "p99_kept": p99_kept,
                            "p99_gap_rel": p99_gap_rel,
                            "state_wind_spread": spread,
                        }
                    )
                else:
                    mnar_rows.append(
                        {
                            "product": product,
                            "phase": phase,
                            "scope": scope,
                            "state_wind_spread": np.nan,
                            "n_base": 0,
                            "n_deciles": 0,
                        }
                    )
                    removed_rows.append(
                        {
                            "product": product,
                            "phase": phase,
                            "scope": scope,
                            "n_base": 0,
                            "n_removed": 0,
                            "n_kept": 0,
                            "removed_rate": np.nan,
                            "p50_removed": np.nan,
                            "p90_removed": np.nan,
                            "p99_removed": np.nan,
                            "p50_kept": np.nan,
                            "p90_kept": np.nan,
                            "p99_kept": np.nan,
                            "p99_gap_rel": np.nan,
                            "state_wind_spread": np.nan,
                        }
                    )

                # top-k metrics by station
                st_rows = []
                for station, st in sub.groupby("station_id", sort=True):
                    p0_top = st.dropna(subset=["wind_gust_P0"]).nlargest(K_TOP, "wind_gust_P0")
                    pr_top = st.dropna(subset=["wind_gust_PR"]).nlargest(K_TOP, "wind_gust_PR")
                    p0_set = set(p0_top["timestamp_utc"].tolist())
                    pr_set = set(pr_top["timestamp_utc"].tolist())
                    overlap = len(p0_set & pr_set) / max(1, K_TOP)

                    coh = (
                        (st["fail_wind_incoherent"] == 0)
                        & (st["fail_wind_negative"] == 0)
                        & (st["pressure_hpa"].isna() | ((st["pressure_fail_zero"] == 0) & (st["pressure_fail_out_range_after_norm"] == 0)))
                    )
                    coh_p0 = float(coh.loc[p0_top.index].sum() / max(1, K_TOP))
                    coh_pr = float(coh.loc[pr_top.index].sum() / max(1, K_TOP))
                    st_rows.append({"overlap": overlap, "coh_p0": coh_p0, "coh_pr": coh_pr, "gain": coh_pr - coh_p0})

                if st_rows:
                    st = pd.DataFrame(st_rows)
                    topk_rows.append(
                        {
                            "product": product,
                            "phase": phase,
                            "scope": scope,
                            "topk_overlap_median": float(st["overlap"].median()),
                            "topk_overlap_p10": float(st["overlap"].quantile(0.10)),
                            "topk_overlap_p90": float(st["overlap"].quantile(0.90)),
                            "topk_coherence_p0_median": float(st["coh_p0"].median()),
                            "topk_coherence_product_median": float(st["coh_pr"].median()),
                            "topk_coherence_gain_median": float(st["gain"].median()),
                            "topk_coherence_gain_p10": float(st["gain"].quantile(0.10)),
                            "topk_coherence_gain_p90": float(st["gain"].quantile(0.90)),
                        }
                    )

    tails = pd.DataFrame(tails_rows)
    miss = pd.DataFrame(miss_rows)
    mnar = pd.DataFrame(mnar_rows)
    topk = pd.DataFrame(topk_rows)
    removed_vs_kept = pd.DataFrame(removed_rows)
    removal_by_decile = pd.DataFrame(decile_rows)
    suspect_burden = pd.DataFrame(suspect_rows)

    tails.to_csv(paths.results / "metrics_tails.csv", index=False)
    miss.to_csv(paths.results / "metrics_missingness.csv", index=False)
    mnar.to_csv(paths.results / "metrics_mnar.csv", index=False)
    topk.to_csv(paths.results / "metrics_topk.csv", index=False)
    removed_vs_kept.to_csv(paths.results / "metrics_removed_vs_kept.csv", index=False)
    removal_by_decile.to_csv(paths.results / "metrics_removal_by_decile.csv", index=False)
    suspect_burden.to_csv(paths.results / "metrics_suspect_burden.csv", index=False)
    return tails, miss, mnar, topk, removed_vs_kept, removal_by_decile, suspect_burden


def _product_label(product: str) -> str:
    return {
        "P0": "P0 Raw",
        "P1": "P1 strict",
        "P2": "P2 fail-only",
        "P3": "P3 budgeted",
        "P4": "P4 policy baseline",
        "P5": "P5-DQC",
    }.get(product, product)


def _compute_exceedance_retention_tables(
    paths: Paths,
    product_data: dict[str, pd.DataFrame],
    miss: pd.DataFrame,
    mnar: pd.DataFrame,
    cost_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _require("P0" in product_data, "P0 missing for exceedance retention metrics.")

    p0 = product_data["P0"][["station_id", "timestamp_utc", "stratum", "wind_gust"]].rename(columns={"wind_gust": "wind_gust_P0"})
    vals_train = p0.loc[_phase_mask(p0, "train"), "wind_gust_P0"].dropna().to_numpy(dtype=float)
    _require(vals_train.size > 0, "No P0 TRAIN wind_gust values for exceedance thresholds.")
    thresholds = {
        "u95": float(np.nanquantile(vals_train, 0.95)),
        "u99": float(np.nanquantile(vals_train, 0.99)),
    }

    detail_rows = []
    for product in ["P0", "P1", "P2", "P3", "P4", "P5"]:
        if product not in product_data or product_data[product].empty:
            continue
        dfp = product_data[product][["station_id", "timestamp_utc", "wind_gust"]].rename(columns={"wind_gust": "wind_gust_product"})
        merged = p0.merge(dfp, on=["station_id", "timestamp_utc"], how="left", validate="one_to_one")
        for scope in ["global", "clean", "medium", "dirty"]:
            sub = merged.loc[_phase_mask(merged, "test") & _scope_mask(merged, scope)].copy()
            if sub.empty:
                continue
            for threshold_name, threshold_value in thresholds.items():
                raw_exceed = sub["wind_gust_P0"].notna() & (sub["wind_gust_P0"] > threshold_value)
                product_retained = raw_exceed & sub["wind_gust_product"].notna()
                product_exceed = sub["wind_gust_product"].notna() & (sub["wind_gust_product"] > threshold_value)
                n_raw_exceed = int(raw_exceed.sum())
                n_product_retained = int(product_retained.sum())
                n_product_exceed = int(product_exceed.sum())
                retention = np.nan if n_raw_exceed == 0 else float(n_product_retained / n_raw_exceed)
                delta_exceed = np.nan if n_raw_exceed == 0 else float((n_product_exceed - n_raw_exceed) / n_raw_exceed)
                detail_rows.append(
                    {
                        "product": product,
                        "product_label": _product_label(product),
                        "phase": "test",
                        "scope": scope,
                        "variable": "wind_gust",
                        "threshold_source": "P0_train_global",
                        "threshold": threshold_name,
                        "threshold_value": threshold_value,
                        "n_raw_exceed": n_raw_exceed,
                        "n_product_retained_raw_exceed": n_product_retained,
                        "retention_raw_exceedance": retention,
                        "n_product_exceed": n_product_exceed,
                        "delta_product_exceed_count_rel": delta_exceed,
                        "n_raw_nonnull": int(sub["wind_gust_P0"].notna().sum()),
                        "n_product_nonnull": int(sub["wind_gust_product"].notna().sum()),
                    }
                )

    detail = pd.DataFrame(detail_rows)
    detail.to_csv(paths.results / "metrics_exceedance_retention.csv", index=False)

    summary_rows = []
    for product in ["P0", "P1", "P2", "P3", "P4", "P5"]:
        if product not in product_data or product_data[product].empty:
            continue

        def scalar(df: pd.DataFrame, filters: dict[str, Any], field: str) -> float:
            sub = df.copy()
            for k, v in filters.items():
                sub = sub[sub[k] == v]
            if sub.empty:
                return np.nan
            return float(sub.iloc[0][field])

        missing_rate = scalar(miss, {"product": product, "phase": "test", "scope": "global", "variable": "wind_gust"}, "missing_rate_product")
        spread_dirty = scalar(mnar, {"product": product, "phase": "test", "scope": "dirty"}, "state_wind_spread")
        ridge_rmse = scalar(cost_df, {"product": product, "model": "ridge", "phase": "test", "scope": "global"}, "rmse_operational")
        ridge_cov = scalar(cost_df, {"product": product, "model": "ridge", "phase": "test", "scope": "global"}, "coverage_operational")
        ridge_score = scalar(cost_df, {"product": product, "model": "ridge", "phase": "test", "scope": "global"}, "score_cost_sensitive")

        def retention(threshold: str, scope: str = "dirty") -> float:
            if detail.empty:
                return np.nan
            sub = detail[
                (detail["product"] == product)
                & (detail["phase"] == "test")
                & (detail["scope"] == scope)
                & (detail["threshold"] == threshold)
            ]
            if sub.empty:
                return np.nan
            return float(sub.iloc[0]["retention_raw_exceedance"])

        summary_rows.append(
            {
                "product": product,
                "product_label": _product_label(product),
                "coverage_wind_gust_test_global": np.nan if np.isnan(missing_rate) else float(1.0 - missing_rate),
                "state_wind_spread_test_dirty": spread_dirty,
                "retention_u95_raw_test_dirty": retention("u95"),
                "retention_u99_raw_test_dirty": retention("u99"),
                "threshold_u95_P0_train_global": thresholds["u95"],
                "threshold_u99_P0_train_global": thresholds["u99"],
                "downstream_ridge_rmse_test_global": ridge_rmse,
                "downstream_ridge_coverage_test_global": ridge_cov,
                "downstream_ridge_score_test_global": ridge_score,
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(paths.results / "tab_policy_test_exceedance_summary.csv", index=False)
    tex_path = paths.results / "tab_policy_test_exceedance_summary.tex"
    if summary.empty:
        tex_path.write_text("% No policy exceedance summary rows generated.\n", encoding="utf-8")
    else:
        cols = [
            "product",
            "coverage_wind_gust_test_global",
            "state_wind_spread_test_dirty",
            "retention_u95_raw_test_dirty",
            "retention_u99_raw_test_dirty",
            "downstream_ridge_score_test_global",
        ]
        headers = ["Product", "Coverage", "MNAR", "Ret. u95", "Ret. u99", "Ridge score"]
        lines = [
            "\\begin{tabular}{lrrrrr}",
            "\\hline",
            " & ".join(headers) + " \\\\",
            "\\hline",
        ]
        for _, row in summary[cols].iterrows():
            vals = [str(row["product"])]
            vals.extend(f"{float(row[c]):.4f}" if not pd.isna(row[c]) else "NA" for c in cols[1:])
            lines.append(" & ".join(vals) + " \\\\")
        lines.extend(["\\hline", "\\end{tabular}", ""])
        tex_path.write_text("\n".join(lines), encoding="utf-8")
    return detail, summary


def _pareto_primary_removed(df_eval: pd.DataFrame, product: str, phase: str, scope: str) -> pd.DataFrame:
    mask = _phase_mask(df_eval, phase) & _scope_mask(df_eval, scope)
    sub = df_eval.loc[mask].copy()
    if sub.empty:
        return pd.DataFrame()

    removed = sub[sub["wind_gust_P0"].notna() & sub["wind_gust_PR"].isna()].copy()
    if removed.empty:
        return pd.DataFrame()

    rules = [
        "suspect_stuck",
        "flag_unit_converted",
        "pressure_fail_zero",
        "pressure_fail_out_range_after_norm",
        "fail_wind_incoherent",
        "fail_wind_negative",
    ]

    arr = removed[rules].to_numpy(dtype=bool)
    has_any = arr.any(axis=1)
    first_idx = np.argmax(arr, axis=1)
    primary = np.where(has_any, np.array(rules, dtype=object)[first_idx], "none")

    tbl = (
        pd.Series(primary, name="rule")
        .value_counts()
        .rename_axis("rule")
        .reset_index(name="rows_n")
        .sort_values("rows_n", ascending=False)
        .reset_index(drop=True)
    )
    total = max(1, int(tbl["rows_n"].sum()))
    tbl["share_pct"] = 100.0 * tbl["rows_n"] / total
    tbl["cum_share_pct"] = tbl["rows_n"].cumsum() / total * 100.0
    tbl["product"] = product
    tbl["phase"] = phase
    tbl["scope"] = scope
    return tbl[["product", "phase", "scope", "rule", "rows_n", "share_pct", "cum_share_pct"]]


def _save_pareto(paths: Paths, base: pd.DataFrame, product_data: dict[str, pd.DataFrame], p3_cfg: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for product in ["P1", "P2", "P3", "P4"]:
        dfp = product_data[product]
        merged = base[[
            "station_id",
            "timestamp_utc",
            "stratum",
            "wind_gust",
            "pressure_fail_zero",
            "pressure_fail_out_range_after_norm",
            "fail_wind_incoherent",
            "fail_wind_negative",
            "flag_unit_converted",
        ]].copy()
        merged = merged.merge(
            dfp[["station_id", "timestamp_utc", "wind_gust"]].rename(columns={"wind_gust": "wind_gust_PR"}),
            on=["station_id", "timestamp_utc"],
            how="left",
        )
        merged["wind_gust_P0"] = merged["wind_gust"]

        if product == "P1":
            stuck_col = f"suspect_stuck_T{T_REF}_tol{TOL_REF:.2f}"
        elif product == "P2":
            stuck_col = f"suspect_stuck_T{T_REF}_tol{TOL_REF:.2f}"
        elif product == "P4":
            stuck_col = f"suspect_stuck_T{T_REF}_tol{TOL_REF:.2f}"
        else:
            stuck_col = f"suspect_stuck_T{int(p3_cfg['T_hours'])}_tol{float(p3_cfg['tol_hpa']):.2f}"
        merged["suspect_stuck"] = base[stuck_col].values

        for phase in ["train", "test"]:
            tbl = _pareto_primary_removed(merged, product=product, phase=phase, scope="dirty")
            if not tbl.empty:
                rows.append(tbl)

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["product", "phase", "scope", "rule", "rows_n", "share_pct", "cum_share_pct"])
    out.to_csv(paths.results / "pareto_drop_rules.csv", index=False)
    return out


def _make_figures(
    paths: Paths,
    tails: pd.DataFrame,
    miss: pd.DataFrame,
    mnar: pd.DataFrame,
    topk: pd.DataFrame,
    pareto: pd.DataFrame,
    suspect_burden: pd.DataFrame,
    budget_grid: pd.DataFrame,
    best_configs: dict[float, dict[str, Any]],
    selected_budget: float,
    removed_vs_kept: pd.DataFrame,
    removal_by_decile: pd.DataFrame,
) -> None:
    # F1 Pareto
    f1 = paths.figures / "F1_pareto.pdf"
    plt.figure(figsize=(12, 6))
    sub = pareto[(pareto["phase"] == "test") & (pareto["scope"] == "dirty")].copy()
    if sub.empty:
        plt.text(0.5, 0.5, "No pareto data", ha="center")
    else:
        for i, product in enumerate(["P1", "P2", "P3", "P4"]):
            p = sub[sub["product"] == product].head(10)
            if p.empty:
                continue
            x = np.arange(p.shape[0]) + i * 0.2
            plt.bar(x, p["share_pct"].to_numpy(dtype=float), width=0.2, label=product)
        plt.xticks(np.arange(10) + 0.3, [f"R{i+1}" for i in range(10)])
        plt.ylabel("Primary-rule share (%)")
        plt.legend()
    plt.title("F1 Pareto del recorte por regla (DIRTY, TEST)")
    plt.tight_layout()
    plt.savefig(f1)
    plt.close()

    # F2 common vs operational tails
    f2 = paths.figures / "F2_common_vs_operational_tails.pdf"
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for i, phase in enumerate(["train", "test"]):
        sub = tails[
            (tails["variable"] == "wind_gust")
            & (tails["scope"].isin(["clean", "medium", "dirty"]))
            & (tails["phase"] == phase)
        ]
        if sub.empty:
            ax[i].text(0.5, 0.5, f"No tails data ({phase})", ha="center")
            ax[i].set_title(phase.upper())
            continue
        plot_rows = []
        for _, r in sub.iterrows():
            plot_rows.append({"x": f"{r['product']}-{r['scope']}", "metric": "d_common_p99", "v": 100.0 * r["delta_common_p99"]})
            plot_rows.append({"x": f"{r['product']}-{r['scope']}", "metric": "d_common_p999", "v": 100.0 * r["delta_common_p999"]})
            plot_rows.append({"x": f"{r['product']}-{r['scope']}", "metric": "d_oper_p99", "v": 100.0 * r["delta_oper_p99"]})
            plot_rows.append({"x": f"{r['product']}-{r['scope']}", "metric": "d_oper_p999", "v": 100.0 * r["delta_oper_p999"]})
        pp = pd.DataFrame(plot_rows)
        xlab = sorted(pp["x"].unique())
        xpos = np.arange(len(xlab))
        m_order = ["d_common_p99", "d_common_p999", "d_oper_p99", "d_oper_p999"]
        for j, m in enumerate(m_order):
            vals = [pp[(pp["x"] == x) & (pp["metric"] == m)]["v"].mean() for x in xlab]
            ax[i].bar(xpos + j * 0.2, vals, width=0.2, label=m)
        ax[i].set_xticks(xpos + 0.3, xlab, rotation=45, ha="right")
        ax[i].set_title(phase.upper())
        ax[i].set_ylabel("Delta (%)")
    handles, labels = ax[1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4)
    fig.suptitle("F2 Common vs Operational Deltas (wind)")
    plt.tight_layout()
    plt.savefig(f2)
    plt.close()

    # F3 MNAR spread + deciles + removed-vs-kept p99 (TRAIN+TEST)
    f3 = paths.figures / "F3_mnar_spread_deciles.pdf"
    fig, ax = plt.subplots(2, 3, figsize=(18, 9))
    for row_i, phase in enumerate(["train", "test"]):
        sub_sp = mnar[(mnar["phase"] == phase) & (mnar["scope"] == "dirty")].copy()
        if sub_sp.empty:
            ax[row_i, 0].text(0.5, 0.5, f"No MNAR data ({phase})", ha="center")
        else:
            ax[row_i, 0].bar(sub_sp["product"], sub_sp["state_wind_spread"])
        ax[row_i, 0].set_title(f"State Wind Spread (DIRTY, {phase.upper()})")

        sub_dec = removal_by_decile[(removal_by_decile["phase"] == phase) & (removal_by_decile["scope"] == "dirty")]
        if sub_dec.empty:
            ax[row_i, 1].text(0.5, 0.5, "No decile data", ha="center")
        else:
            for product in sorted(sub_dec["product"].unique().tolist()):
                pr = sub_dec[sub_dec["product"] == product].sort_values("decile")
                ax[row_i, 1].plot(pr["decile"], 100.0 * pr["removal_rate"], marker="o", label=product)
            ax[row_i, 1].legend(fontsize=7)
        ax[row_i, 1].set_title(f"Removal Rate by Decile ({phase.upper()})")
        ax[row_i, 1].set_xlabel("Decile")
        ax[row_i, 1].set_ylabel("Removal rate (%)")

        sub_rk = removed_vs_kept[(removed_vs_kept["phase"] == phase) & (removed_vs_kept["scope"] == "dirty")].copy()
        if sub_rk.empty:
            ax[row_i, 2].text(0.5, 0.5, "No removed-vs-kept data", ha="center")
        else:
            xpos = np.arange(sub_rk.shape[0])
            ax[row_i, 2].bar(xpos - 0.18, sub_rk["p99_removed"], width=0.35, label="p99_removed")
            ax[row_i, 2].bar(xpos + 0.18, sub_rk["p99_kept"], width=0.35, label="p99_kept")
            ax[row_i, 2].set_xticks(xpos, sub_rk["product"].tolist(), rotation=45, ha="right")
            ax[row_i, 2].legend(fontsize=7)
        ax[row_i, 2].set_title(f"Removed vs Kept p99 ({phase.upper()})")
        ax[row_i, 2].set_ylabel("wind_gust")

    plt.tight_layout()
    plt.savefig(f3)
    plt.close()

    # F4 Tradeoff plot (v3)
    f4 = paths.figures / "F4_tradeoff.pdf"
    plt.figure(figsize=(8, 5))
    sub = budget_grid[(budget_grid["fold"] == "full") & (budget_grid["phase"] == "train") & (budget_grid["scope"] == "dirty")]
    if sub.empty:
        plt.text(0.5, 0.5, "No budget grid data", ha="center")
    else:
        sc = plt.scatter(
            100.0 * sub["missing_additional_train"],
            sub["state_wind_spread_train"],
            c=sub["suspect_rate_train"],
            s=np.clip(30.0 + 500.0 * sub["suspect_rate_train"].to_numpy(dtype=float), 30.0, 120.0),
            alpha=0.7,
            cmap="viridis",
        )
        plt.colorbar(sc, label="suspect_rate_train")
        colors = ["red", "orange", "purple", "brown", "gray"]
        for i, B in enumerate(sorted(best_configs)):
            plt.axvline(100.0 * float(B), color=colors[i % len(colors)], linestyle="--", label=f"B={100.0 * float(B):.1f}%")
        sel_cfg = best_configs.get(float(selected_budget))
        if sel_cfg:
            plt.scatter(
                [100.0 * sel_cfg["missing_additional_train"]],
                [sel_cfg["state_wind_spread_train"]],
                color="green",
                s=90,
                marker="*",
                label=f"selected B={100.0 * float(selected_budget):.1f}%",
            )
        plt.legend()
    plt.xlabel("Missing additional (%)")
    plt.ylabel("State wind spread")
    plt.title("F4 Trade-off v3 (DIRTY, TRAIN)")
    plt.tight_layout()
    plt.savefig(f4)
    plt.close()


def _build_downstream_dataset(df: pd.DataFrame) -> pd.DataFrame:
    d = df[["station_id", "timestamp_utc", "wind_gust", "wind_mean", "pressure_hpa"]].copy()
    d = d.sort_values(["station_id", "timestamp_utc"]).reset_index(drop=True)

    # Target: next observation within 90 minutes.
    d["y_next"] = d.groupby("station_id")["wind_gust"].shift(-1)
    d["t_next"] = d.groupby("station_id")["timestamp_utc"].shift(-1)
    d["dt_next_min"] = (d["t_next"] - d["timestamp_utc"]).dt.total_seconds().div(60.0)
    d["y"] = np.where((d["dt_next_min"] > 0) & (d["dt_next_min"] <= 90), d["y_next"], np.nan)

    for v in ["wind_gust", "wind_mean", "pressure_hpa"]:
        for lag in range(1, 7):
            d[f"{v}_lag{lag}"] = d.groupby("station_id")[v].shift(lag)

    hour = d["timestamp_utc"].dt.hour.astype(float)
    month = d["timestamp_utc"].dt.month.astype(float)
    d["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    d["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    d["month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
    d["month_cos"] = np.cos(2.0 * np.pi * month / 12.0)

    feat_cols = []
    for v in ["wind_gust", "wind_mean", "pressure_hpa"]:
        for lag in range(1, 7):
            feat_cols.append(f"{v}_lag{lag}")
    feat_cols += ["hour_sin", "hour_cos", "month_sin", "month_cos"]

    d["has_all_features"] = d[feat_cols].notna().all(axis=1)
    return d


def _blocked_time_cv_alpha(train: pd.DataFrame, feat_cols: list[str], alpha_grid: list[float]) -> float:
    # Temporal blocked CV on sorted unique timestamps (3 validation blocks).
    times = np.array(sorted(train["timestamp_utc"].unique()))
    _require(times.size >= 90, "Not enough train timestamps for blocked CV.")

    blocks = np.array_split(times, 4)
    folds = [blocks[1], blocks[2], blocks[3]]

    best_alpha = None
    best_score = np.inf

    for a in alpha_grid:
        errs = []
        for val_times in folds:
            is_val = train["timestamp_utc"].isin(val_times)
            tr = train[~is_val]
            va = train[is_val]
            if tr.empty or va.empty:
                continue
            pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=a, random_state=SEED_MODEL))])
            pipe.fit(tr[feat_cols], tr["y"])  # type: ignore[arg-type]
            pred = pipe.predict(va[feat_cols])
            errs.append(float(np.sqrt(mean_squared_error(va["y"], pred))))
        if not errs:
            continue
        score = float(np.mean(errs))
        if score < best_score:
            best_score = score
            best_alpha = a

    _require(best_alpha is not None, "Blocked CV failed to select Ridge alpha.")
    return float(best_alpha)


def _feature_cols() -> list[str]:
    cols = []
    for v in ["wind_gust", "wind_mean", "pressure_hpa"]:
        for lag in range(1, 7):
            cols.append(f"{v}_lag{lag}")
    cols += ["hour_sin", "hour_cos", "month_sin", "month_cos"]
    return cols


def _estimate_gamma_from_p0_train(p0_df: pd.DataFrame) -> tuple[float, float]:
    d = _build_downstream_dataset(p0_df)
    feat_cols = _feature_cols()
    train_mask = (d["timestamp_utc"] >= pd.Timestamp(TRAIN_START, tz="UTC")) & (d["timestamp_utc"] < pd.Timestamp(TRAIN_END, tz="UTC"))
    tr = d[train_mask & d["y"].notna() & d["has_all_features"]].copy()
    _require(not tr.empty, "No TRAIN samples in P0 to estimate gamma.")
    alpha = _blocked_time_cv_alpha(tr, feat_cols=feat_cols, alpha_grid=[0.01, 0.1, 1.0, 10.0, 100.0])
    ridge = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha, random_state=SEED_MODEL))])
    ridge.fit(tr[feat_cols], tr["y"])  # type: ignore[arg-type]
    pred = ridge.predict(tr[feat_cols])
    rmse_train = float(np.sqrt(mean_squared_error(tr["y"], pred)))
    _require(rmse_train > 0, "Gamma estimate is non-positive.")
    return rmse_train, float(alpha)


def _downstream_eval_v3(paths: Paths, product_data: dict[str, pd.DataFrame], gamma_value: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    alpha_grid = [0.01, 0.1, 1.0, 10.0, 100.0]
    feat_cols = _feature_cols()

    eval_products = []
    for p in ["P0", "P1", "P2", "P3", "P4", "P5"]:
        if p in product_data and not product_data[p].empty:
            eval_products.append(p)
    datasets = {k: _build_downstream_dataset(product_data[k]) for k in eval_products}
    _require("P0" in datasets, "P0 dataset missing for downstream coverage reference.")

    d0 = datasets["P0"].copy()
    test_mask0 = (d0["timestamp_utc"] >= pd.Timestamp(TEST_START, tz="UTC")) & (d0["timestamp_utc"] < pd.Timestamp(TEST_END, tz="UTC"))
    potential_ridge = d0[test_mask0 & d0["y"].notna() & d0["has_all_features"]].copy()
    potential_ridge_n = int(potential_ridge.shape[0])
    _require(potential_ridge_n > 0, "No potential downstream samples in P0 TEST for ridge.")
    potential_persistence = potential_ridge[potential_ridge["wind_gust"].notna()].copy()
    potential_persistence_n = int(potential_persistence.shape[0])
    _require(potential_persistence_n > 0, "No potential downstream samples in P0 TEST for persistence.")

    rows = []
    for product, d in datasets.items():
        train_mask = (d["timestamp_utc"] >= pd.Timestamp(TRAIN_START, tz="UTC")) & (d["timestamp_utc"] < pd.Timestamp(TRAIN_END, tz="UTC"))
        test_mask = (d["timestamp_utc"] >= pd.Timestamp(TEST_START, tz="UTC")) & (d["timestamp_utc"] < pd.Timestamp(TEST_END, tz="UTC"))
        tr = d[train_mask & d["y"].notna() & d["has_all_features"]].copy()
        te = d[test_mask & d["y"].notna() & d["has_all_features"]].copy()
        if tr.empty or te.empty:
            continue

        p95_map = tr.groupby("station_id")["y"].quantile(0.95).to_dict()

        # persistence
        te_p = te[te["wind_gust"].notna()].copy()
        if not te_p.empty:
            te_p["y_hat"] = te_p["wind_gust"]
            rmse_p = float(np.sqrt(mean_squared_error(te_p["y"], te_p["y_hat"])))
            coverage_p = float(te_p.shape[0] / potential_persistence_n)
            tail_mask_p = te_p.apply(lambda r: bool(r["y"] >= p95_map.get(r["station_id"], np.inf)), axis=1)
            tail_rmse_p = float(np.sqrt(mean_squared_error(te_p.loc[tail_mask_p, "y"], te_p.loc[tail_mask_p, "y_hat"]))) if tail_mask_p.any() else np.nan
            rows.append(
                {
                    "product": product,
                    "model": "persistence",
                    "alpha": np.nan,
                    "rmse_operational": rmse_p,
                    "coverage_operational": coverage_p,
                    "score_cost_sensitive": float(rmse_p + gamma_value * (1.0 - coverage_p)),
                    "tail_rmse_operational": tail_rmse_p,
                    "n_operational": int(te_p.shape[0]),
                    "n_potential": potential_persistence_n,
                    "phase": "test",
                    "scope": "global",
                }
            )

        # ridge
        alpha = _blocked_time_cv_alpha(tr, feat_cols=feat_cols, alpha_grid=alpha_grid)
        ridge = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha, random_state=SEED_MODEL))])
        ridge.fit(tr[feat_cols], tr["y"])  # type: ignore[arg-type]
        te["y_hat"] = ridge.predict(te[feat_cols])
        rmse_r = float(np.sqrt(mean_squared_error(te["y"], te["y_hat"])))
        coverage_r = float(te.shape[0] / potential_ridge_n)
        tail_mask_r = te.apply(lambda r: bool(r["y"] >= p95_map.get(r["station_id"], np.inf)), axis=1)
        tail_rmse_r = float(np.sqrt(mean_squared_error(te.loc[tail_mask_r, "y"], te.loc[tail_mask_r, "y_hat"]))) if tail_mask_r.any() else np.nan
        rows.append(
            {
                "product": product,
                "model": "ridge",
                "alpha": alpha,
                "rmse_operational": rmse_r,
                "coverage_operational": coverage_r,
                "score_cost_sensitive": float(rmse_r + gamma_value * (1.0 - coverage_r)),
                "tail_rmse_operational": tail_rmse_r,
                "n_operational": int(te.shape[0]),
                "n_potential": potential_ridge_n,
                "phase": "test",
                "scope": "global",
            }
        )

    out = pd.DataFrame(rows)
    _require(not out.empty, "No downstream v3 rows generated.")

    skill = out[["product", "model", "rmse_operational", "coverage_operational", "n_operational", "n_potential", "phase", "scope"]].copy()
    cost = out[["product", "model", "rmse_operational", "coverage_operational", "score_cost_sensitive", "phase", "scope"]].copy()
    tail = out[["product", "model", "tail_rmse_operational", "phase", "scope"]].copy()

    skill.to_csv(paths.downstream / "downstream_skill_vs_coverage_v3.csv", index=False)
    cost.to_csv(paths.downstream / "downstream_cost_sensitive_v3.csv", index=False)
    tail.to_csv(paths.downstream / "downstream_tail_metrics_v3.csv", index=False)
    return skill, cost, tail


def _scalar_from(df: pd.DataFrame, filters: dict[str, Any], field: str) -> float:
    sub = df.copy()
    for k, v in filters.items():
        sub = sub[sub[k] == v]
    if sub.empty:
        return np.nan
    return float(sub.iloc[0][field])


def _safe_ratio(num: float, den: float) -> float:
    if np.isnan(num) or np.isnan(den):
        return np.nan
    if den == 0:
        if num > 0:
            return np.inf
        return 1.0
    return float(num / den)


def _write_decision_summary_v4(
    paths: Paths,
    checklist: dict[str, Any],
    pareto_top3_share: dict[str, float],
    benchmark_selected: dict[str, Any],
) -> None:
    c2 = checklist.get("C2v2", {})
    lines = [
        "# Decision Summary v4",
        "",
        f"- C1: `{checklist.get('C1')}`",
        f"- C2v2: `{c2.get('pass')}`",
        f"- C3: `{checklist.get('C3')}`",
        f"- C4: `{checklist.get('C4')}`",
        f"- C5: `{checklist.get('C5')}`",
        f"- C6: `{checklist.get('C6')}`",
        "",
        "## Pareto Top-3 (DIRTY, TEST)",
        f"- P1 top3_share_pct: `{pareto_top3_share.get('P1', np.nan):.2f}`",
        f"- P2 top3_share_pct: `{pareto_top3_share.get('P2', np.nan):.2f}`",
        f"- P3 top3_share_pct: `{pareto_top3_share.get('P3', np.nan):.2f}`",
        f"- P4 top3_share_pct: `{pareto_top3_share.get('P4', np.nan):.2f}`",
        "",
        "## Benchmark Selected Config",
        f"- config_key: `{benchmark_selected.get('config_key')}`",
        f"- recall_stuck: `{benchmark_selected.get('recall_stuck')}`",
        f"- precision_stuck: `{benchmark_selected.get('precision_stuck')}`",
        "",
        "## C2v2 subchecks",
        f"- C2v2.1 spread ratio train/test >= {C2V2_SPREAD_RATIO_R}: `{c2.get('C2v2.1_pass')}`",
        f"- C2v2.2 missing ratio train/test >= {C2V2_MISSING_RATIO_M}: `{c2.get('C2v2.2_pass')}`",
        f"- C2v2.3 removed-vs-kept evidence train/test: `{c2.get('C2v2.3_pass')}`",
        "",
        "## Final",
        f"- submit_status: `{checklist.get('submit_status')}`",
        f"- envsoft_ready: `{checklist.get('envsoft_ready')}`",
    ]
    (paths.results / "decision_summary_v4.md").write_text("\n".join(lines), encoding="utf-8")


def _station_level_heterogeneity(base: pd.DataFrame, product_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for product in ["P1", "P2", "P3", "P4"]:
        if product not in product_data:
            continue
        dfp = product_data[product]
        merged = base[["station_id", "timestamp_utc", "stratum", "wind_gust"]].rename(columns={"wind_gust": "wind_gust_P0"}).merge(
            dfp[["station_id", "timestamp_utc", "wind_gust"]].rename(columns={"wind_gust": "wind_gust_PR"}),
            on=["station_id", "timestamp_utc"],
            how="left",
        )
        mask = _phase_mask(merged, "test") & (merged["stratum"] == "dirty")
        sub = merged.loc[mask].copy()
        for station, st in sub.groupby("station_id", sort=True):
            miss_p0 = float(st["wind_gust_P0"].isna().mean())
            miss_pr = float(st["wind_gust_PR"].isna().mean())
            base_nonnull = st[st["wind_gust_P0"].notna()].copy()
            if base_nonnull.empty:
                spread = np.nan
            else:
                base_nonnull["decile"] = pd.qcut(base_nonnull["wind_gust_P0"], 10, labels=False, duplicates="drop")
                base_nonnull["decile"] = base_nonnull["decile"].astype(int) + 1
                base_nonnull["removed"] = base_nonnull["wind_gust_PR"].isna()
                dec = base_nonnull.groupby("decile", as_index=False)["removed"].mean().rename(columns={"removed": "rr"})
                spread = float(np.nanquantile(dec["rr"], 0.90) - np.nanquantile(dec["rr"], 0.10))
            rows.append(
                {
                    "product": product,
                    "station_id": station,
                    "missing_additional": miss_pr - miss_p0,
                    "state_wind_spread": spread,
                }
            )
    return pd.DataFrame(rows)


def _make_figure_f5(paths: Paths, hetero: pd.DataFrame) -> None:
    f5 = paths.figures / "F5_station_heterogeneity.pdf"
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    if hetero.empty:
        ax[0].text(0.5, 0.5, "No station heterogeneity data", ha="center")
        ax[1].text(0.5, 0.5, "No station heterogeneity data", ha="center")
    else:
        products = sorted(hetero["product"].unique().tolist())
        spread_data = [hetero[hetero["product"] == p]["state_wind_spread"].dropna().to_numpy(dtype=float) for p in products]
        miss_data = [hetero[hetero["product"] == p]["missing_additional"].dropna().to_numpy(dtype=float) for p in products]
        ax[0].boxplot(spread_data, labels=products)
        ax[0].set_title("State wind spread by station (DIRTY, TEST)")
        ax[0].set_ylabel("Spread")
        ax[1].boxplot([100.0 * x for x in miss_data], labels=products)
        ax[1].set_title("Missing additional by station (DIRTY, TEST)")
        ax[1].set_ylabel("Missing additional (%)")
    plt.tight_layout()
    plt.savefig(f5)
    plt.close()


def _make_figure_f6(paths: Paths, cost_df: pd.DataFrame, tail_df: pd.DataFrame) -> None:
    f6 = paths.figures / "F6_downstream_cost_tail.pdf"
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    if cost_df.empty:
        ax[0].text(0.5, 0.5, "No downstream cost data", ha="center")
    else:
        for model, g in cost_df.groupby("model", sort=True):
            ax[0].scatter(g["coverage_operational"], g["score_cost_sensitive"], label=model, s=40)
            for _, r in g.iterrows():
                ax[0].annotate(f"{r['product']}", (r["coverage_operational"], r["score_cost_sensitive"]), fontsize=7)
        ax[0].set_xlabel("Coverage operational")
        ax[0].set_ylabel("Cost-sensitive score")
        ax[0].set_title("Score vs coverage (TEST)")
        ax[0].legend()

    if tail_df.empty:
        ax[1].text(0.5, 0.5, "No tail metrics data", ha="center")
    else:
        piv = tail_df.pivot(index="product", columns="model", values="tail_rmse_operational")
        piv.plot(kind="bar", ax=ax[1])
        ax[1].set_title("Tail RMSE operational (TEST)")
        ax[1].set_ylabel("Tail RMSE")
    plt.tight_layout()
    plt.savefig(f6)
    plt.close()


def _run_synthetic_stress(paths: Paths, base: pd.DataFrame, product_data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    _require("P2" in product_data, "P2 missing for synthetic stress test.")
    cfg = {"T_hours": 3, "tol_hpa": 0.05, "policy_suspect": "DROP"}
    keep_s1 = _build_product_keep_mask(base, "P3", config=cfg)
    suspect_s1 = _build_product_suspect_mask(base, "P3", config=cfg)
    s1 = _apply_product(base, product="S1", keep_mask=keep_s1, suspect_mask=suspect_s1)
    s1.to_parquet(paths.synthetic / "S1_data.parquet", index=False)

    p2 = product_data["P2"][["station_id", "timestamp_utc", "stratum", "wind_gust"]].rename(columns={"wind_gust": "wind_gust_P2"})
    s1w = s1[["station_id", "timestamp_utc", "wind_gust"]].rename(columns={"wind_gust": "wind_gust_S1"})
    m = p2.merge(s1w, on=["station_id", "timestamp_utc"], how="left")
    rows = []
    for phase in ["train", "test"]:
        for scope in ["global", "clean", "medium", "dirty"]:
            mask = _phase_mask(m, phase) & _scope_mask(m, scope)
            sub = m.loc[mask].copy()
            if sub.empty:
                continue
            miss_p2 = float(sub["wind_gust_P2"].isna().mean())
            miss_s1 = float(sub["wind_gust_S1"].isna().mean())
            b = sub[sub["wind_gust_P2"].notna()].copy()
            spread = np.nan
            gap_rel = np.nan
            if not b.empty:
                b["decile"] = pd.qcut(b["wind_gust_P2"], 10, labels=False, duplicates="drop")
                b["decile"] = b["decile"].astype(int) + 1
                b["removed"] = b["wind_gust_S1"].isna()
                dec = b.groupby("decile", as_index=False)["removed"].mean().rename(columns={"removed": "rr"})
                spread = float(np.nanquantile(dec["rr"], 0.90) - np.nanquantile(dec["rr"], 0.10))
                removed = b[b["removed"]]["wind_gust_P2"]
                kept = b[~b["removed"]]["wind_gust_P2"]
                if (not removed.empty) and (not kept.empty):
                    p99_r = float(np.nanquantile(removed, 0.99))
                    p99_k = float(np.nanquantile(kept, 0.99))
                    if p99_k != 0:
                        gap_rel = float((p99_k - p99_r) / p99_k)
            p99_p2 = float(np.nanquantile(sub["wind_gust_P2"].dropna(), 0.99)) if sub["wind_gust_P2"].notna().any() else np.nan
            p99_s1 = float(np.nanquantile(sub["wind_gust_S1"].dropna(), 0.99)) if sub["wind_gust_S1"].notna().any() else np.nan
            p999_p2 = float(np.nanquantile(sub["wind_gust_P2"].dropna(), 0.999)) if sub["wind_gust_P2"].notna().any() else np.nan
            p999_s1 = float(np.nanquantile(sub["wind_gust_S1"].dropna(), 0.999)) if sub["wind_gust_S1"].notna().any() else np.nan

            d99 = np.nan if (np.isnan(p99_p2) or np.isnan(p99_s1) or p99_p2 == 0) else float((p99_s1 - p99_p2) / p99_p2)
            d999 = np.nan if (np.isnan(p999_p2) or np.isnan(p999_s1) or p999_p2 == 0) else float((p999_s1 - p999_p2) / p999_p2)
            rows.append(
                {
                    "phase": phase,
                    "scope": scope,
                    "missing_additional_vs_P2": miss_s1 - miss_p2,
                    "state_wind_spread_vs_P2": spread,
                    "delta_oper_p99_vs_P2": d99,
                    "delta_oper_p999_vs_P2": d999,
                    "removed_vs_kept_gap_rel": gap_rel,
                }
            )
    syn = pd.DataFrame(rows)
    syn.to_csv(paths.synthetic / "synthetic_metrics.csv", index=False)
    # Tail-shape metric: exceedance distortion at fixed thresholds from P0 TRAIN.
    p0 = product_data["P0"][["station_id", "timestamp_utc", "stratum", "wind_gust"]].copy()
    p0_train = p0[_phase_mask(p0, "train")]
    vals_train = p0_train["wind_gust"].dropna().to_numpy(dtype=float)
    _require(vals_train.size > 0, "No P0 TRAIN wind_gust for synthetic tail-shape thresholds.")
    u95 = float(np.nanquantile(vals_train, 0.95))
    u99 = float(np.nanquantile(vals_train, 0.99))

    tail_rows = []
    for scope in ["global", "dirty"]:
        mask = _phase_mask(m, "test") & _scope_mask(m, scope)
        sub = m.loc[mask].copy()
        sub_p2 = sub[sub["wind_gust_P2"].notna()]
        sub_s1 = sub[sub["wind_gust_S1"].notna()]
        cnt_p2_u95 = int((sub_p2["wind_gust_P2"] > u95).sum())
        cnt_s1_u95 = int((sub_s1["wind_gust_S1"] > u95).sum())
        cnt_p2_u99 = int((sub_p2["wind_gust_P2"] > u99).sum())
        cnt_s1_u99 = int((sub_s1["wind_gust_S1"] > u99).sum())
        d_u95 = np.nan if cnt_p2_u95 == 0 else float((cnt_s1_u95 - cnt_p2_u95) / cnt_p2_u95)
        d_u99 = np.nan if cnt_p2_u99 == 0 else float((cnt_s1_u99 - cnt_p2_u99) / cnt_p2_u99)
        tail_rows.append(
            {
                "phase": "test",
                "scope": scope,
                "threshold_u95": u95,
                "threshold_u99": u99,
                "count_exceed_p2_u95": cnt_p2_u95,
                "count_exceed_s1_u95": cnt_s1_u95,
                "delta_exceed_u95_rel": d_u95,
                "count_exceed_p2_u99": cnt_p2_u99,
                "count_exceed_s1_u99": cnt_s1_u99,
                "delta_exceed_u99_rel": d_u99,
            }
        )
    tailshape = pd.DataFrame(tail_rows)
    tailshape.to_csv(paths.synthetic / "synthetic_tailshape_metrics.csv", index=False)
    return syn, tailshape


def _make_figure_f7(paths: Paths, synthetic_metrics: pd.DataFrame, synthetic_tailshape: pd.DataFrame) -> None:
    f7 = paths.figures / "F7_synthetic_mnar_vs_tailshift.pdf"
    fs = paths.synthetic / "synthetic_mnar_vs_tailshift.pdf"
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    sub = synthetic_metrics[synthetic_metrics["phase"] == "test"].copy()
    if sub.empty:
        ax[0].text(0.5, 0.5, "No synthetic test metrics", ha="center")
    else:
        ax[0].scatter(sub["state_wind_spread_vs_P2"], 100.0 * sub["delta_oper_p99_vs_P2"], s=45)
        for _, r in sub.iterrows():
            ax[0].annotate(r["scope"], (r["state_wind_spread_vs_P2"], 100.0 * r["delta_oper_p99_vs_P2"]), fontsize=8)
    ax[0].set_xlabel("MNAR spread vs P2")
    ax[0].set_ylabel("Delta oper p99 vs P2 (%)")
    ax[0].set_title("MNAR vs tail-shift")

    ts = synthetic_tailshape[synthetic_tailshape["phase"] == "test"].copy()
    if ts.empty:
        ax[1].text(0.5, 0.5, "No synthetic tail-shape metrics", ha="center")
    else:
        dirty = ts[ts["scope"] == "dirty"]
        if dirty.empty:
            dirty = ts.head(1)
        r = dirty.iloc[0]
        vals = [100.0 * float(r["delta_exceed_u95_rel"]) if not np.isnan(r["delta_exceed_u95_rel"]) else np.nan, 100.0 * float(r["delta_exceed_u99_rel"]) if not np.isnan(r["delta_exceed_u99_rel"]) else np.nan]
        ax[1].bar(["u95", "u99"], vals, color=["steelblue", "indianred"])
        ax[1].set_ylabel("Delta exceedance (%)")
        ax[1].set_title("Exceedance distortion (TEST DIRTY)")

    fig.suptitle("F7 Synthetic: MNAR and tail-shape")
    plt.tight_layout()
    plt.savefig(f7)
    plt.savefig(fs)
    plt.close()


def _validate_required_artifacts(paths: Paths, p5_available: bool, budgets: tuple[float, ...]) -> None:
    required = [
        "preregister_v3.json",
        "selection_report.json",
        "selected_stations.csv",
        "case_stations.csv",
        "P0_raw_canon.parquet",
        "flags_base.parquet",
        "budget_search_grid_v3.csv",
        "budget_holdout_summary_v3.csv",
        "metrics_missingness.csv",
        "metrics_tails.csv",
        "metrics_mnar.csv",
        "metrics_removed_vs_kept.csv",
        "metrics_removal_by_decile.csv",
        "metrics_suspect_burden.csv",
        "metrics_topk.csv",
        "metrics_p1_vs_p4_comparison.csv",
        "metrics_p3_vs_p5_dqc.csv",
        "metrics_exceedance_retention.csv",
        "tab_policy_test_exceedance_summary.csv",
        "tab_policy_test_exceedance_summary.tex",
        "pareto_drop_rules.csv",
        "checklist_acceptance_v4.json",
        "decision_summary_v4.md",
        "figures/F1_pareto.pdf",
        "figures/F2_common_vs_operational_tails.pdf",
        "figures/F3_mnar_spread_deciles.pdf",
        "figures/F4_tradeoff.pdf",
        "figures/F5_station_heterogeneity.pdf",
        "figures/F6_downstream_cost_tail.pdf",
        "figures/F7_synthetic_mnar_vs_tailshift.pdf",
        "figures/F8_benchmark_detector_power.pdf",
        "synthetic/S1_data.parquet",
        "synthetic/synthetic_metrics.csv",
        "synthetic/synthetic_tailshape_metrics.csv",
        "synthetic/synthetic_mnar_vs_tailshift.pdf",
        "benchmark_stuck/benchmark_manifest.json",
        "benchmark_stuck/benchmark_labels.parquet",
        "benchmark_stuck/benchmark_perf_by_config.csv",
        "benchmark_stuck/benchmark_perf_selected_config.json",
        "downstream/downstream_skill_vs_coverage_v3.csv",
        "downstream/downstream_cost_sensitive_v3.csv",
        "downstream/downstream_tail_metrics_v3.csv",
    ]
    for p in ["P0", "P1", "P2", "P3", "P4", "P5"]:
        required.append(f"products/{p}/data.parquet")
        required.append(f"products/{p}/manifest.json")
    for B in budgets:
        required.append(_budget_best_filename(float(B)))
    if not p5_available:
        required.append("products/P5/data.parquet")
        required.append("products/P5/manifest.json")

    missing = [str(paths.results / r) for r in required if not (paths.results / r).exists()]
    _require(len(missing) == 0, f"Missing required artifacts: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run QC with guarantees v3 full pipeline for Paper 2.")
    parser.add_argument("--input", default="avamet_all.parquet")
    parser.add_argument("--workdir", default=".")
    parser.add_argument("--budgets", default=None, help="Comma-separated missingness budgets (e.g., 0.05,0.10).")
    parser.add_argument("--c6-enabled", choices=["true", "false"], default="true", help="Whether benchmark anti-gaming C6 is enforced.")
    parser.add_argument("--selected-budget", type=float, default=None, help="Budget used to materialize P3/checklist (defaults to 0.10 if present, else max budget).")
    args = parser.parse_args()

    global B_SET, C6_ENABLED
    if args.budgets:
        parsed = []
        for token in str(args.budgets).split(","):
            tok = token.strip()
            if not tok:
                continue
            parsed.append(float(tok))
        _require(len(parsed) > 0, "Argument --budgets was provided but no valid budget value was parsed.")
        B_SET = tuple(sorted(set(parsed)))
    C6_ENABLED = args.c6_enabled.lower() == "true"

    np.random.seed(SEED_GLOBAL)

    root = Path(args.workdir).resolve()
    paths = _ensure_dirs(root)

    con = duckdb.connect()
    _build_dedup_table(con, args.input)

    av = _station_availability(con)
    eligible = av[
        (av["availability_n_pressure"] >= 0.80)
        & (av["availability_n_wind_mean"] >= 0.80)
        & (av["availability_n_wind_gust"] >= 0.80)
    ].copy()
    _require(not eligible.empty, "No eligible stations with availability >=0.80 for pressure+wind_mean+wind_gust.")

    units = _unit_station_train(con, eligible_ids=eligible["station_id"].tolist())
    _build_canonical_eligible(con, eligible=eligible, units=units)
    _write_phaseA_outputs(con, paths)

    selected, case_df, selection_report = _compute_ci_and_select(con, eligible=eligible, units=units, paths=paths)

    base = _load_selected_canonical(con, selected)
    med_delta_train_s = _compute_station_med_delta_train(base)
    base = _add_stuck_grid(base, med_delta_train_s=med_delta_train_s)

    # Benchmark detector power (anti-trivialization) on injected stuck events.
    benchmark_df, benchmark_labels = _build_stuck_benchmark(paths, base=base, selected=selected, med_delta_train_s=med_delta_train_s)
    benchmark_perf = _evaluate_stuck_benchmark_configs(paths, bench_df=benchmark_df, med_delta_train_s=med_delta_train_s)
    _make_figure_f8(paths, bench_perf=benchmark_perf)
    benchmark_perf_map: dict[str, dict[str, float]] = {}
    for _, rr in benchmark_perf.iterrows():
        benchmark_perf_map[str(rr["config_key"])] = {
            "recall_stuck": float(rr["recall_stuck"]),
            "precision_stuck": float(rr["precision_stuck"]),
        }

    # Pre-register gamma from P0 TRAIN before optimization.
    keep_p0 = _build_product_keep_mask(base, "P0")
    suspect_p0 = _build_product_suspect_mask(base, "P0")
    p0_preview = _apply_product(base, product="P0", keep_mask=keep_p0, suspect_mask=suspect_p0)
    gamma_value, gamma_alpha = _estimate_gamma_from_p0_train(p0_preview)
    _write_preregister(paths, gamma_value=gamma_value)

    # Budgeted search full and holdout folds.
    folds = _stratified_station_folds(selected)

    budget_grid_rows = []
    holdout_rows = []
    best_full: dict[float, dict[str, Any]] = {}

    station_all = set(selected["station_id"].tolist())
    for B in B_SET:
        grid_full, best_cfg = _search_budget_for_station_subset(
            base,
            station_ids=station_all,
            B=float(B),
            phase="train",
            benchmark_perf_map=benchmark_perf_map,
        )
        grid_full["fold"] = "full"
        budget_grid_rows.append(grid_full)
        _require(best_cfg is not None, f"No compliant P3 config in TRAIN for B={B} on full selected stations.")
        best_full[float(B)] = best_cfg

        # validate full on TEST
        keep_test = _build_product_keep_mask(base, "P3", config=best_cfg)
        suspect_test = _build_product_suspect_mask(base, "P3", config=best_cfg)
        summ_test, _, _ = _evaluate_product_scope_phase(
            base,
            keep_mask=keep_test,
            suspect_mask=suspect_test,
            phase="test",
            scope="dirty",
            k_top=K_TOP,
        )
        holdout_rows.append(
            {
                "fold": "full",
                "B": float(B),
                "selected_config": _config_key(best_cfg),
                "phase_eval": "test",
                **summ_test,
                "constraint_missing": bool((not np.isnan(summ_test["missing_additional"])) and (summ_test["missing_additional"] <= B)),
                "constraint_common": bool(
                    (not np.isnan(summ_test["delta_common_p99"]))
                    and (not np.isnan(summ_test["delta_common_p999"]))
                    and (abs(summ_test["delta_common_p99"]) <= EPSILON)
                    and (abs(summ_test["delta_common_p999"]) <= EPSILON)
                ),
                "constraint_mnar": bool((not np.isnan(summ_test["state_wind_spread"])) and (summ_test["state_wind_spread"] <= TAU)),
            }
        )

    # Holdout folds
    for fold in [0, 1, 2]:
        holdout_ids = set(folds[folds["fold"] == fold]["station_id"].tolist())
        dev_ids = station_all - holdout_ids
        for B in B_SET:
            grid_dev, best_cfg = _search_budget_for_station_subset(
                base,
                station_ids=dev_ids,
                B=float(B),
                phase="train",
                benchmark_perf_map=benchmark_perf_map,
            )
            grid_dev["fold"] = f"fold_{fold}"
            budget_grid_rows.append(grid_dev)

            if best_cfg is None:
                holdout_rows.append(
                    {
                        "fold": f"fold_{fold}",
                        "B": float(B),
                        "selected_config": None,
                        "phase_eval": "test",
                        "scope": "dirty",
                        "missing_additional": np.nan,
                        "delta_common_p99": np.nan,
                        "delta_common_p999": np.nan,
                        "state_wind_spread": np.nan,
                        "constraint_missing": False,
                        "constraint_common": False,
                        "constraint_mnar": False,
                    }
                )
                continue

            sub_hold = base[base["station_id"].isin(holdout_ids)].copy()
            keep_hold = _build_product_keep_mask(sub_hold, "P3", config=best_cfg)
            suspect_hold = _build_product_suspect_mask(sub_hold, "P3", config=best_cfg)
            summ, _, _ = _evaluate_product_scope_phase(
                sub_hold,
                keep_mask=keep_hold,
                suspect_mask=suspect_hold,
                phase="test",
                scope="dirty",
                k_top=K_TOP,
            )
            holdout_rows.append(
                {
                    "fold": f"fold_{fold}",
                    "B": float(B),
                    "selected_config": _config_key(best_cfg),
                    "phase_eval": "test",
                    **summ,
                    "constraint_missing": bool((not np.isnan(summ["missing_additional"])) and (summ["missing_additional"] <= B)),
                    "constraint_common": bool(
                        (not np.isnan(summ["delta_common_p99"]))
                        and (not np.isnan(summ["delta_common_p999"]))
                        and (abs(summ["delta_common_p99"]) <= EPSILON)
                        and (abs(summ["delta_common_p999"]) <= EPSILON)
                    ),
                    "constraint_mnar": bool((not np.isnan(summ["state_wind_spread"])) and (summ["state_wind_spread"] <= TAU)),
                }
            )

    budget_grid = pd.concat(budget_grid_rows, ignore_index=True)
    budget_grid.to_csv(paths.results / "budget_search_grid_v3.csv", index=False)

    holdout = pd.DataFrame(holdout_rows)
    holdout["constraints_all"] = holdout[["constraint_missing", "constraint_common", "constraint_mnar"]].fillna(False).all(axis=1)
    holdout.to_csv(paths.results / "budget_holdout_summary_v3.csv", index=False)

    for B, cfg in best_full.items():
        (paths.results / _budget_best_filename(float(B))).write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    selected_budget = float(args.selected_budget) if args.selected_budget is not None else (0.10 if 0.10 in best_full else max(best_full))
    selected_budget_key = None
    for k in best_full:
        if abs(float(k) - float(selected_budget)) <= 1e-12:
            selected_budget_key = float(k)
            break
    _require(selected_budget_key is not None, f"Selected budget {selected_budget} is not in active B_SET={list(B_SET)}")
    best_selected = best_full[selected_budget_key]

    alt_budget_key = None
    for k in sorted(best_full):
        if abs(float(k) - float(selected_budget_key)) > 1e-12:
            alt_budget_key = float(k)
            break
    alt_cfg = None if alt_budget_key is None else best_full[alt_budget_key]

    perf_sel = benchmark_perf[benchmark_perf["config_key"] == _config_key(best_selected)].copy()
    _require(not perf_sel.empty, f"Selected P3 config not found in benchmark performance: {_config_key(best_selected)}")
    perf_sel_row = perf_sel.iloc[0].to_dict()
    perf_sel_row_out = {}
    for k, v in perf_sel_row.items():
        if isinstance(v, (np.integer,)):
            perf_sel_row_out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            perf_sel_row_out[k] = float(v)
        else:
            perf_sel_row_out[k] = v
    (paths.benchmark / "benchmark_perf_selected_config.json").write_text(json.dumps(perf_sel_row_out, indent=2), encoding="utf-8")

    # Build products
    keep_masks = {
        "P0": _build_product_keep_mask(base, "P0"),
        "P1": _build_product_keep_mask(base, "P1"),
        "P2": _build_product_keep_mask(base, "P2"),
        "P3": _build_product_keep_mask(base, "P3", config=best_selected),
        "P4": _build_product_keep_mask(base, "P4"),
        "P5": _build_product_keep_mask(base, "P5"),
    }
    suspect_masks = {
        "P0": _build_product_suspect_mask(base, "P0"),
        "P1": _build_product_suspect_mask(base, "P1"),
        "P2": _build_product_suspect_mask(base, "P2"),
        "P3": _build_product_suspect_mask(base, "P3", config=best_selected),
        "P4": _build_product_suspect_mask(base, "P4"),
        "P5": _build_product_suspect_mask(base, "P5"),
    }

    product_data: dict[str, pd.DataFrame] = {}
    manifests: dict[str, dict[str, Any]] = {
        "P0": {"policy": "raw_canon", "unit": "convert_only", "drop_fail": False, "drop_suspect": False, "quality_weights": True},
        "P1": {"policy": "strict_censor", "drop_fail": True, "drop_suspect": True, "unit_censor": True, "T": T_REF, "tol": TOL_REF, "quality_weights": True},
        "P2": {"policy": "convert_verify", "drop_fail": True, "drop_suspect": False, "unit_censor": False, "quality_weights": True},
        "P3": {
            "policy": "budgeted_v3",
            "selected_for_B": float(selected_budget_key),
            "config": best_selected,
            "alt_config": alt_cfg,
            "quality_weights": True,
            "c6_enabled": bool(C6_ENABLED),
        },
        "P4": {
            "policy": "policy_baseline",
            "description": "baseline generalista de consumo de flags sin garantias (drop FAIL+SUSPECT, sin censura de unidades)",
            "drop_fail": True,
            "drop_suspect": True,
            "unit_censor": False,
            "T": T_REF,
            "tol": TOL_REF,
            "quality_weights": True,
        },
        "P5": {
            "policy": "domain_rule_qc",
            "description": "fixed meteorological domain-rule baseline; no budget search and no tuning on outcomes",
            "drop_fail": True,
            "drop_suspect": True,
            "unit_censor": False,
            "T": T_REF,
            "tol": TOL_REF,
            "rules": [
                "pressure_zero_fail",
                "pressure_out_of_range_after_unit_normalization",
                "wind_gust_less_than_wind_mean",
                "negative_wind",
                "pressure_persistence_stuck_T3_tol0.10",
                "wind_direction_sentinel_or_out_of_range_masked",
            ],
            "quality_weights": True,
        },
    }

    for p in ["P0", "P1", "P2", "P3", "P4", "P5"]:
        d = _apply_product(base, product=p, keep_mask=keep_masks[p], suspect_mask=suspect_masks[p])
        product_data[p] = d
        _write_product_data(paths, p, d, manifests[p])
    p5_available = bool("P5" in product_data and not product_data["P5"].empty)
    _require(p5_available, "P5-DQC domain-rule baseline product is empty.")

    tails, miss, mnar, topk, removed_vs_kept, removal_by_decile, suspect_burden = _aggregate_metrics(base, product_data, paths)

    # Pareto
    pareto = _save_pareto(paths, base, product_data, p3_cfg=best_selected)

    # Core figures F1-F4
    _make_figures(
        paths,
        tails=tails,
        miss=miss,
        mnar=mnar,
        topk=topk,
        pareto=pareto,
        suspect_burden=suspect_burden,
        budget_grid=budget_grid,
        best_configs=best_full,
        selected_budget=float(selected_budget_key),
        removed_vs_kept=removed_vs_kept,
        removal_by_decile=removal_by_decile,
    )

    # F5 heterogeneity
    hetero = _station_level_heterogeneity(base, product_data)
    hetero.to_csv(paths.results / "metrics_station_heterogeneity.csv", index=False)
    _make_figure_f5(paths, hetero)

    # Downstream v3 + F6
    skill_df, cost_df, tail_df = _downstream_eval_v3(paths, product_data, gamma_value=gamma_value)
    _make_figure_f6(paths, cost_df=cost_df, tail_df=tail_df)

    # Real TEST exceedance retention at fixed P0-TRAIN thresholds.
    exceedance_retention, policy_exceedance_summary = _compute_exceedance_retention_tables(
        paths,
        product_data=product_data,
        miss=miss,
        mnar=mnar,
        cost_df=cost_df,
    )

    # Synthetic stress test + F7
    synthetic_metrics, synthetic_tailshape = _run_synthetic_stress(paths, base=base, product_data=product_data)
    _make_figure_f7(paths, synthetic_metrics=synthetic_metrics, synthetic_tailshape=synthetic_tailshape)

    # P1 vs P4 comparativo
    miss_cmp = miss[(miss["variable"] == "wind_gust") & (miss["product"].isin(["P1", "P4"]))][
        ["product", "phase", "scope", "missing_additional"]
    ].pivot(index=["phase", "scope"], columns="product", values="missing_additional").reset_index()
    mnar_cmp = mnar[mnar["product"].isin(["P1", "P4"])][["product", "phase", "scope", "state_wind_spread"]]
    mnar_cmp = mnar_cmp.pivot(index=["phase", "scope"], columns="product", values="state_wind_spread").reset_index()
    tails_cmp = tails[(tails["variable"] == "wind_gust") & (tails["product"].isin(["P1", "P4"]))][
        ["product", "phase", "scope", "delta_common_p99", "delta_oper_p99", "delta_common_p999", "delta_oper_p999"]
    ]
    tails_p1 = tails_cmp[tails_cmp["product"] == "P1"].drop(columns=["product"]).rename(
        columns={
            "delta_common_p99": "delta_common_p99_P1",
            "delta_oper_p99": "delta_oper_p99_P1",
            "delta_common_p999": "delta_common_p999_P1",
            "delta_oper_p999": "delta_oper_p999_P1",
        }
    )
    tails_p4 = tails_cmp[tails_cmp["product"] == "P4"].drop(columns=["product"]).rename(
        columns={
            "delta_common_p99": "delta_common_p99_P4",
            "delta_oper_p99": "delta_oper_p99_P4",
            "delta_common_p999": "delta_common_p999_P4",
            "delta_oper_p999": "delta_oper_p999_P4",
        }
    )
    p14 = miss_cmp.merge(mnar_cmp, on=["phase", "scope"], how="outer", suffixes=("_miss", "_mnar"))
    p14 = p14.merge(tails_p1, on=["phase", "scope"], how="left").merge(tails_p4, on=["phase", "scope"], how="left")
    p14 = p14.rename(columns={"P1_miss": "missing_additional_P1", "P4_miss": "missing_additional_P4", "P1_mnar": "state_wind_spread_P1", "P4_mnar": "state_wind_spread_P4"})
    p14["missing_additional_diff_P1_minus_P4"] = p14["missing_additional_P1"] - p14["missing_additional_P4"]
    p14["state_wind_spread_diff_P1_minus_P4"] = p14["state_wind_spread_P1"] - p14["state_wind_spread_P4"]
    p14.to_csv(paths.results / "metrics_p1_vs_p4_comparison.csv", index=False)

    # P3 vs P5 fixed domain-rule QC baseline.
    p3m = mnar[(mnar["product"] == "P3") & (mnar["phase"] == "test") & (mnar["scope"] == "dirty")][["state_wind_spread"]]
    p5m = mnar[(mnar["product"] == "P5") & (mnar["phase"] == "test") & (mnar["scope"] == "dirty")][["state_wind_spread"]]
    p3miss = miss[(miss["product"] == "P3") & (miss["phase"] == "test") & (miss["scope"] == "dirty") & (miss["variable"] == "wind_gust")]
    p5miss = miss[(miss["product"] == "P5") & (miss["phase"] == "test") & (miss["scope"] == "dirty") & (miss["variable"] == "wind_gust")]
    out_p35 = pd.DataFrame(
        [
            {
                "status": "available",
                "baseline": "fixed_domain_rule_qc",
                "p3_state_wind_spread_test_dirty": float(p3m.iloc[0]["state_wind_spread"]) if not p3m.empty else np.nan,
                "p5_dqc_state_wind_spread_test_dirty": float(p5m.iloc[0]["state_wind_spread"]) if not p5m.empty else np.nan,
                "p3_missing_additional_test_dirty": float(p3miss.iloc[0]["missing_additional"]) if not p3miss.empty else np.nan,
                "p5_dqc_missing_additional_test_dirty": float(p5miss.iloc[0]["missing_additional"]) if not p5miss.empty else np.nan,
            }
        ]
    )
    out_p35.to_csv(paths.results / "metrics_p3_vs_p5_dqc.csv", index=False)

    # Budget sensitivity report (top-3 full/train)
    top3_rows = []
    for B in B_SET:
        sub = budget_grid[
            (budget_grid["fold"] == "full")
            & (budget_grid["phase"] == "train")
            & (budget_grid["B"] == float(B))
            & (budget_grid["compliant"] == True)  # noqa: E712
        ].copy()
        sub = sub.sort_values(["rank_train"]).head(3)
        if sub.empty:
            continue
        spread_u = float(sub["utility_v3_train"].max() - sub["utility_v3_train"].min())
        for _, r in sub.iterrows():
            top3_rows.append(
                {
                    "B": float(B),
                    "config_key": r["config_key"],
                    "rank_train": int(r["rank_train"]),
                    "rank_test": int(r["rank_test"]),
                    "utility_v3_train": float(r["utility_v3_train"]),
                    "utility_v3_test": float(r["utility_v3_test"]),
                    "missing_additional_train": float(r["missing_additional_train"]),
                    "state_wind_spread_train": float(r["state_wind_spread_train"]),
                    "suspect_rate_train": float(r["suspect_rate_train"]),
                    "top3_utility_spread_train": spread_u,
                }
            )
    pd.DataFrame(top3_rows).to_csv(paths.results / "budget_top3_stability.csv", index=False)

    # Checklist C1-C6 (v4)
    checklist_budget = 0.10 if any(abs(float(b) - 0.10) <= 1e-12 for b in B_SET) else max(B_SET)
    c1_rows = holdout[(holdout["B"] == float(checklist_budget)) & (holdout["fold"].str.startswith("fold_"))]
    c1_pass = int(c1_rows["constraints_all"].sum()) >= 2

    # C2v2.1 spread ratio P1/P2 in train+test dirty
    p1_spread_train = _scalar_from(mnar, {"product": "P1", "phase": "train", "scope": "dirty"}, "state_wind_spread")
    p2_spread_train = _scalar_from(mnar, {"product": "P2", "phase": "train", "scope": "dirty"}, "state_wind_spread")
    p1_spread_test = _scalar_from(mnar, {"product": "P1", "phase": "test", "scope": "dirty"}, "state_wind_spread")
    p2_spread_test = _scalar_from(mnar, {"product": "P2", "phase": "test", "scope": "dirty"}, "state_wind_spread")
    spread_ratio_train = _safe_ratio(p1_spread_train, p2_spread_train)
    spread_ratio_test = _safe_ratio(p1_spread_test, p2_spread_test)
    c2v21 = bool(
        (not np.isnan(spread_ratio_train))
        and (not np.isnan(spread_ratio_test))
        and (spread_ratio_train >= C2V2_SPREAD_RATIO_R)
        and (spread_ratio_test >= C2V2_SPREAD_RATIO_R)
    )

    # C2v2.2 missing ratio P1/P2 in train+test dirty
    p1_miss_train = _scalar_from(miss, {"product": "P1", "phase": "train", "scope": "dirty", "variable": "wind_gust"}, "missing_additional")
    p2_miss_train = _scalar_from(miss, {"product": "P2", "phase": "train", "scope": "dirty", "variable": "wind_gust"}, "missing_additional")
    p1_miss_test = _scalar_from(miss, {"product": "P1", "phase": "test", "scope": "dirty", "variable": "wind_gust"}, "missing_additional")
    p2_miss_test = _scalar_from(miss, {"product": "P2", "phase": "test", "scope": "dirty", "variable": "wind_gust"}, "missing_additional")
    miss_ratio_train = _safe_ratio(p1_miss_train, p2_miss_train)
    miss_ratio_test = _safe_ratio(p1_miss_test, p2_miss_test)
    c2v22 = bool(
        (not np.isnan(miss_ratio_train))
        and (not np.isnan(miss_ratio_test))
        and (miss_ratio_train >= C2V2_MISSING_RATIO_M)
        and (miss_ratio_test >= C2V2_MISSING_RATIO_M)
    )

    # C2v2.3 removed-vs-kept evidence in train+test dirty
    p1_gap_train = _scalar_from(removed_vs_kept, {"product": "P1", "phase": "train", "scope": "dirty"}, "p99_gap_rel")
    p1_gap_test = _scalar_from(removed_vs_kept, {"product": "P1", "phase": "test", "scope": "dirty"}, "p99_gap_rel")
    c2v23_train = bool(
        ((not np.isnan(p1_gap_train)) and (p1_gap_train >= C2V2_REMOVED_KEPT_DELTA_MIN_TRAIN))
        or ((not np.isnan(p1_spread_train)) and (p1_spread_train >= C2V2_SPREAD_MIN))
    )
    c2v23_test = bool(
        ((not np.isnan(p1_gap_test)) and (p1_gap_test >= C2V2_REMOVED_KEPT_DELTA_MIN_TEST))
        or ((not np.isnan(p1_spread_test)) and (p1_spread_test >= C2V2_SPREAD_MIN))
    )
    c2v23 = bool(c2v23_train and c2v23_test)
    c2v2_pass = bool(c2v21 and c2v22)

    # C3: P3 reduces MNAR and missing vs P1/P4 + common invariance
    p3_spread_test = _scalar_from(mnar, {"product": "P3", "phase": "test", "scope": "dirty"}, "state_wind_spread")
    p1_spread_test_ref = _scalar_from(mnar, {"product": "P1", "phase": "test", "scope": "dirty"}, "state_wind_spread")
    p4_spread_test_ref = _scalar_from(mnar, {"product": "P4", "phase": "test", "scope": "dirty"}, "state_wind_spread")
    p3_miss_test = _scalar_from(miss, {"product": "P3", "phase": "test", "scope": "dirty", "variable": "wind_gust"}, "missing_additional")
    p1_miss_test_ref = _scalar_from(miss, {"product": "P1", "phase": "test", "scope": "dirty", "variable": "wind_gust"}, "missing_additional")
    p4_miss_test_ref = _scalar_from(miss, {"product": "P4", "phase": "test", "scope": "dirty", "variable": "wind_gust"}, "missing_additional")
    p3_dcommon_p99 = _scalar_from(tails, {"product": "P3", "phase": "test", "scope": "dirty", "variable": "wind_gust"}, "delta_common_p99")
    p3_dcommon_p999 = _scalar_from(tails, {"product": "P3", "phase": "test", "scope": "dirty", "variable": "wind_gust"}, "delta_common_p999")
    c3 = bool(
        (not np.isnan(p3_spread_test))
        and (not np.isnan(p1_spread_test_ref))
        and (not np.isnan(p4_spread_test_ref))
        and (p3_spread_test < p1_spread_test_ref)
        and (p3_spread_test < p4_spread_test_ref)
        and (not np.isnan(p3_miss_test))
        and (not np.isnan(p1_miss_test_ref))
        and (not np.isnan(p4_miss_test_ref))
        and (p3_miss_test < p1_miss_test_ref)
        and (p3_miss_test < p4_miss_test_ref)
        and (not np.isnan(p3_dcommon_p99))
        and (not np.isnan(p3_dcommon_p999))
        and (abs(p3_dcommon_p99) <= EPSILON)
        and (abs(p3_dcommon_p999) <= EPSILON)
    )

    c4 = bool(
        (paths.results / "products" / "P4" / "data.parquet").exists()
        and (paths.results / "products" / "P4" / "manifest.json").exists()
        and (paths.results / "products" / "P5" / "data.parquet").exists()
        and (paths.results / "products" / "P5" / "manifest.json").exists()
        and (paths.results / "metrics_p1_vs_p4_comparison.csv").exists()
        and (paths.results / "metrics_p3_vs_p5_dqc.csv").exists()
        and (paths.results / "metrics_exceedance_retention.csv").exists()
    )

    c5_model = "ridge"
    score_p3 = _scalar_from(cost_df, {"product": "P3", "model": c5_model, "phase": "test", "scope": "global"}, "score_cost_sensitive")
    score_p0 = _scalar_from(cost_df, {"product": "P0", "model": c5_model, "phase": "test", "scope": "global"}, "score_cost_sensitive")
    score_p1 = _scalar_from(cost_df, {"product": "P1", "model": c5_model, "phase": "test", "scope": "global"}, "score_cost_sensitive")
    c5 = bool(
        (not np.isnan(score_p3))
        and (not np.isnan(score_p0))
        and (not np.isnan(score_p1))
        and (score_p3 <= score_p0)
        and (score_p3 <= score_p1)
    )

    # C6 detector-power anti-truco
    sel_bench_recall = float(perf_sel_row_out.get("recall_stuck", np.nan))
    sel_bench_precision = float(perf_sel_row_out.get("precision_stuck", np.nan))
    if C6_ENABLED:
        c6 = bool(
            (not np.isnan(sel_bench_recall))
            and (not np.isnan(sel_bench_precision))
            and (sel_bench_recall >= BENCHMARK_RECALL_MIN)
            and (sel_bench_precision >= BENCHMARK_PRECISION_MIN)
        )
    else:
        c6 = True

    submit_status = bool(c1_pass and c2v2_pass and c3 and c4 and c5 and c6)
    envsoft_ready = bool(submit_status and c5)
    checklist = {
        "version": "v4",
        "gamma": gamma_value,
        "gamma_alpha": gamma_alpha,
        "thresholds": {
            "spread_ratio_r": C2V2_SPREAD_RATIO_R,
            "missing_ratio_m": C2V2_MISSING_RATIO_M,
            "removed_vs_kept_delta_min_train": C2V2_REMOVED_KEPT_DELTA_MIN_TRAIN,
            "removed_vs_kept_delta_min_test": C2V2_REMOVED_KEPT_DELTA_MIN_TEST,
            "spread_min": C2V2_SPREAD_MIN,
            "benchmark_recall_min": BENCHMARK_RECALL_MIN,
            "benchmark_precision_min": BENCHMARK_PRECISION_MIN,
        },
        "C1": bool(c1_pass),
        "C2v2": {
            "pass": bool(c2v2_pass),
            "C2v2.1_pass": bool(c2v21),
            "C2v2.2_pass": bool(c2v22),
            "C2v2.3_pass": bool(c2v23),
            "spread_ratio_train": spread_ratio_train,
            "spread_ratio_test": spread_ratio_test,
            "missing_ratio_train": miss_ratio_train,
            "missing_ratio_test": miss_ratio_test,
            "p99_gap_rel_train": p1_gap_train,
            "p99_gap_rel_test": p1_gap_test,
            "required_subchecks": ["C2v2.1", "C2v2.2"],
        },
        "C3": bool(c3),
        "C4": bool(c4),
        "C4_detail": {
            "p4_policy_baseline_available": bool((paths.results / "products" / "P4" / "manifest.json").exists()),
            "p5_domain_rule_qc_available": bool(p5_available),
            "p5_policy": "domain_rule_qc",
            "exceedance_retention_table": "metrics_exceedance_retention.csv",
        },
        "C5": bool(c5),
        "C5_detail": {
            "model": c5_model,
            "scope": "global",
            "score_P3": score_p3,
            "score_P0": score_p0,
            "score_P1": score_p1,
        },
        "C6": bool(c6),
        "C6_detail": {
            "selected_config": _config_key(best_selected),
            "recall_stuck": sel_bench_recall,
            "precision_stuck": sel_bench_precision,
            "c6_enabled": bool(C6_ENABLED),
        },
        "submit_status": "SUBMIT" if submit_status else "NO_SUBMIT",
        "envsoft_ready": bool(envsoft_ready),
    }
    (paths.results / "checklist_acceptance_v4.json").write_text(json.dumps(checklist, indent=2), encoding="utf-8")

    pareto_top3_share = {}
    for prod in ["P1", "P2", "P3", "P4"]:
        p = pareto[(pareto["phase"] == "test") & (pareto["scope"] == "dirty") & (pareto["product"] == prod)].sort_values("rows_n", ascending=False).head(3)
        pareto_top3_share[prod] = float(p["share_pct"].sum()) if not p.empty else np.nan
    _write_decision_summary_v4(paths, checklist, pareto_top3_share=pareto_top3_share, benchmark_selected=perf_sel_row_out)

    _validate_required_artifacts(paths, p5_available=p5_available, budgets=tuple(float(b) for b in B_SET))

    print("Pipeline complete")
    print(json.dumps(checklist, indent=2))


if __name__ == "__main__":
    main()
