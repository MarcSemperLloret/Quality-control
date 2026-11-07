from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from saqc import SaQC


TRAIN_END = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
TEST_END = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    paper_dir = repo_root
    parser = argparse.ArgumentParser(
        description="SaQC conservative baseline and selection-stress experiment."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=paper_dir / "runs_additional" / "BASELINE" / "results_v3",
    )
    parser.add_argument("--article-dir", type=Path, default=paper_dir / "article")
    return parser.parse_args()


def load_product(results_dir: Path, product: str) -> pd.DataFrame:
    df = pd.read_parquet(
        results_dir / "products" / product / "data.parquet",
        columns=[
            "station_id",
            "timestamp_utc",
            "stratum",
            "wind_gust",
            "wind_mean",
            "pressure_hpa",
        ],
        engine="pyarrow",
    )
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df[df["timestamp_utc"] < TEST_END].reset_index(drop=True)


def phase_mask(df: pd.DataFrame, phase: str) -> pd.Series:
    if phase == "train":
        return df["timestamp_utc"] < TRAIN_END
    if phase == "test":
        return (df["timestamp_utc"] >= TRAIN_END) & (df["timestamp_utc"] < TEST_END)
    raise ValueError(phase)


def saqc_conservative_unavailable(p0: pd.DataFrame) -> pd.Series:
    unavailable = pd.Series(False, index=p0.index)
    fields = ["wind_gust", "wind_mean", "pressure_hpa"]
    for _, part in p0.groupby("station_id", sort=False):
        idx = part.index
        data = part.set_index("timestamp_utc")[fields].sort_index()
        qc = SaQC(data)
        qc = qc.flagRange("wind_gust", min=0.0, max=120.0)
        qc = qc.flagRange("wind_mean", min=0.0, max=80.0)
        qc = qc.flagRange("pressure_hpa", min=850.0, max=1100.0)
        qc = qc.flagConstants("pressure_hpa", thresh=0.05, window="3h", min_periods=3)
        flags = qc.flags.to_pandas().reindex(data.index)
        wind_bad = flags[["wind_gust", "wind_mean"]].gt(0).any(axis=1)
        wind_bad = wind_bad | data["wind_gust"].lt(data["wind_mean"]).fillna(False)
        unavailable.loc[idx] = wind_bad.to_numpy(dtype=bool)
    return unavailable


def wind_unavailable_from_product(df: pd.DataFrame) -> pd.Series:
    return df["wind_gust"].isna()


def state_spread(p0: pd.DataFrame, unavailable: pd.Series, phase: str, scope: str = "dirty") -> float:
    mask = phase_mask(p0, phase) & p0["stratum"].eq(scope) & p0["wind_gust"].notna()
    part = p0.loc[mask, ["wind_gust"]].copy()
    if part.empty:
        return float("nan")
    part["decile"] = pd.qcut(part["wind_gust"], 10, labels=False, duplicates="drop") + 1
    part["unavailable"] = unavailable.loc[mask].to_numpy(dtype=bool)
    rates = part.groupby("decile")["unavailable"].mean()
    # Methods-defined robust spread: Q0.90 - Q0.10 of the removal-by-decile curve
    # (consistent with metrics_mnar / the main MNAR table), not the max-min range.
    return float(rates.quantile(0.90) - rates.quantile(0.10))


def additional_missingness(p0: pd.DataFrame, unavailable: pd.Series, phase: str, scope: str = "dirty") -> float:
    mask = phase_mask(p0, phase) & p0["stratum"].eq(scope)
    raw_missing = p0.loc[mask, "wind_gust"].isna()
    return float(unavailable.loc[mask].mean() - raw_missing.mean())


def retention(p0: pd.DataFrame, unavailable: pd.Series, q: str) -> float:
    train = p0[p0["timestamp_utc"] < TRAIN_END]
    thresholds = train.groupby("station_id")["wind_gust"].quantile({"u95": 0.95, "u99": 0.99}[q])
    mask = phase_mask(p0, "test") & p0["stratum"].eq("dirty") & p0["wind_gust"].notna()
    base = p0.loc[mask, ["station_id", "wind_gust"]]
    event_mask = base["wind_gust"].ge(base["station_id"].map(thresholds))
    idx = base.index[event_mask.to_numpy()]
    if len(idx) == 0:
        return float("nan")
    return float((~unavailable.loc[idx]).mean())


def write_saqc_table(metrics: pd.DataFrame, path: Path) -> None:
    order = ["Delete-all-flags", "SaQC conservative", "Conservative FAIL-only", "Constraint-selected P3"]
    rows = metrics.set_index("policy").loc[order]
    lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Policy & Add. miss. & Spread & Ret. $u95$ & Ret. $u99$ \\\\",
        "\\midrule",
    ]
    for policy, row in rows.iterrows():
        lines.append(
            f"{policy} & {row['test_dirty_additional_missingness']:.4f} & "
            f"{row['test_dirty_state_spread']:.4f} & "
            f"{row['test_dirty_u95_retention']:.3f} & "
            f"{row['test_dirty_u99_retention']:.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_selection_table(summary: pd.DataFrame, path: Path) -> None:
    def latex_escape(value: object) -> str:
        return str(value).replace("_", r"\_")

    lines = [
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Policy choice & TRAIN add. miss. & TEST add. miss. & TEST spread & Stuck recall & Stuck precision \\\\",
        "\\midrule",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"{latex_escape(row['policy_choice'])} & {row['train_additional_missingness']:.4f} & "
            f"{row['test_additional_missingness']:.4f} & {row['test_state_spread']:.4f} & "
            f"{row['recall_stuck']:.3f} & {row['precision_stuck']:.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = args.results_dir / "saqc_selection_stress"
    out_dir.mkdir(parents=True, exist_ok=True)
    args.article_dir.mkdir(parents=True, exist_ok=True)

    p0 = load_product(args.results_dir, "P0")
    p1 = load_product(args.results_dir, "P1")
    p2 = load_product(args.results_dir, "P2")
    p3 = load_product(args.results_dir, "P3")

    unavailable = {
        "Delete-all-flags": wind_unavailable_from_product(p1),
        "SaQC conservative": saqc_conservative_unavailable(p0),
        "Conservative FAIL-only": wind_unavailable_from_product(p2),
        "Constraint-selected P3": wind_unavailable_from_product(p3),
    }

    rows = []
    for policy, mask in unavailable.items():
        rows.append(
            {
                "policy": policy,
                "train_dirty_additional_missingness": additional_missingness(p0, mask, "train"),
                "test_dirty_additional_missingness": additional_missingness(p0, mask, "test"),
                "test_dirty_state_spread": state_spread(p0, mask, "test"),
                "test_dirty_u95_retention": retention(p0, mask, "u95"),
                "test_dirty_u99_retention": retention(p0, mask, "u99"),
            }
        )
    saqc_metrics = pd.DataFrame(rows)
    saqc_metrics.to_csv(out_dir / "saqc_conservative_baseline_metrics.csv", index=False)
    write_saqc_table(saqc_metrics, args.article_dir / "tab_saqc_conservative_baseline.tex")

    grid = pd.read_csv(args.results_dir / "budget_search_grid_v3.csv")
    full = grid[grid["fold"].eq("full")].copy()
    c6 = full[full["config_key"].eq("T3_tol0.05_KEEP")].iloc[0]
    no_c6 = full.sort_values("rank_train").iloc[0]
    manual = c6.copy()
    selection = pd.DataFrame(
        [
            {
                "policy_choice": "Manual conservative FAIL-only",
                "train_additional_missingness": manual["missing_additional_train"],
                "test_additional_missingness": manual["missing_additional_test"],
                "test_state_spread": manual["state_wind_spread_test"],
                "recall_stuck": manual["recall_stuck"],
                "precision_stuck": manual["precision_stuck"],
            },
            {
                "policy_choice": "Selector with C6",
                "train_additional_missingness": c6["missing_additional_train"],
                "test_additional_missingness": c6["missing_additional_test"],
                "test_state_spread": c6["state_wind_spread_test"],
                "recall_stuck": c6["recall_stuck"],
                "precision_stuck": c6["precision_stuck"],
            },
            {
                "policy_choice": f"Scalar-best without C6 ({no_c6['config_key']})",
                "train_additional_missingness": no_c6["missing_additional_train"],
                "test_additional_missingness": no_c6["missing_additional_test"],
                "test_state_spread": no_c6["state_wind_spread_test"],
                "recall_stuck": no_c6["recall_stuck"],
                "precision_stuck": no_c6["precision_stuck"],
            },
        ]
    )
    selection.to_csv(out_dir / "selection_stress_summary.csv", index=False)
    write_selection_table(selection, args.article_dir / "tab_selection_stress.tex")

    manifest = {
        "saqc_version": __import__("saqc").__version__,
        "saqc_baseline": {
            "wind_gust_range": [0.0, 120.0],
            "wind_mean_range": [0.0, 80.0],
            "pressure_hpa_range": [850.0, 1100.0],
            "pressure_constant": {"threshold_hpa": 0.05, "window": "3h", "min_periods": 3},
            "wind_gust_constant": "not used; calm or steady wind can be physically valid and is not deleted by this conservative baseline",
            "wind_internal_consistency": "wind_gust < wind_mean is unavailable",
        },
        "selection_stress": "Compares manual conservative action, selector with C6, and scalar-best grid choice if C6 is ignored.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
