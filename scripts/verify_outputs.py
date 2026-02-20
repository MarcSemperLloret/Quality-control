from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _req(path: Path, errors: list[str]) -> None:
    if not path.exists():
        errors.append(f"Missing required artifact: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify required outputs and key metrics.")
    parser.add_argument("--results-dir", default="results_v3", help="Path to results_v3 directory.")
    parser.add_argument(
        "--reference",
        default="paper_artifacts/expected_reference.json",
        help="Reference JSON with expected values for numeric checks.",
    )
    parser.add_argument("--tolerance", type=float, default=1e-6, help="Absolute tolerance for numeric checks.")
    args = parser.parse_args()

    results = Path(args.results_dir)
    ref_path = Path(args.reference)
    errors: list[str] = []

    required = [
        "preregister_v3.json",
        "selected_stations.csv",
        "budget_search_grid_v3.csv",
        "budget_holdout_summary_v3.csv",
        "checklist_acceptance_v4.json",
        "decision_summary_v4.md",
        "benchmark_stuck/benchmark_manifest.json",
        "benchmark_stuck/benchmark_perf_by_config.csv",
        "benchmark_stuck/benchmark_perf_selected_config.json",
        "figures/F1_pareto.pdf",
        "figures/F2_common_vs_operational_tails.pdf",
        "figures/F3_mnar_spread_deciles.pdf",
        "figures/F4_tradeoff.pdf",
        "figures/F5_station_heterogeneity.pdf",
        "figures/F6_downstream_cost_tail.pdf",
        "figures/F7_synthetic_mnar_vs_tailshift.pdf",
        "figures/F8_benchmark_detector_power.pdf",
        "downstream/downstream_cost_sensitive_v3.csv",
        "synthetic/synthetic_metrics.csv",
        "synthetic/synthetic_tailshape_metrics.csv",
    ]
    for rel in required:
        _req(results / rel, errors)

    if errors:
        raise SystemExit("\n".join(["[FAIL] Output verification failed:"] + errors))

    checklist = _load_json(results / "checklist_acceptance_v4.json")
    bench_sel = _load_json(results / "benchmark_stuck/benchmark_perf_selected_config.json")
    grid = pd.read_csv(results / "benchmark_stuck/benchmark_perf_by_config.csv")
    sel = pd.read_csv(results / "selected_stations.csv")

    if sel.shape[0] != 30:
        errors.append(f"selected_stations.csv should have 30 rows, got {sel.shape[0]}")
    by_stratum = sel.groupby("stratum").size().to_dict()
    for s in ["clean", "medium", "dirty"]:
        if int(by_stratum.get(s, 0)) != 10:
            errors.append(f"selected_stations should contain 10 stations in '{s}', got {by_stratum.get(s, 0)}")

    if int(grid.shape[0]) != 30:
        errors.append(f"benchmark_perf_by_config.csv should have 30 rows, got {grid.shape[0]}")

    if checklist.get("submit_status") != "SUBMIT":
        errors.append(f"submit_status expected SUBMIT, got {checklist.get('submit_status')}")

    for k in ["C1", "C3", "C4", "C5", "C6"]:
        if bool(checklist.get(k, False)) is not True:
            errors.append(f"{k} expected true, got {checklist.get(k)}")
    if bool(checklist.get("C2v2", {}).get("pass", False)) is not True:
        errors.append(f"C2v2.pass expected true, got {checklist.get('C2v2', {}).get('pass')}")

    if ref_path.exists():
        ref = _load_json(ref_path)
        cfg_expected = ref.get("selected_config")
        if cfg_expected and str(bench_sel.get("config_key")) != str(cfg_expected):
            errors.append(f"selected config mismatch: expected {cfg_expected}, got {bench_sel.get('config_key')}")

        for key in ["gamma", "recall_stuck", "precision_stuck"]:
            if key not in ref:
                continue
            if key == "gamma":
                val = float(checklist.get("gamma"))
            elif key == "recall_stuck":
                val = float(bench_sel.get("recall_stuck"))
            else:
                val = float(bench_sel.get("precision_stuck"))
            if abs(val - float(ref[key])) > float(args.tolerance):
                errors.append(f"{key} mismatch: expected {ref[key]}, got {val}")

    if errors:
        raise SystemExit("\n".join(["[FAIL] Output verification failed:"] + errors))

    print("[OK] Output verification passed")
    print(f"- results_dir: {results.resolve()}")
    print(f"- selected_config: {bench_sel.get('config_key')}")
    print(f"- C1..C6: {checklist.get('C1')}, {checklist.get('C2v2', {}).get('pass')}, {checklist.get('C3')}, {checklist.get('C4')}, {checklist.get('C5')}, {checklist.get('C6')}")


if __name__ == "__main__":
    main()

