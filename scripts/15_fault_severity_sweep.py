"""Anti-circularity severity sweep for the controlled measurement-fault suite.

The controlled suite in 11_measurement_fault_suite.py injects faults at a single
hand-picked magnitude that sits comfortably above the spatial-redundancy
detection thresholds. A natural reviewer objection is circularity: the detector
may only "find what it was built to find" because the injected magnitude was
chosen large.

This script rebuts that objection without changing the detector. The spatial
thresholds are fixed from TRAIN clean data BEFORE any injection (identical to the
main suite). We then sweep the injected magnitude across a range that crosses the
fixed thresholds and report Rule+spatial evidence recall as a continuous function
of fault severity. A graceful recall curve with a genuine miss regime at small
magnitudes shows that the reported operating point is not a knife-edge artefact of
a generous injection, and that detection is independent of the injection size.

It reuses the exact loader, neighbour, threshold, event-selection, rule, spatial,
and metric functions from 11_measurement_fault_suite.py.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use("Agg")


def _load_suite_module():
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "measurement_fault_suite", here / "11_measurement_fault_suite.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SUITE = _load_suite_module()


# Severity grids per fault. Units: hPa for pressure, gust units above TRAIN u99
# for the wind spike. The main suite uses offset=20 hPa, drift final=20 hPa, and
# spike >= u99+30; these are marked as the reference operating points below.
OFFSET_HPA = [3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0]
DRIFT_FINAL_HPA = [3.0, 5.0, 8.0, 12.0, 20.0, 30.0, 40.0]
SPIKE_ABOVE_U99 = [0.0, 5.0, 10.0, 20.0, 30.0, 45.0, 60.0]

REFERENCE = {"pressure_offset": 20.0, "pressure_drift": 20.0, "wind_spike_burst": 30.0}


def apply_fault_with_magnitude(
    test_dirty: pd.DataFrame,
    events: pd.DataFrame,
    spec,
    thresholds: pd.DataFrame,
    magnitude: float,
):
    """Inject a single fault family at a parametrised magnitude.

    Mirrors apply_faults() in the main suite but with the magnitude exposed as an
    argument instead of the hard-coded constant.
    """
    faulted = test_dirty.sort_values(["station_id", "timestamp_utc"]).copy()
    faulted["_injected"] = False
    faulted = faulted.reset_index(drop=True)
    label_rows: list[dict[str, object]] = []
    context_rows: list[dict[str, object]] = []
    context = 6

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

        if spec.key == "pressure_offset":
            faulted.loc[locs, "pressure_hpa"] = original_pressure + magnitude
        elif spec.key == "pressure_drift":
            faulted.loc[locs, "pressure_hpa"] = original_pressure + np.linspace(0.0, magnitude, len(locs))
        elif spec.key == "wind_spike_burst":
            u99 = float(thresholds.loc[station_id, "wind_u99"])
            faulted.loc[locs, "wind_gust"] = np.maximum(original_gust, u99 + magnitude)
            faulted.loc[locs, "wind_mean"] = original_mean
        else:
            raise ValueError(f"severity sweep not defined for {spec.key}")

        faulted.loc[locs, "_injected"] = True
        for loc in locs:
            label_rows.append(
                {
                    "fault_type": spec.key,
                    "event_id": str(event["event_id"]),
                    "station_id": station_id,
                    "_row_id": int(faulted.loc[loc, "_row_id"]),
                    "timestamp_utc": faulted.loc[loc, "timestamp_utc"],
                    "is_injected": True,
                    "start_time": event["start_time"],
                }
            )

        cstart = max(0, start - context)
        cstop = min(len(idx), stop + context)
        for loc in idx[cstart:cstop]:
            if bool(faulted.loc[loc, "_injected"]):
                continue
            context_rows.append(
                {
                    "fault_type": spec.key,
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


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    paper_dir = repo_root
    parser = argparse.ArgumentParser(description="Fault severity sweep (anti-circularity).")
    parser.add_argument(
        "--results-dir", type=Path, default=paper_dir / "runs_additional" / "BASELINE" / "results_v3"
    )
    parser.add_argument("--raw-parquet", type=Path, default=repo_root / "avamet_all.parquet")
    parser.add_argument("--article-dir", type=Path, default=paper_dir / "article")
    args = parser.parse_args()

    out_dir = args.results_dir / "fault_severity_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SUITE.SEED)
    p0 = SUITE.load_p0(args.results_dir)
    p0 = p0[p0["timestamp_utc"] < SUITE.TEST_END].copy()
    station_ids = sorted(p0["station_id"].unique().tolist())
    coords = SUITE.load_coordinates(args.raw_parquet, station_ids)
    neighbors = SUITE.nearest_neighbors(coords)
    pressure_medians, pressure_residual_threshold = SUITE.pressure_residual_thresholds(p0, neighbors)

    train = p0[p0["timestamp_utc"] < SUITE.TRAIN_END].copy()
    thresholds = pd.DataFrame(
        {
            "wind_u90": train.groupby("station_id")["wind_gust"].quantile(0.90),
            "wind_u99": train.groupby("station_id")["wind_gust"].quantile(0.99),
            "wind_delta_q995": (train["wind_gust"] - train["wind_mean"]).groupby(train["station_id"]).quantile(0.995),
        }
    )
    thresholds["wind_delta_q995"] = thresholds["wind_delta_q995"].clip(lower=30.0)

    test_dirty = p0[(p0["timestamp_utc"] >= SUITE.TRAIN_END) & (p0["stratum"].eq("dirty"))].copy()
    test_dirty = test_dirty.sort_values(["station_id", "timestamp_utc"]).reset_index(drop=True)

    sweeps = {
        "pressure_offset": OFFSET_HPA,
        "pressure_drift": DRIFT_FINAL_HPA,
        "wind_spike_burst": SPIKE_ABOVE_U99,
    }
    specs = {spec.key: spec for spec in SUITE.FAULTS}

    rows: list[dict[str, object]] = []
    for key, grid in sweeps.items():
        spec = specs[key]
        # Events (windows) are chosen ONCE per fault so the only thing that varies
        # across the curve is the injected magnitude, not the locations.
        events = SUITE.choose_events(test_dirty, spec, np.random.default_rng(SUITE.SEED))
        if events.empty:
            raise RuntimeError(f"No events selected for {key}")
        for magnitude in grid:
            faulted, labels = apply_fault_with_magnitude(test_dirty, events, spec, thresholds, magnitude)
            fail, suspect = SUITE.rule_flags(faulted)
            pressure_spatial, wind_spatial = SUITE.spatial_flags(
                faulted, thresholds, pressure_medians, pressure_residual_threshold, neighbors
            )
            spatial = pressure_spatial if spec.spatial_family == "pressure" else wind_spatial
            flags = fail | suspect | spatial
            by_row = pd.Series(flags.to_numpy(dtype=bool), index=faulted["_row_id"].astype(int).to_numpy())
            m = SUITE.metrics_for_detector(labels, by_row, events, "Rule + spatial evidence")
            rows.append(
                {
                    "fault_type": key,
                    "magnitude": float(magnitude),
                    "is_reference": bool(abs(magnitude - REFERENCE[key]) < 1e-9),
                    "recall": m["recall"],
                    "recall_ci95_low": m["recall_ci95_low"],
                    "recall_ci95_high": m["recall_ci95_high"],
                    "precision": m["precision"],
                    "recall_n": m["recall_n"],
                    "median_latency_min": m["median_latency_min"],
                    "pressure_residual_threshold_hpa": float(pressure_residual_threshold),
                }
            )
            print(f"{key:18s} mag={magnitude:6.1f}  recall={m['recall']:.3f}  prec={m['precision']:.3f}")

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "fault_severity_sweep.csv", index=False)
    print(f"\nPressure spatial residual threshold (TRAIN-fixed): {pressure_residual_threshold:.3f} hPa")
    print(f"Wrote {out_dir / 'fault_severity_sweep.csv'}")

    # Figure: recall vs magnitude, one panel per fault family.
    labels_map = {
        "pressure_offset": ("Pressure offset", "Injected offset (hPa)"),
        "pressure_drift": ("Pressure drift", "Final injected drift (hPa)"),
        "wind_spike_burst": ("Wind spike/burst", "Spike height above TRAIN $u99$ (gust units)"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4))
    for ax, key in zip(axes, sweeps.keys(), strict=False):
        sub = summary[summary["fault_type"].eq(key)].sort_values("magnitude")
        x = sub["magnitude"].to_numpy()
        y = sub["recall"].to_numpy()
        lo = sub["recall_ci95_low"].to_numpy()
        hi = sub["recall_ci95_high"].to_numpy()
        ax.fill_between(x, lo, hi, color="#009e73", alpha=0.18)
        ax.plot(x, y, "-o", color="#009e73", lw=1.6, ms=4)
        ref = sub[sub["is_reference"]]
        if not ref.empty:
            ax.plot(
                ref["magnitude"], ref["recall"], "*", color="#d55e00", ms=14,
                label="Main-suite operating point",
            )
        if key == "pressure_offset":
            thr = float(sub["pressure_residual_threshold_hpa"].iloc[0])
            ax.axvline(thr, color="0.4", ls="--", lw=1.0)
            ax.text(thr, 0.04, f"  TRAIN-fixed\n  threshold {thr:.1f} hPa", fontsize=7, color="0.3")
        title, xlabel = labels_map[key]
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylim(-0.03, 1.05)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        if key == "pressure_offset":
            ax.set_ylabel("Rule+spatial evidence recall")
        ax.legend(loc="lower right", fontsize=7, frameon=False)
    fig.tight_layout()
    pdf = out_dir / "Supplementary_Figure_S2_fault_severity_sweep.pdf"
    png = out_dir / "Supplementary_Figure_S2_fault_severity_sweep.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=220)
    plt.close(fig)

    article_dir: Path = args.article_dir
    (article_dir / "Supplementary_Figure_S2_fault_severity_sweep.pdf").write_bytes(pdf.read_bytes())
    (article_dir / "Supplementary_Figure_S2_fault_severity_sweep.png").write_bytes(png.read_bytes())
    print(f"Wrote {article_dir / 'Supplementary_Figure_S2_fault_severity_sweep.pdf'}")


if __name__ == "__main__":
    main()
