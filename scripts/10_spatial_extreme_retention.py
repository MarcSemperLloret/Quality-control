from __future__ import annotations

import argparse
import math
from pathlib import Path

import duckdb
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

matplotlib.use("Agg")


TRAIN_START = pd.Timestamp("2023-01-01 00:00:00", tz="UTC")
TRAIN_END = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
TEST_START = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
TEST_END = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")

PRODUCT_LABELS = {
    "P0": "Raw",
    "P1": "Delete-all-flags",
    "P3": "Constraint-selected P3",
    "P4": "Fixed domain-rule QC",
}

COLORS = {
    "Raw": "#4d4d4d",
    "Delete-all-flags": "#d55e00",
    "Constraint-selected P3": "#009e73",
    "Fixed domain-rule QC": "#0072b2",
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    paper_dir = repo_root
    default_results = paper_dir / "runs_additional" / "BASELINE" / "results_v3"
    parser = argparse.ArgumentParser(
        description=(
            "Spatial-consistency diagnostic for raw high-state wind-gust exceedances. "
            "The diagnostic checks whether retained extremes are supported by nearby stations."
        )
    )
    parser.add_argument("--results-dir", type=Path, default=default_results)
    parser.add_argument("--raw-parquet", type=Path, default=repo_root / "avamet_all.parquet")
    parser.add_argument("--article-dir", type=Path, default=paper_dir / "article")
    parser.add_argument("--k-neighbors", type=int, default=5)
    parser.add_argument("--neighbor-quantile", type=float, default=0.90)
    parser.add_argument("--support-min-neighbors", type=int, default=2)
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


def load_product(results_dir: Path, product: str) -> pd.DataFrame:
    path = results_dir / "products" / product / "data.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(
        path,
        columns=["station_id", "timestamp_utc", "stratum", "wind_gust"],
        engine="pyarrow",
    )


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
    missing = sorted(set(station_ids) - set(coords["station_id"]))
    if missing:
        raise RuntimeError(f"Missing coordinates for stations: {missing}")
    if coords[["lat", "lon"]].isna().any().any():
        raise RuntimeError("Missing latitude/longitude in coordinate table.")
    return coords


def nearest_neighbors(coords: pd.DataFrame, k: int) -> dict[str, list[str]]:
    rows = coords.set_index("station_id")
    out: dict[str, list[str]] = {}
    for sid, row in rows.iterrows():
        distances: list[tuple[float, str]] = []
        for other, other_row in rows.iterrows():
            if other == sid:
                continue
            distances.append(
                (
                    haversine_km(float(row["lat"]), float(row["lon"]), float(other_row["lat"]), float(other_row["lon"])),
                    str(other),
                )
            )
        distances.sort(key=lambda x: x[0])
        out[str(sid)] = [sid2 for _, sid2 in distances[:k]]
    return out


def add_spatial_support(
    events: pd.DataFrame,
    test_wide: pd.DataFrame,
    neighbor_high: pd.DataFrame,
    neighbors: dict[str, list[str]],
    support_min_neighbors: int,
) -> pd.DataFrame:
    events = events.copy()
    events["neighbor_valid_n"] = 0
    events["neighbor_support_n"] = 0
    for station_id, idx in events.groupby("station_id").groups.items():
        neighbor_ids = [n for n in neighbors[str(station_id)] if n in test_wide.columns]
        ts = events.loc[idx, "timestamp_utc"]
        valid = test_wide.reindex(index=ts, columns=neighbor_ids).notna().sum(axis=1).to_numpy(dtype=int)
        support = neighbor_high.reindex(index=ts, columns=neighbor_ids).fillna(False).sum(axis=1).to_numpy(dtype=int)
        events.loc[idx, "neighbor_valid_n"] = valid
        events.loc[idx, "neighbor_support_n"] = support

    events["spatial_class"] = np.where(
        events["neighbor_support_n"] >= support_min_neighbors,
        "spatially supported",
        "weak/isolated support",
    )
    return events


def summarize_retention(
    p0: pd.DataFrame,
    products: dict[str, pd.DataFrame],
    thresholds: pd.DataFrame,
    test_wide: pd.DataFrame,
    neighbor_high: pd.DataFrame,
    neighbors: dict[str, list[str]],
    support_min_neighbors: int,
    metrics_missingness: pd.DataFrame,
    metrics_mnar: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_mask = (p0["timestamp_utc"] >= TEST_START) & (p0["timestamp_utc"] < TEST_END)
    dirty_test = p0.loc[test_mask & (p0["stratum"] == "dirty"), ["station_id", "timestamp_utc", "wind_gust"]].copy()
    dirty_test["_row_id"] = dirty_test.index.to_numpy()

    rows: list[dict[str, object]] = []
    event_rows: list[pd.DataFrame] = []

    for q_label, q_col in [("u95", "u95"), ("u99", "u99")]:
        station_threshold = thresholds[q_col].to_dict()
        threshold_vals = dirty_test["station_id"].map(station_threshold)
        events = dirty_test[dirty_test["wind_gust"] >= threshold_vals].copy()
        events["threshold"] = q_label
        events = add_spatial_support(events, test_wide, neighbor_high, neighbors, support_min_neighbors)
        event_rows.append(events)

        for product, product_df in products.items():
            retained = product_df.loc[events["_row_id"], "wind_gust"].notna().to_numpy()
            tmp = events[["threshold", "spatial_class"]].copy()
            tmp["retained"] = retained
            for spatial_class, part in tmp.groupby("spatial_class"):
                miss_row = metrics_missingness.query(
                    "product == @product and phase == 'test' and scope == 'dirty' and variable == 'wind_gust'"
                )
                mnar_row = metrics_mnar.query("product == @product and phase == 'test' and scope == 'dirty'")
                n_events = int(len(part))
                n_retained = int(part["retained"].sum())
                ci_low, ci_high = wilson_ci(n_retained, n_events)
                rows.append(
                    {
                        "threshold": q_label,
                        "spatial_class": spatial_class,
                        "product": product,
                        "policy": PRODUCT_LABELS[product],
                        "n_events": n_events,
                        "retained_n": n_retained,
                        "removed_n": int(n_events - n_retained),
                        "retention": float(n_retained / n_events) if n_events else np.nan,
                        "retention_ci95_low": ci_low,
                        "retention_ci95_high": ci_high,
                        "coverage_test_dirty": (
                            1.0 - float(miss_row["missing_additional"].iloc[0]) if not miss_row.empty else np.nan
                        ),
                        "mnar_test_dirty": float(mnar_row["state_wind_spread"].iloc[0]) if not mnar_row.empty else np.nan,
                    }
                )

    all_events = pd.concat(event_rows, ignore_index=True)
    return pd.DataFrame(rows), all_events


def write_tex_table(summary: pd.DataFrame, output_tex: Path) -> None:
    keep = summary[
        (summary["threshold"] == "u99")
        & (summary["policy"].isin(["Delete-all-flags", "Constraint-selected P3", "Fixed domain-rule QC"]))
    ].copy()
    keep["Retention"] = (100.0 * keep["retention"]).map(lambda x: f"{x:.2f}")
    keep["Retention CI"] = keep.apply(
        lambda row: f"{100.0 * row['retention']:.2f} [{100.0 * row['retention_ci95_low']:.2f}, {100.0 * row['retention_ci95_high']:.2f}]",
        axis=1,
    )
    keep["Coverage"] = (100.0 * keep["coverage_test_dirty"]).map(lambda x: f"{x:.2f}")
    keep["MNAR"] = keep["mnar_test_dirty"].map(lambda x: f"{x:.4f}")
    keep = keep.rename(
        columns={
            "policy": "Policy",
            "spatial_class": "Spatial class",
            "n_events": "Raw events",
        }
    )
    keep = keep[["Policy", "Spatial class", "Raw events", "Retention CI", "Coverage", "MNAR"]]
    order = {
        "Constraint-selected P3": 0,
        "Delete-all-flags": 1,
        "Fixed domain-rule QC": 2,
        "spatially supported": 0,
        "weak/isolated support": 1,
    }
    keep = keep.sort_values(
        by=["Policy", "Spatial class"],
        key=lambda s: s.map(order).fillna(99),
    )

    lines = [
        "\\begin{tabular}{lllrrr}",
        "\\toprule",
        "Policy & Spatial class & Raw events & Retention \\% [95\\% CI] & Coverage (\\%) & State-dep. spread \\\\",
        "\\midrule",
    ]
    for _, row in keep.iterrows():
        lines.append(
            f"{row['Policy']} & {row['Spatial class']} & {int(row['Raw events'])} & "
            f"{row['Retention CI']} & {row['Coverage']} & {row['MNAR']} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    output_tex.write_text("\n".join(lines), encoding="utf-8")


def make_figure(summary: pd.DataFrame, output_pdf: Path, output_png: Path) -> None:
    plot = summary[
        (summary["threshold"] == "u99")
        & (summary["policy"].isin(["Delete-all-flags", "Constraint-selected P3", "Fixed domain-rule QC"]))
    ].copy()
    class_order = ["spatially supported", "weak/isolated support"]
    policy_order = ["Constraint-selected P3", "Delete-all-flags", "Fixed domain-rule QC"]

    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    x = np.arange(len(class_order))
    width = 0.23
    for i, policy in enumerate(policy_order):
        sub = (
            plot[plot["policy"] == policy]
            .set_index("spatial_class")
            .reindex(class_order)
            .reset_index()
        )
        values = 100.0 * sub["retention"].to_numpy(dtype=float)
        bars = ax.bar(
            x + (i - 1) * width,
            values,
            width=width,
            label=policy,
            color=COLORS[policy],
            edgecolor="black",
            linewidth=0.5,
        )
        for bar, value in zip(bars, values, strict=False):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                min(100.0, value + 1.2),
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_ylim(0, 108)
    ax.set_ylabel("Retained raw u99 exceedances (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(["Spatially supported", "Weak/isolated support"])
    ax.set_xlabel("Support from five nearest stations")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=3, frameon=False, fontsize=8)
    fig.tight_layout()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf)
    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_dir: Path = args.results_dir
    article_dir: Path = args.article_dir
    out_dir = results_dir / "spatial_consistency"
    out_dir.mkdir(parents=True, exist_ok=True)

    p0 = load_product(results_dir, "P0")
    products = {product: load_product(results_dir, product) for product in PRODUCT_LABELS}
    for product, product_df in products.items():
        if not p0[["station_id", "timestamp_utc"]].equals(product_df[["station_id", "timestamp_utc"]]):
            raise RuntimeError(f"Product {product} is not row-aligned with P0.")

    station_ids = sorted(p0["station_id"].unique().tolist())
    coords = load_coordinates(args.raw_parquet, station_ids)
    coords.to_csv(out_dir / "spatial_station_coordinates.csv", index=False)
    neighbors = nearest_neighbors(coords, int(args.k_neighbors))
    pd.DataFrame(
        [
            {"station_id": sid, "neighbor_rank": i + 1, "neighbor_id": nid}
            for sid, nids in neighbors.items()
            for i, nid in enumerate(nids)
        ]
    ).to_csv(out_dir / "spatial_nearest_neighbors.csv", index=False)

    train_mask = (p0["timestamp_utc"] >= TRAIN_START) & (p0["timestamp_utc"] < TRAIN_END)
    thresholds = (
        p0.loc[train_mask]
        .groupby("station_id")["wind_gust"]
        .quantile([args.neighbor_quantile, 0.95, 0.99])
        .unstack()
        .rename(columns={args.neighbor_quantile: "u_neighbor", 0.95: "u95", 0.99: "u99"})
    )

    test_mask = (p0["timestamp_utc"] >= TEST_START) & (p0["timestamp_utc"] < TEST_END)
    test_wide = p0.loc[test_mask].pivot(index="timestamp_utc", columns="station_id", values="wind_gust")
    neighbor_high = test_wide.ge(thresholds["u_neighbor"], axis="columns")

    metrics_missingness = pd.read_csv(results_dir / "metrics_missingness.csv")
    metrics_mnar = pd.read_csv(results_dir / "metrics_mnar.csv")
    summary, events = summarize_retention(
        p0=p0,
        products=products,
        thresholds=thresholds,
        test_wide=test_wide,
        neighbor_high=neighbor_high,
        neighbors=neighbors,
        support_min_neighbors=int(args.support_min_neighbors),
        metrics_missingness=metrics_missingness,
        metrics_mnar=metrics_mnar,
    )

    summary.to_csv(out_dir / "spatial_extreme_retention.csv", index=False)
    events.to_csv(out_dir / "spatial_extreme_events.csv", index=False)
    write_tex_table(summary, out_dir / "tab_spatial_extreme_retention.tex")

    fig_pdf = out_dir / "F7_spatial_extreme_retention.pdf"
    fig_png = out_dir / "F7_spatial_extreme_retention.png"
    make_figure(summary, fig_pdf, fig_png)

    article_dir.mkdir(parents=True, exist_ok=True)
    (article_dir / "F7_spatial_extreme_retention.pdf").write_bytes(fig_pdf.read_bytes())
    (article_dir / "tab_spatial_extreme_retention.tex").write_text(
        (out_dir / "tab_spatial_extreme_retention.tex").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    print(f"Wrote {out_dir / 'spatial_extreme_retention.csv'}")
    print(f"Wrote {article_dir / 'F7_spatial_extreme_retention.pdf'}")
    print(f"Wrote {article_dir / 'tab_spatial_extreme_retention.tex'}")


if __name__ == "__main__":
    main()
