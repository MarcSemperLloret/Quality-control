from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

matplotlib.use("Agg")


COLORS = {
    "raw": "#475569",
    "aggressive": "#2563eb",
    "certified": "#059669",
    "domain": "#dc2626",
    "feasible": "#22c55e",
    "accent": "#f59e0b",
    "muted": "#94a3b8",
}


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(f"ABORT: {msg}")


def _load_csv(path: Path) -> pd.DataFrame:
    _require(path.exists(), f"Missing required file: {path}")
    return pd.read_csv(path)


def _optional_csv(root: Path, rels: Iterable[str]) -> pd.DataFrame:
    for rel in rels:
        path = root / rel
        if path.exists():
            return pd.read_csv(path)
    return pd.DataFrame()


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "figure.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#e2e8f0",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.9,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _save(fig: plt.Figure, fig_dir: Path, stem: str) -> None:
    fig.savefig(fig_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _phase_scope(df: pd.DataFrame, phase: str, scope: str) -> pd.DataFrame:
    return df[(df["phase"] == phase) & (df["scope"] == scope)].copy()


def _collapse_product_rows(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    rows = []
    groups = [
        ("P1", "P1", ["P1"]),
        ("P2/P3", "P2/P3", ["P2", "P3"]),
        ("P4/P5-DQC", "P4/P5-DQC", ["P4", "P5"]),
    ]
    for key, label, products in groups:
        sub = df[df["product"].isin(products)]
        if sub.empty:
            continue
        row = {"key": key, "label": label}
        for col in value_cols:
            row[col] = float(sub[col].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def make_f1_benchmark(fig_dir: Path, bench: pd.DataFrame, selected_cfg: str) -> None:
    b = bench[bench["policy_suspect"] == "KEEP"].copy()
    if b.empty:
        b = bench.copy()
    _require(not b.empty, "No benchmark rows available for F1.")

    recall = b.pivot_table(index="T_hours", columns="tol_hpa", values="recall_stuck", aggfunc="mean").sort_index().sort_index(axis=1)
    precision = b.pivot_table(index="T_hours", columns="tol_hpa", values="precision_stuck", aggfunc="mean").reindex_like(recall)
    feasible = (recall >= 0.95) & (precision >= 0.90)

    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    im = ax.imshow(recall.to_numpy(dtype=float), vmin=0.0, vmax=1.0, cmap="YlGnBu", aspect="auto")
    ax.grid(False)
    ax.set_xticks(np.arange(len(recall.columns)), [f"{c:.2f}" for c in recall.columns])
    ax.set_yticks(np.arange(len(recall.index)), [str(int(v)) for v in recall.index])
    ax.set_xlabel("Tolerance (hPa)")
    ax.set_ylabel("Window length (hours)")
    ax.set_title("C6 stuck benchmark: recall and feasible region")

    for i in range(recall.shape[0]):
        for j in range(recall.shape[1]):
            val = float(recall.iloc[i, j])
            color = "white" if val >= 0.60 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9)
            if bool(feasible.iloc[i, j]):
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=COLORS["feasible"], linewidth=3.5))

    sel = b[b["config_key"] == selected_cfg]
    if not sel.empty:
        t = int(sel.iloc[0]["T_hours"])
        tol = float(sel.iloc[0]["tol_hpa"])
        if t in recall.index and tol in recall.columns:
            i = recall.index.tolist().index(t)
            j = recall.columns.tolist().index(tol)
            ax.scatter([j + 0.28], [i - 0.28], marker="*", s=230, color="#fde047", edgecolor="black", linewidth=0.9, zorder=5)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Recall")
    fig.subplots_adjust(bottom=0.10)
    _save(fig, fig_dir, "F1_benchmark_c6_region")


def make_f2_downstream(fig_dir: Path, cost: pd.DataFrame) -> None:
    ridge = cost[(cost["phase"] == "test") & (cost["scope"] == "global") & (cost["model"] == "ridge")].copy()
    _require(not ridge.empty, "No ridge downstream rows available for F2.")

    groups = [
        ("P0", "P0", ["P0"], COLORS["raw"], "o", 95),
        ("P1/P4/P5-DQC", "P1 = P4/P5-DQC", ["P1", "P4", "P5"], COLORS["aggressive"], "o", 125),
        ("P2/P3", "P2/P3", ["P2", "P3"], COLORS["certified"], "o", 125),
    ]

    rows = []
    for key, label, products, color, marker, size in groups:
        sub = ridge[ridge["product"].isin(products)]
        if sub.empty:
            continue
        rows.append(
            {
                "key": key,
                "label": label,
                "coverage": float(sub["coverage_operational"].mean()),
                "score": float(sub["score_cost_sensitive"].mean()),
                "color": color,
                "marker": marker,
                "size": size,
            }
        )
    plot = pd.DataFrame(rows)
    _require(not plot.empty, "No collapsed downstream rows available for F2.")

    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    for _, r in plot.iterrows():
        if r["key"] == "P1/P4/P5-DQC":
            ax.scatter(r["coverage"], r["score"], s=float(r["size"]) + 90, facecolors="none", edgecolors=COLORS["domain"], linewidth=2.0, zorder=3)
        ax.scatter(r["coverage"], r["score"], s=float(r["size"]), color=r["color"], edgecolor="black", linewidth=0.7, zorder=4)

    offsets = {
        "P0": (8, 6, "left"),
        "P1/P4/P5-DQC": (10, -26, "left"),
        "P2/P3": (8, 12, "left"),
    }
    for _, r in plot.iterrows():
        dx, dy, ha = offsets.get(str(r["key"]), (6, 6, "left"))
        ax.annotate(
            str(r["label"]),
            (r["coverage"], r["score"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=9,
        )

    ax.set_xlabel("Coverage (TEST, global)")
    ax.set_ylabel("Cost-sensitive ridge score")
    ax.set_title("Downstream trade-off")
    ax.set_xlim(min(plot["coverage"]) - 0.006, 1.003)
    y_margin = 0.08 * (float(plot["score"].max()) - float(plot["score"].min()))
    ax.set_ylim(float(plot["score"].min()) - y_margin, float(plot["score"].max()) + 2.5 * y_margin)
    _save(fig, fig_dir, "F2_downstream_tradeoff")


def make_f3_deciles(fig_dir: Path, remdec: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), sharey=False)
    phases = [("train", "TRAIN"), ("test", "TEST")]
    styles = {
        "P1/P4/P5-DQC": dict(color=COLORS["domain"], linestyle="-", marker="o", label="Aggressive/domain-rule"),
        "P2/P3": dict(color=COLORS["certified"], linestyle="-", marker="o", label="Constraint-certified"),
    }

    for ax, (phase, title) in zip(axes, phases, strict=False):
        sub = _phase_scope(remdec, phase, "dirty")
        _require(not sub.empty, f"No removal-by-decile rows for {phase}/dirty.")

        collapsed = []
        for key, products in [("P1/P4/P5-DQC", ["P1", "P4", "P5"]), ("P2/P3", ["P2", "P3"])]:
            g = sub[sub["product"].isin(products)].groupby("decile", as_index=False)["removal_rate"].mean()
            g["key"] = key
            collapsed.append(g)
        plot = pd.concat(collapsed, ignore_index=True)

        for key in ["P1/P4/P5-DQC", "P2/P3"]:
            g = plot[plot["key"] == key].sort_values("decile")
            x = g["decile"].to_numpy(dtype=float)
            ax.plot(x, 100.0 * g["removal_rate"], linewidth=2.0, markersize=4.8, **styles[key])

        for key, label, y_pad in [
            ("P1/P4/P5-DQC", "Aggressive/domain\n(P1/P4/P5-DQC)", 0.0),
            ("P2/P3", "Constraint-certified\n(P2/P3)", 1.2),
        ]:
            g = plot[plot["key"] == key].sort_values("decile")
            y = float(100.0 * g["removal_rate"].iloc[-1]) + y_pad
            ax.text(10.25, y, label, color=styles[key]["color"], fontsize=8.7, va="center", ha="left", clip_on=False)

        ax.set_title(title)
        ax.set_xlabel("Wind-gust decile in P0")
        ax.set_ylabel("Removal rate (%)")
        ax.set_xticks(np.arange(1, 11))
        ax.set_xlim(0.8, 13.4)
        ax.set_ylim(bottom=0.0)
        if phase == "test":
            ax.set_ylim(0.0, max(16.5, ax.get_ylim()[1]))

    fig.suptitle("Removal by wind-gust decile (dirty stratum)", y=1.02)
    _save(fig, fig_dir, "F3_removal_by_decile")


def make_f4_removed_kept(fig_dir: Path, rvk: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    phases = [("train", "TRAIN dirty"), ("test", "TEST dirty")]
    products = ["P1", "P2/P3", "P4/P5-DQC"]
    x = np.arange(len(products))

    for ax, (phase, title) in zip(axes, phases, strict=False):
        sub = _phase_scope(rvk, phase, "dirty")
        _require(not sub.empty, f"No removed-vs-kept rows for {phase}/dirty.")
        plot = _collapse_product_rows(sub, ["p99_removed", "p99_kept", "removed_rate"])
        plot = plot.set_index("key").reindex(products).reset_index()

        b1 = ax.bar(x - 0.18, plot["p99_removed"], width=0.36, color=COLORS["muted"], edgecolor="black", linewidth=0.5, label="p99 removed")
        b2 = ax.bar(x + 0.18, plot["p99_kept"], width=0.36, color="#0ea5e9", edgecolor="black", linewidth=0.5, label="p99 kept")
        tick_labels = [f"{p}\nremoved {100.0 * float(rr):.2f}%" for p, rr in zip(products, plot["removed_rate"], strict=False)]
        ax.set_xticks(x, tick_labels)
        ax.set_title(title)
        ax.set_ylabel("wind_gust")
        ax.set_ylim(0, 70)

        for bars in [b1, b2]:
            for bar in bars:
                h = float(bar.get_height())
                if np.isnan(h):
                    continue
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8, f"{h:.1f}", ha="center", va="bottom", fontsize=8)

    fig.legend(handles=[b1[0], b2[0]], labels=["p99 removed", "p99 kept"], loc="lower center", ncol=2, frameon=True)
    fig.suptitle("Selection evidence in tails: removed vs kept p99", y=1.02)
    fig.subplots_adjust(bottom=0.18)
    _save(fig, fig_dir, "F4_removed_vs_kept_p99")


def make_f5_robustness(fig_dir: Path, ablation: pd.DataFrame, budget: pd.DataFrame) -> None:
    _require(not ablation.empty, "Missing summary_ablation_c6.csv for F5.")
    _require(not budget.empty, "Missing summary_budget_sweep.csv for F5.")

    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    labels = ["C6 ON", "C6 OFF"]
    run_map = {"BASELINE": "C6 ON", "ABL_C6_OFF": "C6 OFF"}
    a = ablation.copy()
    a["label"] = a["run_id"].map(run_map).fillna(a["run_id"])
    a = a.set_index("label").reindex(labels).reset_index()
    bars = ax.bar(labels, a["benchmark_recall"], color=[COLORS["certified"], COLORS["domain"]], edgecolor="black", linewidth=0.6)
    ax.axhline(0.95, color="#334155", linestyle="--", linewidth=1.0)
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Recall on stuck benchmark")
    ax.set_title("C6 ablation")
    for bar, prec in zip(bars, a["benchmark_precision"], strict=False):
        h = float(bar.get_height())
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.025, f"{h:.2f}", ha="center", va="bottom", fontsize=9)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            0.08,
            f"Precision = {float(prec):.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#334155",
        )
    _save(fig, fig_dir, "F5_robustness_checks")

    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    b = budget.copy()
    ax.scatter(b["coverage_test"], b["downstream_ridge_score"], s=100, color=COLORS["certified"], edgecolor="black", linewidth=0.7)
    x = float(b["coverage_test"].mean())
    y = float(b["downstream_ridge_score"].mean())
    ax.scatter([x], [y], s=240, facecolors="none", edgecolors=COLORS["accent"], linewidth=2.0)
    budgets = ", ".join(f"{100.0 * float(v):g}%" for v in b["budget"].tolist())
    ax.annotate(
        f"all budgets overlap\n{budgets}",
        (x, y),
        xytext=(-30, 34),
        textcoords="offset points",
        ha="right",
        va="bottom",
        arrowprops=dict(arrowstyle="-", color="#64748b", lw=0.8),
        fontsize=8.5,
    )
    ax.set_title("Budget sensitivity")
    ax.set_xlabel("Coverage (TEST, global)")
    ax.set_ylabel("Cost-sensitive ridge score")
    ax.set_xlim(x - 0.004, x + 0.004)
    ax.set_ylim(y - 0.03, y + 0.045)
    _save(fig, fig_dir, "S_budget_sensitivity")


def _load_budget_best_for_redraw(results_dir: Path) -> dict:
    for name in ["budget_best_B100_v3.json", "budget_best_B10_v3.json"]:
        path = results_dir / name
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    candidates = sorted(results_dir.glob("budget_best_B*_v3.json"))
    _require(len(candidates) > 0, "Missing budget_best_B*_v3.json file for redraw.")
    return json.loads(candidates[-1].read_text(encoding="utf-8"))


def main() -> None:
    _set_style()
    root = Path(".").resolve()
    r = root / "results_v3"
    fig_dir = r / "figures"
    _require(r.exists(), f"Missing results folder: {r}")
    fig_dir.mkdir(parents=True, exist_ok=True)

    bench = _load_csv(r / "benchmark_stuck" / "benchmark_perf_by_config.csv")
    cost = _load_csv(r / "downstream" / "downstream_cost_sensitive_v3.csv")
    remdec = _load_csv(r / "metrics_removal_by_decile.csv")
    rvk = _load_csv(r / "metrics_removed_vs_kept.csv")

    ablation = _optional_csv(
        root,
        [
            "runs_additional/summary_ablation_c6.csv",
            "../paper 2/runs_additional/summary_ablation_c6.csv",
        ],
    )
    budget = _optional_csv(
        root,
        [
            "runs_additional/summary_budget_sweep.csv",
            "../paper 2/runs_additional/summary_budget_sweep.csv",
        ],
    )

    best = _load_budget_best_for_redraw(r)
    sel_bench = json.loads((r / "benchmark_stuck" / "benchmark_perf_selected_config.json").read_text(encoding="utf-8"))
    selected_cfg = str(sel_bench.get("config_key", best.get("config_key", "")))

    make_f1_benchmark(fig_dir, bench, selected_cfg=selected_cfg)
    make_f2_downstream(fig_dir, cost)
    make_f3_deciles(fig_dir, remdec)
    make_f4_removed_kept(fig_dir, rvk)
    make_f5_robustness(fig_dir, ablation, budget)

    print("Redesigned main figures regenerated in results_v3/figures")


if __name__ == "__main__":
    main()
