from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

matplotlib.use("Agg")

POLICY_ORDER = ["P1", "P2", "P3", "P4"]
POLICY_COLORS = {
    "P1": "#3b82f6",  # azul
    "P2": "#10b981",  # verde
    "P3": "#f59e0b",  # naranja
    "P4": "#ef4444",  # rojo
}
PLOT_ORDER = ["P1", "PG", "P4"]
PLOT_LABELS = {
    "P1": "P1",
    "PG": "Guaranteed",
    "P4": "P4",
}
PLOT_COLORS = {
    "P1": POLICY_COLORS["P1"],
    "PG": POLICY_COLORS["P2"],
    "P4": POLICY_COLORS["P4"],
}
WIND_UNIT_LABEL = "wind_gust (unidad nativa AVAMET)"


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(f"ABORT: {msg}")


def _load_csv(path: Path) -> pd.DataFrame:
    _require(path.exists(), f"Missing required file: {path}")
    return pd.read_csv(path)


def _rule_label(rule: str) -> str:
    mapping = {
        "suspect_stuck": "stuck",
        "fail_wind_incoherent": "wind_incoh",
        "flag_unit_converted": "unit_conv",
        "pressure_fail_zero": "pres_zero",
        "pressure_fail_out_range_after_norm": "pres_range",
        "fail_wind_negative": "wind_neg",
    }
    return mapping.get(rule, rule)


def _plot_product_key(product: str) -> str:
    return "PG" if product in {"P2", "P3"} else product


def _metric_from_indexed(df_idx: pd.DataFrame, col: str, key: str) -> float:
    if key == "PG":
        vals = df_idx.loc[df_idx.index.intersection(["P2", "P3"]), col]
        return float(vals.mean()) if len(vals) > 0 else np.nan
    if key in df_idx.index:
        return float(df_idx.loc[key, col])
    return np.nan


def _series_from_df(df: pd.DataFrame, xcol: str, ycol: str, key: str) -> pd.DataFrame:
    if key == "PG":
        out = df[df["product"].isin(["P2", "P3"])].groupby(xcol, as_index=False)[ycol].mean()
    else:
        out = df[df["product"] == key][[xcol, ycol]].copy()
    return out.sort_values(xcol)


def make_f1_pareto_png(fig_dir: Path, pareto: pd.DataFrame) -> None:
    sub = pareto[(pareto["phase"] == "test") & (pareto["scope"] == "dirty") & (pareto["product"].isin(["P1", "P2", "P3", "P4"]))].copy()
    _require(not sub.empty, "No pareto data for F1.")

    product_order = PLOT_ORDER
    rows = []
    n_removed = {}
    for p in product_order:
        if p == "PG":
            g = (
                sub[sub["product"].isin(["P2", "P3"])]
                .groupby("rule", as_index=False)
                .agg(rows_n=("rows_n", "mean"), share_pct=("share_pct", "mean"))
                .sort_values("rows_n", ascending=False)
                .reset_index(drop=True)
            )
        else:
            g = sub[sub["product"] == p][["rule", "rows_n", "share_pct"]].sort_values("rows_n", ascending=False).reset_index(drop=True)
        _require(not g.empty, f"No pareto rows for {p}.")
        g["cum"] = g["share_pct"].cumsum()
        keep_idx = g.index[g["cum"] <= 99.9].tolist()
        if not keep_idx or keep_idx[-1] != len(g) - 1:
            keep_idx.append(min(len(g) - 1, max(keep_idx + [0]) + 1))
        gk = g.loc[sorted(set(keep_idx))].copy()
        gk["product"] = p
        rows.append(gk)
        n_removed[p] = int(round(float(g["rows_n"].sum())))
    slim = pd.concat(rows, ignore_index=True)
    slim["rule_short"] = slim["rule"].map(_rule_label)

    rule_order = (
        slim.groupby("rule_short", as_index=False)["share_pct"]
        .sum()
        .sort_values("share_pct", ascending=False)["rule_short"]
        .tolist()
    )
    mat = slim.pivot_table(index="product", columns="rule_short", values="share_pct", aggfunc="sum", fill_value=0.0)
    mat = mat.reindex(index=product_order, columns=rule_order, fill_value=0.0)

    plt.figure(figsize=(11, 6))
    x = np.arange(len(product_order))
    bottom = np.zeros(len(product_order), dtype=float)
    cmap = plt.get_cmap("tab20")
    for i, r in enumerate(rule_order):
        vals = mat[r].to_numpy(dtype=float)
        bars = plt.bar(x, vals, bottom=bottom, label=r, color=cmap(i % 20), width=0.7)
        for j, (b, v) in enumerate(zip(bars, vals, strict=False)):
            if v >= 3.0:
                plt.text(
                    b.get_x() + b.get_width() / 2,
                    bottom[j] + v / 2,
                    f"{v:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                )
        bottom += vals

    labels = [f"{PLOT_LABELS[p]}\n(n={n_removed[p]})" for p in product_order]
    plt.xticks(x, labels)
    plt.ylabel("Percentage of rows removed (%)")
    plt.ylim(0, 100.0)
    plt.title("Primary-rule removal Pareto (DIRTY, TEST)\nGuaranteed = P2/P3 in this rerun")
    plt.legend(loc="upper right", fontsize=9, ncol=2, title="Primary rule")
    plt.tight_layout()
    plt.savefig(fig_dir / "F1_pareto.png", dpi=220, bbox_inches="tight")
    plt.savefig(fig_dir / "F1_pareto.pdf", bbox_inches="tight")
    plt.close()


def make_f2_tails_png(fig_dir: Path, tails: pd.DataFrame) -> None:
    sub = tails[
        (tails["variable"] == "wind_gust")
        & (tails["scope"] == "dirty")
        & (tails["phase"].isin(["train", "test"]))
        & (tails["product"].isin(["P1", "P2", "P3", "P4"]))
    ].copy()
    _require(not sub.empty, "No tails data for F2.")

    product_order = PLOT_ORDER
    colors = [PLOT_COLORS[p] for p in product_order]
    metrics = [("p99", "delta_oper_p99", "delta_common_p99"), ("p999", "delta_oper_p999", "delta_common_p999")]
    eps_vis = 1e-6

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # TRAIN panels only (requested 1x2)
    train_df = sub[sub["phase"] == "train"].set_index("product")
    train_vals_by_metric = {}
    for name, op, cm in metrics:
        vals = []
        for p in product_order:
            opv = _metric_from_indexed(train_df, op, p)
            cmv = _metric_from_indexed(train_df, cm, p)
            vals.append(100.0 * float(opv - cmv) if (not np.isnan(opv) and not np.isnan(cmv)) else 0.0)
        train_vals_by_metric[name] = vals

    for j, (name, _, _) in enumerate(metrics):
        ax = axes[j]
        vals = train_vals_by_metric[name]
        x = np.arange(len(product_order))
        bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.7)
        ax.set_xticks(x, [PLOT_LABELS[p] for p in product_order])
        ax.axhline(0.0, color="black", linewidth=1)
        ylim = max(0.05, 1.10 * float(np.nanmax(np.array(vals, dtype=float))))
        ax.set_ylim(0.0, ylim)
        ax.set_title(f"TRAIN - Delta(oper-common) {name} (%)", fontsize=11)
        ax.set_ylabel("Delta (%)", fontsize=10)
        ax.tick_params(labelsize=9)
        for b, v in zip(bars, vals, strict=False):
            offset = 0.02 * ylim
            ytxt = max(0.0, v) + offset
            ax.text(
                b.get_x() + b.get_width() / 2,
                ytxt,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # Single textual TEST note (instead of empty row)
    test_df = sub[sub["phase"] == "test"].set_index("product")
    test_lines = []
    all_test_abs = []
    for name, op, cm in metrics:
        vals = []
        for p in product_order:
            if p in test_df.index:
                pass
            opv = _metric_from_indexed(test_df, op, p)
            cmv = _metric_from_indexed(test_df, cm, p)
            vals.append(abs(100.0 * float(opv - cmv)) if (not np.isnan(opv) and not np.isnan(cmv)) else 0.0)
        m = float(np.nanmax(np.array(vals, dtype=float)))
        all_test_abs.append(m)
        test_lines.append(f"{name}: max |Delta| = {m:.2e}%")
    if max(all_test_abs) < eps_vis:
        test_note = "TEST (DIRTY): max |Delta| < 1e-6 en p99 y p999"
    else:
        test_note = "TEST (DIRTY): " + " | ".join(test_lines)

    fig.suptitle("Brecha Operational vs Common (DIRTY, wind_gust)", y=0.98, fontsize=13)
    fig.text(
        0.5,
        0.035,
        "TRAIN: Guaranteed (P2/P3) = 0.00 en p99/p999 | " + test_note,
        ha="center",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0.07, 1, 0.94])
    plt.savefig(fig_dir / "F2_common_vs_operational_tails.png", dpi=220, bbox_inches="tight")
    plt.savefig(fig_dir / "F2_common_vs_operational_tails.pdf", bbox_inches="tight")
    plt.close()


def make_f3_mnar_png(fig_dir: Path, mnar: pd.DataFrame, remdec: pd.DataFrame, rvk: pd.DataFrame) -> None:
    product_order = PLOT_ORDER
    raw_products = ["P1", "P2", "P3", "P4"]
    phase_order = ["train", "test"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True)

    line_cfg = {
        "P1": {"color": POLICY_COLORS["P1"], "linestyle": "-", "label": "P1"},
        "P4": {"color": POLICY_COLORS["P1"], "linestyle": "--", "label": "P4 (overlap P1)"},
        "PG": {"color": POLICY_COLORS["P2"], "linestyle": "-", "label": "Guaranteed (P2=P3)"},
    }

    for r, ph in enumerate(phase_order):
        ms = mnar[(mnar["phase"] == ph) & (mnar["scope"] == "dirty") & (mnar["product"].isin(raw_products))].set_index("product")
        spread_vals = [_metric_from_indexed(ms, "state_wind_spread", p) for p in product_order]
        axes[r, 0].bar([PLOT_LABELS[p] for p in product_order], spread_vals, color=[PLOT_COLORS[p] for p in product_order], edgecolor="black", linewidth=0.5)
        axes[r, 0].set_title(f"{ph.upper()} spread", fontsize=11)
        axes[r, 0].set_ylabel("state_wind_spread", fontsize=10)
        axes[r, 0].tick_params(labelsize=9)

        ds = remdec[(remdec["phase"] == ph) & (remdec["scope"] == "dirty") & (remdec["product"].isin(raw_products))]
        for p in ["P1", "P4", "PG"]:
            dp = _series_from_df(ds, xcol="decile", ycol="removal_rate", key=p)
            axes[r, 1].plot(
                dp["decile"],
                100.0 * dp["removal_rate"],
                marker="o",
                linewidth=1.6,
                color=line_cfg[p]["color"],
                linestyle=line_cfg[p]["linestyle"],
                label=line_cfg[p]["label"],
            )
        axes[r, 1].set_title(f"{ph.upper()} removal by decile", fontsize=11)
        axes[r, 1].set_xlabel("Decil de wind_gust en P0", fontsize=10)
        axes[r, 1].set_ylabel("Removal rate (%)", fontsize=10)
        axes[r, 1].tick_params(labelsize=9)
        if r == 0:
            axes[r, 1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, borderaxespad=0)

        # inset zoom for Guaranteed line
        # train: intermediate height (avoid covering blue line)
        # test: upper-right but slightly lowered
        if r == 0:
            iax = inset_axes(
                axes[r, 1],
                width="35%",
                height="35%",
                loc="center right",
                bbox_to_anchor=(-0.18, -0.10, 1.0, 1.0),
                bbox_transform=axes[r, 1].transAxes,
                borderpad=0.7,
            )
        else:
            iax = inset_axes(
                axes[r, 1],
                width="35%",
                height="35%",
                loc="upper right",
                bbox_to_anchor=(0.0, -0.03, 1.0, 1.0),
                bbox_transform=axes[r, 1].transAxes,
                borderpad=0.7,
            )
        for p in ["PG"]:
            dp = _series_from_df(ds, xcol="decile", ycol="removal_rate", key=p)
            iax.plot(
                dp["decile"],
                100.0 * dp["removal_rate"],
                marker="o",
                linewidth=1.2,
                color=line_cfg[p]["color"],
                linestyle=line_cfg[p]["linestyle"],
            )
        iax.set_ylim(0.0, 1.0)
        iax.set_xticks([])
        iax.set_yticks([])
        iax.set_title("Zoom Guaranteed", fontsize=8)

        rk = rvk[(rvk["phase"] == ph) & (rvk["scope"] == "dirty") & (rvk["product"].isin(raw_products))].set_index("product")
        x = np.arange(len(product_order))
        removed = [_metric_from_indexed(rk, "p99_removed", p) for p in product_order]
        kept = [_metric_from_indexed(rk, "p99_kept", p) for p in product_order]
        removed_rate = [_metric_from_indexed(rk, "removed_rate", p) for p in product_order]

        b_removed = axes[r, 2].bar(x - 0.18, removed, width=0.36, label="p99_removed", color="#94a3b8")
        b_kept = axes[r, 2].bar(x + 0.18, kept, width=0.36, label="p99_kept", color="#0ea5e9")
        axes[r, 2].set_xticks(x, [PLOT_LABELS[p] for p in product_order])
        axes[r, 2].set_title(f"{ph.upper()} removed vs kept p99", fontsize=11)
        axes[r, 2].set_ylabel(WIND_UNIT_LABEL, fontsize=10)
        axes[r, 2].tick_params(labelsize=9)
        if r == 0:
            axes[r, 2].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, borderaxespad=0)

        valid_vals = np.array([v for v in removed + kept if not np.isnan(v)], dtype=float)
        if valid_vals.size > 0:
            ymin = float(np.nanmin(valid_vals))
            ymax = float(np.nanmax(valid_vals))
            margin = max(0.5, 0.08 * (ymax - ymin if ymax > ymin else 1.0))
            axes[r, 2].set_ylim(ymin - margin, ymax + margin)

        # bar labels
        for bars in [b_removed, b_kept]:
            for b in bars:
                h = b.get_height()
                if np.isnan(h):
                    continue
                axes[r, 2].annotate(
                    f"{h:.1f}",
                    (b.get_x() + b.get_width() / 2, h),
                    textcoords="offset points",
                    xytext=(0, 3),
                    ha="center",
                    fontsize=7,
                )

        # keep removed_rate available for metrics, but do not annotate on figure
        _ = removed_rate

    fig.suptitle("MNAR evidence (DIRTY): spread, deciles and p99 removed/kept", y=1.02, fontsize=13)
    plt.savefig(fig_dir / "F3_mnar_spread_deciles.png", dpi=220, bbox_inches="tight")
    plt.savefig(fig_dir / "F3_mnar_spread_deciles.pdf", bbox_inches="tight")
    plt.close()


def make_f4_tradeoff_png(fig_dir: Path, budget: pd.DataFrame, best_b10: dict) -> None:
    sub = budget[(budget["fold"] == "full") & (budget["phase"] == "train") & (budget["scope"] == "dirty")].copy()
    _require(not sub.empty, "No budget grid data for F4.")
    feas = sub[sub["compliant"] == True].copy()  # noqa: E712
    rej = sub[sub["compliant"] == False].copy()  # noqa: E712

    fig, (axg, axz) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.35, 1.0]})

    def _aggregate_points(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        d = df.copy()
        d["x"] = (100.0 * d["missing_additional_train"]).round(6)
        d["y"] = d["state_wind_spread_train"].round(6)
        return d.groupby(["x", "y"], as_index=False).agg(
            k=("config_key", "count"),
            suspect_rate=("suspect_rate_train", "mean"),
        )

    rejg = _aggregate_points(rej)
    feasg = _aggregate_points(feas)

    sx = np.nan
    sy = np.nan

    def _plot_tradeoff_axis(ax: plt.Axes, xlim: tuple[float, float], ylim: tuple[float, float], show_k: bool) -> plt.Axes:
        sc_local = None
        if not rejg.empty:
            ax.scatter(
                rejg["x"],
                rejg["y"],
                s=20 + 25 * rejg["k"],
                c="lightgray",
                alpha=0.7,
                label="Rejected (fails constraints/C6)",
            )
            if show_k:
                for _, rr in rejg.iterrows():
                    if (float(rr["x"]) < 2.0) and (float(rr["y"]) < 0.02):
                        continue
                    ax.annotate(
                        f"k={int(rr['k'])}",
                        (rr["x"], rr["y"]),
                        xytext=(5, -8),
                        textcoords="offset points",
                        fontsize=7,
                        color="gray",
                    )

        if not feasg.empty:
            sc_local = ax.scatter(
                feasg["x"],
                feasg["y"],
                c=feasg["suspect_rate"],
                cmap="viridis",
                s=30 + 35 * feasg["k"],
                edgecolor="black",
                linewidth=0.4,
                label="Feasible (passes all incl. C6)",
            )
            if show_k:
                for _, rr in feasg.iterrows():
                    if (float(rr["x"]) < 2.0) and (float(rr["y"]) < 0.02):
                        continue
                    ax.annotate(
                        f"k={int(rr['k'])}",
                        (rr["x"], rr["y"]),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=7,
                    )

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.axvline(5.0, linestyle="--", color="#ef4444", linewidth=1, label="B=5%")
        if xlim[1] >= 10.0:
            ax.axvline(10.0, linestyle="--", color="#f59e0b", linewidth=1, label="B=10%")
        ax.set_xlabel("Missing additional (%)")
        ax.set_ylabel("state_wind_spread")
        return sc_local

    sc_global = _plot_tradeoff_axis(axg, xlim=(-1.0, 52.0), ylim=(-0.001, 0.165), show_k=True)

    sel_key = str(best_b10.get("config_key", ""))
    sel = feas[feas["config_key"] == sel_key]
    if not sel.empty:
        sx = 100.0 * float(sel.iloc[0]["missing_additional_train"])
        sy = float(sel.iloc[0]["state_wind_spread_train"])
        axg.scatter([sx], [sy], marker="*", s=220, color="red", zorder=7, label=f"Selected: {sel_key}")
        axg.annotate(
            sel_key,
            (sx, sy),
            textcoords="offset points",
            xytext=(12, 10),
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
        )

    # zoom panel
    _plot_tradeoff_axis(axz, xlim=(-0.1, 2.0), ylim=(-0.001, 0.020), show_k=False)
    if not sel.empty:
        axz.scatter([sx], [sy], marker="*", s=220, color="red", zorder=7)
        axz.annotate(
            sel_key,
            (sx, sy),
            textcoords="offset points",
            xytext=(12, 8),
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
        )

    axg.text(0.02, 0.96, "<- mejor (menos missing)", transform=axg.transAxes, fontsize=8, va="top")
    axg.text(0.02, 0.90, "v mejor (menos spread)", transform=axg.transAxes, fontsize=8, va="top")
    axg.set_title("Global")
    axz.set_title("Zoom (region factible)")

    if sc_global is not None:
        cbar = fig.colorbar(sc_global, ax=[axg, axz], fraction=0.03, pad=0.02)
        cbar.set_label("suspect_rate_train")

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color="lightgray", markersize=8, label="Rejected (fails constraints/C6)"),
        plt.Line2D([0], [0], marker="o", linestyle="", color="#2aa198", markeredgecolor="black", markersize=8, label="Feasible (passes all incl. C6)"),
        plt.Line2D([0], [0], color="#ef4444", linestyle="--", linewidth=1, label="B=5%"),
        plt.Line2D([0], [0], color="#f59e0b", linestyle="--", linewidth=1, label="B=10%"),
        plt.Line2D([0], [0], marker="*", linestyle="", color="red", markersize=12, label=f"Selected: {sel_key}" if sel_key else "Selected"),
    ]
    fig.legend(handles=handles, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=3)
    fig.suptitle("Trade-off v4 (DIRTY, TRAIN) con filtro C6", y=0.99, fontsize=13)
    fig.subplots_adjust(bottom=0.16, top=0.84, right=0.92, wspace=0.15)
    plt.savefig(fig_dir / "F4_tradeoff.png", dpi=220, bbox_inches="tight")
    plt.savefig(fig_dir / "F4_tradeoff.pdf", bbox_inches="tight")
    plt.close()

    # Keep separate zoom export for supplements
    figz, axz = plt.subplots(1, 1, figsize=(8, 6))
    if not rejg.empty:
        axz.scatter(rejg["x"], rejg["y"], s=20 + 25 * rejg["k"], c="lightgray", alpha=0.7, label="Rejected")
    if not feasg.empty:
        scz = axz.scatter(
            feasg["x"],
            feasg["y"],
            c=feasg["suspect_rate"],
            cmap="viridis",
            s=30 + 35 * feasg["k"],
            edgecolor="black",
            linewidth=0.4,
            label="Feasible",
        )
        cbarz = figz.colorbar(scz, ax=axz, fraction=0.046, pad=0.02)
        cbarz.set_label("suspect_rate_train")
    if not sel.empty:
        axz.scatter([sx], [sy], marker="*", s=220, color="red", zorder=6, label=f"Selected: {sel_key}")
        axz.annotate(
            sel_key,
            (sx, sy),
            textcoords="offset points",
            xytext=(12, 8),
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
        )

    # Zoom bounds near feasible operating region
    x_upper = 12.0
    if not feasg.empty:
        x_upper = max(2.0, min(12.0, float(np.nanpercentile(feasg["x"], 95)) + 1.0))
        y_upper = max(0.02, float(np.nanpercentile(feasg["y"], 95)) + 0.005)
    else:
        y_upper = 0.02
    axz.set_xlim(-0.1, x_upper)
    axz.set_ylim(-0.001, y_upper)
    axz.axvline(5.0, linestyle="--", color="#ef4444", linewidth=1, label="B=5%")
    axz.axvline(10.0, linestyle="--", color="#f59e0b", linewidth=1, label="B=10%")
    axz.set_xlabel("Missing additional (%)")
    axz.set_ylabel("state_wind_spread")
    axz.set_title("Trade-off zoom v4 (DIRTY, TRAIN)")
    axz.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, borderaxespad=0)
    figz.subplots_adjust(bottom=0.18, top=0.92, right=0.92)
    plt.savefig(fig_dir / "F4_tradeoff_zoom.png", dpi=220, bbox_inches="tight")
    plt.close()


def make_f5_hetero_png(fig_dir: Path, hetero: pd.DataFrame) -> None:
    sub = hetero[hetero["product"].isin(["P1", "P2", "P3", "P4"])].copy()
    _require(not sub.empty, "No station heterogeneity data for F5.")
    sub["product_plot"] = sub["product"].map(_plot_product_key)
    sub = sub.groupby(["product_plot", "station_id"], as_index=False).agg(
        missing_additional=("missing_additional", "mean"),
        state_wind_spread=("state_wind_spread", "mean"),
    )
    product_order = PLOT_ORDER
    rng = np.random.default_rng(20260218)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for j, (col, ylabel, scale) in enumerate(
        [
            ("state_wind_spread", "state_wind_spread", 1.0),
            ("missing_additional", "missing_additional (%)", 100.0),
        ]
    ):
        data = [scale * sub[sub["product_plot"] == p][col].to_numpy(dtype=float) for p in product_order]
        ax[j].boxplot(data, tick_labels=[PLOT_LABELS[p] for p in product_order])
        for i, arr in enumerate(data, start=1):
            if arr.size == 0:
                continue
            xj = i + rng.normal(0, 0.04, size=arr.size)
            ax[j].scatter(xj, arr, s=34, alpha=0.9, color="#334155", edgecolor="white", linewidth=0.3)
        ax[j].set_ylabel(ylabel)
    ax[0].set_title("Spread por estacion (DIRTY, TEST)")
    ax[1].set_title("Missingness por estacion (DIRTY, TEST)")
    fig.suptitle("Heterogeneidad (n=10 estaciones DIRTY; Spread = P90-P10 de removal por decil)", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(fig_dir / "F5_station_heterogeneity.png", dpi=220, bbox_inches="tight")
    plt.savefig(fig_dir / "F5_station_heterogeneity.pdf", bbox_inches="tight")
    plt.close()


def make_f6_downstream_png(fig_dir: Path, cost: pd.DataFrame, tail: pd.DataFrame, gamma: float) -> None:
    c = cost[(cost["phase"] == "test") & (cost["scope"] == "global")].copy()
    t = tail[(tail["phase"] == "test") & (tail["scope"] == "global")].copy()
    _require(not c.empty, "No downstream cost data for F6.")
    c["product_plot"] = c["product"].map(_plot_product_key)
    t["product_plot"] = t["product"].map(_plot_product_key)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    marker_map = {"ridge": "o", "persistence": "s"}
    for model, g in c.groupby("model", sort=True):
        g2 = (
            g.groupby(["product_plot"], as_index=False)
            .agg(
                coverage_operational=("coverage_operational", "mean"),
                score_cost_sensitive=("score_cost_sensitive", "mean"),
            )
        )
        ax[0].scatter(
            g2["coverage_operational"],
            g2["score_cost_sensitive"],
            s=75,
            marker=marker_map.get(model, "o"),
            label=model,
        )
        for _, r in g2.iterrows():
            ax[0].annotate(
                PLOT_LABELS.get(str(r["product_plot"]), str(r["product_plot"])),
                (r["coverage_operational"], r["score_cost_sensitive"]),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8,
            )
    # connect same product across models to show cost-sensitive gap
    cp = c.pivot_table(index="product_plot", columns="model", values=["coverage_operational", "score_cost_sensitive"], aggfunc="mean")
    for p in cp.index:
        try:
            x1 = float(cp.loc[p, ("coverage_operational", "persistence")])
            y1 = float(cp.loc[p, ("score_cost_sensitive", "persistence")])
            x2 = float(cp.loc[p, ("coverage_operational", "ridge")])
            y2 = float(cp.loc[p, ("score_cost_sensitive", "ridge")])
            ax[0].plot([x1, x2], [y1, y2], linestyle=":", linewidth=0.9, color="gray", alpha=0.7)
        except Exception:
            continue
    ax[0].set_xlabel("Coverage operational")
    ax[0].set_ylabel("Cost-sensitive score")
    ax[0].set_title(f"Score = RMSE + gamma*(1-coverage), gamma={gamma:.4f}")
    ax[0].legend(fontsize=8)

    piv = t.pivot_table(index="product_plot", columns="model", values="tail_rmse_operational", aggfunc="mean")
    piv = piv.reindex(["P0", "P1", "PG", "P4"])
    piv = piv.rename(index={"P0": "P0", "P1": "P1", "PG": "Guaranteed", "P4": "P4"})
    piv.plot(kind="bar", ax=ax[1])
    ax[1].set_title("Tail RMSE operational (tail: y_true >= p95_train(station))")
    ax[1].set_ylabel(WIND_UNIT_LABEL)
    ax[1].set_xlabel("product")
    ax[1].tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.savefig(fig_dir / "F6_downstream_cost_tail.png", dpi=220, bbox_inches="tight")
    plt.savefig(fig_dir / "F6_downstream_cost_tail.pdf", bbox_inches="tight")
    plt.close()


def make_f7_synthetic_png(fig_dir: Path, syn: pd.DataFrame, syn_tail: pd.DataFrame) -> None:
    s = syn[syn["phase"] == "test"].copy()
    st = syn_tail[syn_tail["phase"] == "test"].copy()
    _require(not s.empty and not st.empty, "No synthetic data for F7.")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    # Panel A: MNAR spread vs exceedance distortion(u95)
    m = s.merge(st[["scope", "delta_exceed_u95_rel"]], on="scope", how="left")
    ax[0].axhline(0.0, color="black", linewidth=1)
    ax[0].scatter(m["state_wind_spread_vs_P2"], 100.0 * m["delta_exceed_u95_rel"], s=90, color="#0ea5e9")
    if len(m) >= 2:
        ms = m.sort_values("state_wind_spread_vs_P2")
        ax[0].plot(ms["state_wind_spread_vs_P2"], 100.0 * ms["delta_exceed_u95_rel"], linestyle="--", color="#0ea5e9", alpha=0.8)
    for _, r in m.iterrows():
        ax[0].annotate(
            str(r["scope"]),
            (r["state_wind_spread_vs_P2"], 100.0 * r["delta_exceed_u95_rel"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
        )
    ax[0].set_xlabel("MNAR spread vs P2")
    ax[0].set_ylabel("Delta exceedance u95 (%)")
    ax[0].set_title("MNAR vs tail-shape (u95)")
    ax[0].set_ylim(-7.0, 0.5)

    # Panel B: exceedance deltas u95/u99 (global vs dirty)
    scopes = ["global", "dirty"]
    x = np.arange(len(scopes))
    d95 = [100.0 * float(st[st["scope"] == sc]["delta_exceed_u95_rel"].iloc[0]) for sc in scopes]
    d99 = [100.0 * float(st[st["scope"] == sc]["delta_exceed_u99_rel"].iloc[0]) for sc in scopes]
    ax[1].axhline(0.0, color="black", linewidth=1)
    b1 = ax[1].bar(x - 0.18, d95, width=0.35, label="u95")
    b2 = ax[1].bar(x + 0.18, d99, width=0.35, label="u99")
    ax[1].set_xticks(x, scopes)
    u95 = float(st["threshold_u95"].iloc[0])
    u99 = float(st["threshold_u99"].iloc[0])
    ax[1].set_title(f"Exceedance distortion (u95={u95:.1f}, u99={u99:.1f})")
    ax[1].set_ylabel("Delta exceedance (%)")
    ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2, fontsize=8)
    for bars in [b1, b2]:
        for b in bars:
            h = b.get_height()
            ax[1].annotate(f"{h:.2f}", (b.get_x() + b.get_width() / 2, h), ha="center", va="bottom", fontsize=8)

    fig.suptitle("Synthetic: MNAR and tail-shape (S1 vs P2)", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = fig_dir / "F7_synthetic_mnar_vs_tailshift.png"
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.savefig(fig_dir / "F7_synthetic_mnar_vs_tailshift.pdf", bbox_inches="tight")
    plt.close()


def make_f8_benchmark_png(fig_dir: Path, bench: pd.DataFrame, selected_cfg: str) -> None:
    b = bench[bench["policy_suspect"] == "KEEP"].copy()
    if b.empty:
        b = bench.copy()
    _require(not b.empty, "No benchmark data for F8.")

    piv_r = b.pivot_table(index="T_hours", columns="tol_hpa", values="recall_stuck", aggfunc="mean").sort_index().sort_index(axis=1)
    piv_p = b.pivot_table(index="T_hours", columns="tol_hpa", values="precision_stuck", aggfunc="mean").sort_index().sort_index(axis=1)
    piv_tp = b.pivot_table(index="T_hours", columns="tol_hpa", values="tp", aggfunc="mean").sort_index().sort_index(axis=1)
    piv_p = piv_p.where(piv_tp > 0, np.nan)
    pass_mask = (piv_r >= 0.95) & (piv_p >= 0.90)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    im0 = ax[0].imshow(piv_r.to_numpy(dtype=float), vmin=0.0, vmax=1.0, cmap="YlGnBu", aspect="auto")
    cmap_p = plt.get_cmap("magma").copy()
    cmap_p.set_bad(color="lightgray")
    im1 = ax[1].imshow(piv_p.to_numpy(dtype=float), vmin=0.9, vmax=1.0, cmap=cmap_p, aspect="auto")
    images = [im0, im1]
    for j, (a, piv, title) in enumerate([(ax[0], piv_r, "Recall stuck"), (ax[1], piv_p, "Precision stuck")]):
        a.set_xticks(np.arange(len(piv.columns)), [f"{c:.2f}" for c in piv.columns])
        a.set_yticks(np.arange(len(piv.index)), [str(int(v)) for v in piv.index])
        a.set_xlabel("tol_hpa")
        a.set_ylabel("T_hours")
        a.set_title(title)
        for i in range(piv.shape[0]):
            for k in range(piv.shape[1]):
                val = piv.iloc[i, k]
                if np.isnan(val):
                    a.text(k, i, "NA", ha="center", va="center", fontsize=8, color="black")
                else:
                    rgba = images[j].cmap(images[j].norm(float(val)))
                    lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                    txt_color = "black" if lum > 0.60 else "white"
                    a.text(k, i, f"{float(val):.2f}", ha="center", va="center", fontsize=8, color=txt_color)
                if bool(pass_mask.iloc[i, k]) and (not np.isnan(val)):
                    rect = plt.Rectangle((k - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="lime", linewidth=1.8)
                    a.add_patch(rect)
    fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

    sel = b[b["config_key"] == selected_cfg]
    if not sel.empty:
        t = int(sel.iloc[0]["T_hours"])
        tol = float(sel.iloc[0]["tol_hpa"])
        if t in piv_r.index and tol in piv_r.columns:
            i = piv_r.index.tolist().index(t)
            k = piv_r.columns.tolist().index(tol)
            star_y = i - 0.18
            ax[0].scatter([k], [star_y], marker="*", s=160, color="yellow", edgecolor="black", zorder=5)
            ax[1].scatter([k], [star_y], marker="*", s=160, color="yellow", edgecolor="black", zorder=5)
    fig.suptitle("Benchmark detector power (green=pass C6; star=selected; precision NA if TP=0)", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(fig_dir / "F8_benchmark_detector_power.png", dpi=220, bbox_inches="tight")
    plt.savefig(fig_dir / "F8_benchmark_detector_power.pdf", bbox_inches="tight")
    plt.close()


def main() -> None:
    root = Path(".").resolve()
    r = root / "results_v3"
    fig_dir = r / "figures"
    _require(r.exists(), f"Missing results folder: {r}")
    fig_dir.mkdir(parents=True, exist_ok=True)

    pareto = _load_csv(r / "pareto_drop_rules.csv")
    tails = _load_csv(r / "metrics_tails.csv")
    mnar = _load_csv(r / "metrics_mnar.csv")
    remdec = _load_csv(r / "metrics_removal_by_decile.csv")
    rvk = _load_csv(r / "metrics_removed_vs_kept.csv")
    budget = _load_csv(r / "budget_search_grid_v3.csv")
    hetero = _load_csv(r / "metrics_station_heterogeneity.csv")
    bench = _load_csv(r / "benchmark_stuck" / "benchmark_perf_by_config.csv")
    cost = _load_csv(r / "downstream" / "downstream_cost_sensitive_v3.csv")
    tail = _load_csv(r / "downstream" / "downstream_tail_metrics_v3.csv")
    syn = _load_csv(r / "synthetic" / "synthetic_metrics.csv")
    syn_tail = _load_csv(r / "synthetic" / "synthetic_tailshape_metrics.csv")

    best_b10 = json.loads((r / "budget_best_B10_v3.json").read_text(encoding="utf-8"))
    chk = json.loads((r / "checklist_acceptance_v4.json").read_text(encoding="utf-8"))
    sel_bench = json.loads((r / "benchmark_stuck" / "benchmark_perf_selected_config.json").read_text(encoding="utf-8"))
    selected_cfg = str(sel_bench.get("config_key", best_b10.get("config_key", "")))
    gamma = float(chk.get("gamma", np.nan))

    make_f1_pareto_png(fig_dir, pareto)
    make_f2_tails_png(fig_dir, tails)
    make_f3_mnar_png(fig_dir, mnar, remdec, rvk)
    make_f4_tradeoff_png(fig_dir, budget, best_b10)
    make_f5_hetero_png(fig_dir, hetero)
    make_f6_downstream_png(fig_dir, cost, tail, gamma=gamma)
    make_f7_synthetic_png(fig_dir, syn, syn_tail)
    make_f8_benchmark_png(fig_dir, bench, selected_cfg=selected_cfg)

    print("PNG figures regenerated in results_v3/figures")


if __name__ == "__main__":
    main()
