from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

matplotlib.use("Agg")


FAULTS = [
    "pressure_stuck",
    "pressure_unit_scale",
    "wind_incoherence",
    "pressure_offset",
    "pressure_drift",
    "wind_spike_burst",
]

FAULT_LABELS = {
    "pressure_stuck": "Pressure stuck",
    "pressure_unit_scale": "Pressure unit-scale",
    "wind_incoherence": "Wind incoherence",
    "pressure_offset": "Pressure offset",
    "pressure_drift": "Pressure drift",
    "wind_spike_burst": "Wind spike/burst",
}

HARD_FAULTS = {
    "pressure_unit_scale",
    "wind_incoherence",
    "pressure_offset",
    "pressure_drift",
    "wind_spike_burst",
}

MAINTENANCE_SIGNAL_FAULTS = {"pressure_stuck"}

POLICY_ORDER = [
    "Raw",
    "Delete-all-flags",
    "Constraint-selected P3",
    "P4 pressure quarantine",
    "P5 full quarantine",
    "P6 full direct deletion",
]

POLICY_COLORS = {
    "Raw": "#4d4d4d",
    "Delete-all-flags": "#d62728",
    "Constraint-selected P3": "#009e73",
    "P4 pressure quarantine": "#0072b2",
    "P5 full quarantine": "#9467bd",
    "P6 full direct deletion": "#8c564b",
}


@dataclass(frozen=True)
class Scenario:
    name: str
    missed_hard_fault: float
    missed_maintenance_signal: float
    false_deletion: float
    false_maintenance_signal: float
    clean_availability_loss: float


SCENARIOS = [
    Scenario("Balanced", 10.0, 3.0, 2.0, 0.5, 1.0),
    Scenario("Safety-critical", 50.0, 5.0, 2.0, 0.5, 1.0),
    Scenario("Availability-critical", 10.0, 3.0, 20.0, 0.5, 10.0),
    Scenario("Maintenance-overloaded", 10.0, 3.0, 2.0, 5.0, 1.0),
    Scenario("Fault-conservative", 20.0, 5.0, 5.0, 1.0, 2.0),
]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Cost-sensitive fault-to-action analysis for reliability-oriented journal framing."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=root / "runs_additional" / "BASELINE" / "results_v3",
    )
    parser.add_argument("--article-dir", type=Path, default=root / "article")
    return parser.parse_args()


def load_inputs(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fault_dir = results_dir / "measurement_fault_suite"
    summary = pd.read_csv(fault_dir / "measurement_fault_suite_summary.csv")
    context = pd.read_csv(fault_dir / "measurement_fault_suite_policy_context.csv")
    removed = pd.read_csv(results_dir / "metrics_removed_vs_kept.csv")
    return summary, context, removed


def detector_row(summary: pd.DataFrame, fault_type: str, detector: str) -> pd.Series:
    rows = summary[summary["fault_type"].eq(fault_type) & summary["detector"].eq(detector)]
    if rows.empty:
        raise KeyError(f"Missing detector={detector!r} for fault_type={fault_type!r}")
    return rows.iloc[0]


def clean_false_rejection(context: pd.DataFrame, policy: str) -> float:
    rows = context[context["policy"].eq(policy)]
    if rows.empty:
        return 0.0
    return float(rows["false_rejection_clean_test"].iloc[0])


def action_counts(summary: pd.DataFrame, policy: str, fault_type: str) -> dict[str, float]:
    if policy == "Raw":
        template = detector_row(summary, fault_type, "Rule evidence")
        return {
            "action_tp": 0.0,
            "action_fp": 0.0,
            "action_fn": float(template["recall_n"]),
            "evidence_tp": 0.0,
            "evidence_fp": 0.0,
            "evidence_fn": float(template["recall_n"]),
            "n_injected": float(template["recall_n"]),
            "n_benchmark": float(template["n_benchmark_rows"]),
        }

    if policy == "Delete-all-flags":
        action = detector_row(summary, fault_type, "Aggressive deletion action")
        evidence = detector_row(summary, fault_type, "Rule evidence")
    elif policy == "Constraint-selected P3":
        action = detector_row(summary, fault_type, "P3 deletion action")
        evidence = detector_row(summary, fault_type, "Rule evidence")
    elif policy == "P4 pressure quarantine":
        evidence = detector_row(summary, fault_type, "Rule + spatial evidence")
        if fault_type in {"pressure_offset", "pressure_drift"}:
            action = detector_row(summary, fault_type, "Rule + spatial evidence")
        else:
            action = detector_row(summary, fault_type, "P3 deletion action")
    elif policy in {"P5 full quarantine", "P6 full direct deletion"}:
        evidence = detector_row(summary, fault_type, "Rule + spatial evidence")
        if fault_type == "pressure_stuck":
            action = detector_row(summary, fault_type, "P3 deletion action")
        elif fault_type in {"pressure_unit_scale", "wind_incoherence"}:
            action = detector_row(summary, fault_type, "P3 deletion action")
        else:
            action = detector_row(summary, fault_type, "Rule + spatial evidence")
    else:
        raise ValueError(policy)

    return {
        "action_tp": float(action["tp"]),
        "action_fp": float(action["fp"]),
        "action_fn": float(action["fn"]),
        "evidence_tp": float(evidence["tp"]),
        "evidence_fp": float(evidence["fp"]),
        "evidence_fn": float(evidence["fn"]),
        "n_injected": float(action["recall_n"]),
        "n_benchmark": float(action["n_benchmark_rows"]),
    }


def build_policy_metrics(summary: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for policy in POLICY_ORDER:
        hard_injected = 0.0
        missed_hard = 0.0
        maintenance_injected = 0.0
        missed_maintenance = 0.0
        false_deletions = 0.0
        false_maintenance = 0.0
        benchmark_context = 0.0

        for fault_type in FAULTS:
            counts = action_counts(summary, policy, fault_type)
            n_injected = counts["n_injected"]
            n_context = max(0.0, counts["n_benchmark"] - n_injected)
            benchmark_context += n_context
            false_deletions += counts["action_fp"]
            false_maintenance += max(0.0, counts["evidence_fp"] - counts["action_fp"])

            if fault_type in HARD_FAULTS:
                hard_injected += n_injected
                missed_hard += counts["action_fn"]
            if fault_type in MAINTENANCE_SIGNAL_FAULTS:
                maintenance_injected += n_injected
                if policy in {"Raw", "Delete-all-flags"}:
                    missed_maintenance += n_injected
                else:
                    missed_maintenance += counts["evidence_fn"]

        clean_loss = {
            "Raw": 0.0,
            "Delete-all-flags": clean_false_rejection(context, "Aggressive censoring"),
            "Constraint-selected P3": clean_false_rejection(context, "P3"),
            "P4 pressure quarantine": clean_false_rejection(context, "P3"),
            "P5 full quarantine": 0.002608744557905044,
            "P6 full direct deletion": 0.002608744557905044,
        }[policy]

        rows.append(
            {
                "policy": policy,
                "missed_hard_fault_rate": missed_hard / hard_injected,
                "missed_maintenance_signal_rate": missed_maintenance / maintenance_injected,
                "false_deletion_rate_context": false_deletions / benchmark_context,
                "false_maintenance_signal_rate_context": false_maintenance / benchmark_context,
                "clean_test_false_rejection_rate": clean_loss,
                "missed_hard_faults": int(missed_hard),
                "hard_fault_rows": int(hard_injected),
                "missed_maintenance_signal_rows": int(missed_maintenance),
                "maintenance_signal_rows": int(maintenance_injected),
                "false_deletion_context_rows": int(false_deletions),
                "false_maintenance_context_rows": int(false_maintenance),
                "benchmark_context_rows": int(benchmark_context),
            }
        )

    return pd.DataFrame(rows)


def score_metrics(metrics: pd.DataFrame, scenario: Scenario) -> pd.DataFrame:
    scored = metrics.copy()
    scored["loss_score"] = (
        scenario.missed_hard_fault * scored["missed_hard_fault_rate"]
        + scenario.missed_maintenance_signal * scored["missed_maintenance_signal_rate"]
        + scenario.false_deletion * scored["false_deletion_rate_context"]
        + scenario.false_maintenance_signal * scored["false_maintenance_signal_rate_context"]
        + scenario.clean_availability_loss * scored["clean_test_false_rejection_rate"]
    )
    scored["scenario"] = scenario.name
    scored["w_missed_hard_fault"] = scenario.missed_hard_fault
    scored["w_missed_maintenance_signal"] = scenario.missed_maintenance_signal
    scored["w_false_deletion"] = scenario.false_deletion
    scored["w_false_maintenance_signal"] = scenario.false_maintenance_signal
    scored["w_clean_availability_loss"] = scenario.clean_availability_loss
    return scored


def scenario_scores(metrics: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([score_metrics(metrics, scenario) for scenario in SCENARIOS], ignore_index=True)


def sensitivity_grid(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for missed_hard in [3.0, 10.0, 30.0, 100.0]:
        for missed_maintenance in [1.0, 3.0, 10.0]:
            for false_deletion in [1.0, 3.0, 10.0, 30.0]:
                for false_alarm in [0.1, 1.0, 5.0]:
                    scenario = Scenario(
                        name="grid",
                        missed_hard_fault=missed_hard,
                        missed_maintenance_signal=missed_maintenance,
                        false_deletion=false_deletion,
                        false_maintenance_signal=false_alarm,
                        clean_availability_loss=false_deletion,
                    )
                    scored = score_metrics(metrics, scenario)
                    winner = scored.loc[scored["loss_score"].idxmin(), "policy"]
                    scored["winner"] = winner
                    rows.append(scored)
    return pd.concat(rows, ignore_index=True)


def adversarial_sensitivity_grid(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for missed_hard in [1.0, 3.0, 10.0, 30.0, 100.0]:
        for missed_maintenance in [0.0, 1.0, 3.0, 10.0]:
            for false_deletion in [1.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0]:
                for false_alarm in [0.0, 0.1, 1.0, 5.0, 20.0]:
                    scenario = Scenario(
                        name="adversarial_grid",
                        missed_hard_fault=missed_hard,
                        missed_maintenance_signal=missed_maintenance,
                        false_deletion=false_deletion,
                        false_maintenance_signal=false_alarm,
                        clean_availability_loss=false_deletion,
                    )
                    scored = score_metrics(metrics, scenario)
                    winner = scored.loc[scored["loss_score"].idxmin(), "policy"]
                    scored["winner"] = winner
                    rows.append(scored)
    return pd.concat(rows, ignore_index=True)


def break_even_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    p3 = metrics[metrics["policy"].eq("Constraint-selected P3")].iloc[0]
    spatial = metrics[metrics["policy"].eq("P4 pressure quarantine")].iloc[0]
    delta_h = float(spatial["missed_hard_fault_rate"] - p3["missed_hard_fault_rate"])
    delta_d = float(spatial["false_deletion_rate_context"] - p3["false_deletion_rate_context"])
    delta_a = float(spatial["false_maintenance_signal_rate_context"] - p3["false_maintenance_signal_rate_context"])
    threshold_no_false_alarm = np.inf if abs(delta_d) < 1e-12 else -delta_h / delta_d
    return pd.DataFrame(
        [
            {
                "comparison": "P4 pressure quarantine vs Constraint-selected P3",
                "delta_missed_hard_fault_rate": delta_h,
                "delta_false_deletion_rate_context": delta_d,
                "delta_false_maintenance_signal_rate_context": delta_a,
                "break_even_w_false_deletion_over_w_missed_hard_fault_when_w_false_alarm_zero": threshold_no_false_alarm,
                "interpretation": "P3 wins only when false-deletion cost is greater than this multiple of missed-hard-fault cost, ignoring the small false-alarm advantage of P4.",
            }
        ]
    )


INELIGIBLE: set[str] = set()


def write_tex_table(scores: pd.DataFrame, path: Path) -> None:
    pivot = scores.pivot(index="scenario", columns="policy", values="loss_score")
    pivot = pivot[[policy for policy in POLICY_ORDER if policy in pivot.columns]]
    pivot = pivot.reindex([scenario.name for scenario in SCENARIOS])

    lines = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Scenario & Raw & Delete-all & P3 & P4 press. & P5 full & P6 direct \\\\",
        "\\midrule",
    ]
    for scenario, row in pivot.iterrows():
        best = row.drop(labels=[p for p in INELIGIBLE if p in row.index]).idxmin()
        values = []
        for policy in POLICY_ORDER:
            text = f"{row[policy]:.3f}"
            if policy == best:
                text = f"\\textbf{{{text}}}"
            values.append(text)
        lines.append(f"{scenario} & {' & '.join(values)} \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "",
            "% Scores are dimensionless weighted sums of missed hard faults, missed maintenance signals, false deletions, false maintenance signals, and clean-test false rejection.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_adversarial_summary(adversarial_winners: pd.DataFrame, break_even: pd.DataFrame, path: Path) -> None:
    lines = [
        "\\begin{tabular}{lrr}",
        "\\toprule",
        "Policy & Scenarios won & Share \\\\",
        "\\midrule",
    ]
    for _, row in adversarial_winners.iterrows():
        lines.append(f"{row['policy']} & {int(row['n_scenarios_won'])} & {row['share_scenarios_won']:.3f} \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "",
            (
                "% Break-even false-deletion / missed-hard-fault cost ratio for P3 to beat "
                f"P4, with false-alarm cost set to zero: "
                f"{break_even.iloc[0]['break_even_w_false_deletion_over_w_missed_hard_fault_when_w_false_alarm_zero']:.1f}"
            ),
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def make_risk_figure(scores: pd.DataFrame, metrics: pd.DataFrame, path_pdf: Path, path_png: Path) -> None:
    plot = scores.copy()
    scenarios = [scenario.name for scenario in SCENARIOS]
    x = np.arange(len(scenarios))
    width = 0.13

    fig = plt.figure(figsize=(7.4, 6.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.15, 1.0], hspace=0.54)
    ax = fig.add_subplot(gs[0, 0])
    for i, policy in enumerate(POLICY_ORDER):
        sub = plot[plot["policy"].eq(policy)].set_index("scenario").reindex(scenarios)
        ax.bar(
            x + (i - (len(POLICY_ORDER) - 1) / 2.0) * width,
            sub["loss_score"].to_numpy(dtype=float),
            width=width,
            label=policy,
            color=POLICY_COLORS[policy],
            edgecolor="black",
            linewidth=0.45,
        )

    ax.set_ylabel("Loss-weighted action score")
    ax.set_xticks(x)
    ax.set_xticklabels(["Balanced", "Safety\ncritical", "Availability\ncritical", "Maintenance\noverloaded", "Fault\nconservative"])
    ax.grid(axis="y", color="#dfe6ee", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.33), ncol=3, frameon=False, fontsize=8.0)
    ax.text(-0.075, 1.07, "A", transform=ax.transAxes, fontweight="bold", fontsize=11, va="top")

    component_cols = [
        ("missed_hard_fault_rate", "Missed\nhard faults"),
        ("missed_maintenance_signal_rate", "Missed\nmaint. signals"),
        ("false_deletion_rate_context", "False\ndeletions"),
        ("false_maintenance_signal_rate_context", "False\nmaint. signals"),
        ("clean_test_false_rejection_rate", "Clean false\nrejection"),
    ]
    heat = metrics.set_index("policy").reindex(POLICY_ORDER)
    values = heat[[col for col, _ in component_cols]].to_numpy(dtype=float)
    display_values = np.log10(values + 1e-5)

    ax_h = fig.add_subplot(gs[1, 0])
    im = ax_h.imshow(display_values, cmap="YlOrRd", aspect="auto", vmin=-5, vmax=0)
    ax_h.set_yticks(np.arange(len(POLICY_ORDER)))
    ax_h.set_yticklabels(["Raw", "Delete-all", "P3", "P4 press.", "P5 full", "P6 direct"])
    ax_h.set_xticks(np.arange(len(component_cols)))
    ax_h.set_xticklabels([label for _, label in component_cols])
    ax_h.tick_params(axis="x", labelsize=8)
    ax_h.tick_params(axis="y", labelsize=8)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            text = "0" if value == 0 else f"{value:.3g}"
            ax_h.text(col, row, text, ha="center", va="center", fontsize=7.5, color="black")
    for spine in ax_h.spines.values():
        spine.set_visible(False)
    ax_h.set_xticks(np.arange(-0.5, len(component_cols), 1), minor=True)
    ax_h.set_yticks(np.arange(-0.5, len(POLICY_ORDER), 1), minor=True)
    ax_h.grid(which="minor", color="white", linewidth=1.0)
    ax_h.tick_params(which="minor", bottom=False, left=False)
    ax_h.text(-0.075, 1.10, "B", transform=ax_h.transAxes, fontweight="bold", fontsize=11, va="top")
    cbar = fig.colorbar(im, ax=ax_h, fraction=0.025, pad=0.02)
    cbar.set_label("log10(rate + 1e-5)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.tight_layout()
    fig.savefig(path_pdf, bbox_inches="tight")
    fig.savefig(path_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_component_figure(metrics: pd.DataFrame, path_pdf: Path, path_png: Path) -> None:
    components = [
        ("missed_hard_fault_rate", "Missed hard faults"),
        ("missed_maintenance_signal_rate", "Missed maintenance signals"),
        ("false_deletion_rate_context", "False deletions"),
        ("false_maintenance_signal_rate_context", "False maintenance signals"),
        ("clean_test_false_rejection_rate", "Clean-test false rejection"),
    ]
    x = np.arange(len(components))
    width = 0.13

    fig, ax = plt.subplots(figsize=(9.4, 4.5))
    for i, policy in enumerate(POLICY_ORDER):
        sub = metrics[metrics["policy"].eq(policy)].iloc[0]
        values = [float(sub[col]) for col, _ in components]
        ax.bar(
            x + (i - (len(POLICY_ORDER) - 1) / 2.0) * width,
            values,
            width=width,
            label=policy,
            color=POLICY_COLORS[policy],
            edgecolor="black",
            linewidth=0.45,
        )

    ax.set_yscale("symlog", linthresh=1e-4)
    ax.set_ylabel("Rate")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in components], rotation=20, ha="right")
    ax.grid(axis="y", color="#dfe6ee", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.24), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(path_pdf, bbox_inches="tight")
    fig.savefig(path_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summary, context, _ = load_inputs(args.results_dir)
    ret_path = args.results_dir / 'spatial_quarantine_archive' / 'archive_retention.csv'
    if ret_path.exists():
        ret = pd.read_csv(ret_path)
        ret = ret[(ret.threshold == 'u99') & (ret.policy == 'P5 full spatial quarantine')].retention
        if len(ret) and float(ret.iloc[0]) < 0.95:
            INELIGIBLE.update({'P5 full quarantine', 'P6 full direct deletion'})
    out_dir = args.results_dir / "operational_risk"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = build_policy_metrics(summary, context)
    scores = scenario_scores(metrics)
    grid = sensitivity_grid(metrics)
    adversarial_grid = adversarial_sensitivity_grid(metrics)
    break_even = break_even_summary(metrics)
    winners = (
        grid.drop_duplicates(
            [
                "w_missed_hard_fault",
                "w_missed_maintenance_signal",
                "w_false_deletion",
                "w_false_maintenance_signal",
                "w_clean_availability_loss",
            ]
        )["winner"]
        .value_counts()
        .rename_axis("policy")
        .reset_index(name="n_scenarios_won")
    )
    winners["share_scenarios_won"] = winners["n_scenarios_won"] / winners["n_scenarios_won"].sum()
    adversarial_winners = (
        adversarial_grid.drop_duplicates(
            [
                "w_missed_hard_fault",
                "w_missed_maintenance_signal",
                "w_false_deletion",
                "w_false_maintenance_signal",
                "w_clean_availability_loss",
            ]
        )["winner"]
        .value_counts()
        .rename_axis("policy")
        .reset_index(name="n_scenarios_won")
    )
    adversarial_winners["share_scenarios_won"] = (
        adversarial_winners["n_scenarios_won"] / adversarial_winners["n_scenarios_won"].sum()
    )

    metrics.to_csv(out_dir / "operational_risk_components.csv", index=False)
    scores.to_csv(out_dir / "operational_risk_scenarios.csv", index=False)
    grid.to_csv(out_dir / "operational_risk_sensitivity_grid.csv", index=False)
    winners.to_csv(out_dir / "operational_risk_sensitivity_winners.csv", index=False)
    adversarial_grid.to_csv(out_dir / "operational_risk_adversarial_sensitivity_grid.csv", index=False)
    adversarial_winners.to_csv(out_dir / "operational_risk_adversarial_sensitivity_winners.csv", index=False)
    break_even.to_csv(out_dir / "operational_risk_break_even.csv", index=False)

    tex_path = out_dir / "tab_operational_risk_scenarios.tex"
    write_tex_table(scores, tex_path)
    write_adversarial_summary(
        adversarial_winners,
        break_even,
        out_dir / "tab_operational_risk_adversarial_summary.tex",
    )
    risk_pdf = out_dir / "F9_operational_risk_scenarios.pdf"
    risk_png = out_dir / "F9_operational_risk_scenarios.png"
    components_pdf = out_dir / "S_operational_risk_components.pdf"
    components_png = out_dir / "S_operational_risk_components.png"
    make_risk_figure(scores, metrics, risk_pdf, risk_png)
    make_component_figure(metrics, components_pdf, components_png)

    args.article_dir.mkdir(parents=True, exist_ok=True)
    (args.article_dir / "tab_operational_risk_scenarios.tex").write_text(
        tex_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (args.article_dir / "F9_operational_risk_scenarios.pdf").write_bytes(risk_pdf.read_bytes())
    (args.article_dir / "F9_operational_risk_scenarios.png").write_bytes(risk_png.read_bytes())
    (args.article_dir / "S_operational_risk_components.pdf").write_bytes(components_pdf.read_bytes())

    print(f"Wrote {out_dir / 'operational_risk_components.csv'}")
    print(f"Wrote {out_dir / 'operational_risk_scenarios.csv'}")
    print(f"Wrote {out_dir / 'operational_risk_sensitivity_winners.csv'}")
    print(f"Wrote {out_dir / 'operational_risk_adversarial_sensitivity_winners.csv'}")
    print(f"Wrote {out_dir / 'operational_risk_break_even.csv'}")
    print(f"Wrote {args.article_dir / 'tab_operational_risk_scenarios.tex'}")
    print(f"Wrote {args.article_dir / 'F9_operational_risk_scenarios.pdf'}")


if __name__ == "__main__":
    main()
