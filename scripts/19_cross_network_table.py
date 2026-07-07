"""Build the cross-network generality table (AVAMET vs public DWD).

Reads the injected-fault-suite summaries produced by 11_measurement_fault_suite.py
for both networks and emits a compact LaTeX table contrasting rule vs
rule+spatial evidence recall for the spatially-recovered fault families. This is
the external-reproduction evidence: the same action-verification layer, applied
unchanged to a second, fully public network, recovers the same fault families
that rule evidence alone misses.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
AVAMET = PAPER / "runs_additional" / "BASELINE" / "results_v3" / "measurement_fault_suite" / "measurement_fault_suite_summary.csv"
DWD = PAPER / "runs_additional" / "DWD" / "results_v3" / "measurement_fault_suite" / "measurement_fault_suite_summary.csv"
OUT = PAPER / "article" / "tab_cross_network_generality.tex"

FAULTS = [
    ("pressure_offset", "Pressure offset"),
    ("pressure_drift", "Pressure drift"),
    ("wind_spike_burst", "Wind spike/burst"),
]


def recall(df, key, detector):
    row = df[(df["fault_type"].eq(key)) & (df["detector"].eq(detector))]
    if row.empty:
        return None
    return float(row["recall"].iloc[0]), float(row["precision"].iloc[0])


def main() -> None:
    a = pd.read_csv(AVAMET)
    d = pd.read_csv(DWD)
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r" & \multicolumn{2}{c}{AVAMET (regional, sparse)} & \multicolumn{2}{c}{DWD (national, dense, public)} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Injected fault & Rule rec. & Rule+spatial rec. & Rule rec. & Rule+spatial rec. \\",
        r"\midrule",
    ]
    for key, label in FAULTS:
        ar = recall(a, key, "Rule evidence")
        asp = recall(a, key, "Rule + spatial evidence")
        dr = recall(d, key, "Rule evidence")
        dsp = recall(d, key, "Rule + spatial evidence")
        lines.append(
            f"{label} & {ar[0]:.2f} & {asp[0]:.2f} & {dr[0]:.2f} & {dsp[0]:.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(OUT.read_text())
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
