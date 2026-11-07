from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDOneClassSVM
from sklearn.metrics import precision_recall_curve
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_v3"
OUT = RESULTS / "computational_intelligence"
RANDOM_STATE = 42
TRAIN_END = pd.Timestamp("2025-01-01", tz="UTC")
FEATURES = [
    "wind_gust",
    "wind_mean",
    "pressure_hpa",
    "wind_dir_sin",
    "wind_dir_cos",
    "gust_minus_mean",
    "gust_to_mean",
    "pressure_delta_abs",
]
BUDGETS = [0.05, 0.10]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["station_id", "timestamp_utc"]).copy()
    direction = np.deg2rad(df["wind_dir"].astype(float))
    df["wind_dir_sin"] = np.sin(direction)
    df["wind_dir_cos"] = np.cos(direction)
    df["gust_minus_mean"] = df["wind_gust"] - df["wind_mean"]
    df["gust_to_mean"] = df["wind_gust"] / (df["wind_mean"].abs() + 0.1)
    df["pressure_delta_abs"] = (
        df.groupby("station_id", sort=False)["pressure_hpa"].diff().abs()
    )
    return df


def split_phase(df: pd.DataFrame) -> pd.Series:
    ts = pd.to_datetime(df["timestamp_utc"], utc=True)
    return np.where(ts < TRAIN_END, "train", "test")


def state_wind_spread(wind: pd.Series, removed: np.ndarray) -> float:
    bins = pd.qcut(wind, 10, labels=False, duplicates="drop")
    rates = []
    for bin_id in sorted(pd.Series(bins).dropna().unique()):
        idx = np.asarray(bins == bin_id)
        if idx.any():
            rates.append(float(np.mean(removed[idx])))
    if len(rates) == 0:
        return float("nan")
    return float(np.quantile(rates, 0.90) - np.quantile(rates, 0.10))


def retention(wind: pd.Series, removed: np.ndarray, threshold: float) -> float:
    idx = np.asarray(wind >= threshold)
    if not idx.any():
        return float("nan")
    return float(np.mean(~removed[idx]))


def standard_matrix(df: pd.DataFrame, sample_size: int) -> tuple[np.ndarray, np.ndarray]:
    x = df[FEATURES].replace([np.inf, -np.inf], np.nan)
    if len(x) > sample_size:
        sample = x.sample(sample_size, random_state=RANDOM_STATE)
    else:
        sample = x
    return x, sample


def score_isolation_forest(train_df: pd.DataFrame, all_df: pd.DataFrame) -> np.ndarray:
    x_all, x_train = standard_matrix(train_df, 120_000)
    _ = x_all
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        IsolationForest(
            n_estimators=250,
            max_samples=4096,
            contamination="auto",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    )
    model.fit(x_train)
    return -model.decision_function(all_df[FEATURES].replace([np.inf, -np.inf], np.nan))


def score_kernel_ocsvm(train_df: pd.DataFrame, all_df: pd.DataFrame) -> np.ndarray:
    _, x_train = standard_matrix(train_df, 100_000)
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Nystroem(gamma=0.2, n_components=300, random_state=RANDOM_STATE),
        SGDOneClassSVM(nu=0.05, random_state=RANDOM_STATE, max_iter=2000, tol=1e-4),
    )
    model.fit(x_train)
    return -model.decision_function(all_df[FEATURES].replace([np.inf, -np.inf], np.nan))


def score_mlp_autoencoder(train_df: pd.DataFrame, all_df: pd.DataFrame) -> np.ndarray:
    _, x_train = standard_matrix(train_df, 80_000)
    prep = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    x_fit = prep.fit_transform(x_train)
    model = MLPRegressor(
        hidden_layer_sizes=(12, 4, 12),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=1024,
        learning_rate_init=1e-3,
        max_iter=80,
        random_state=RANDOM_STATE,
        early_stopping=True,
        n_iter_no_change=8,
    )
    model.fit(x_fit, x_fit)
    x_all = prep.transform(all_df[FEATURES].replace([np.inf, -np.inf], np.nan))
    recon = model.predict(x_all)
    return np.mean((x_all - recon) ** 2, axis=1)


def score_robust_distance(train_df: pd.DataFrame, all_df: pd.DataFrame) -> np.ndarray:
    x_train = train_df[FEATURES].replace([np.inf, -np.inf], np.nan)
    x_all = all_df[FEATURES].replace([np.inf, -np.inf], np.nan)
    med = x_train.median()
    mad = (x_train - med).abs().median().replace(0, np.nan)
    z = ((x_all - med) / mad).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return np.sqrt((z**2).sum(axis=1)).to_numpy()


def write_ci_tradeoff_figure(compact: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    colors = {
        "Constraint-selected P3": "#1b9e77",
        "IF": "#d95f02",
        "kOCSVM": "#7570b3",
        "MLP-AE": "#e7298a",
        "robust-z": "#666666",
    }
    label_offsets = {
        "Constraint-selected P3": (18, 14, "left"),
        "IF": (10, 10, "left"),
        "kOCSVM": (10, 8, "left"),
        "MLP-AE": (-12, 14, "right"),
        "robust-z": (12, -18, "left"),
    }
    for _, row in compact.iterrows():
        size = 85 + 360 * float(row["retention_u99"])
        ax.scatter(
            row["missing_additional"],
            row["state_wind_spread"],
            s=size,
            color=colors.get(row["policy"], "#444444"),
            alpha=0.85,
            edgecolor="white",
            linewidth=1.1,
            zorder=3,
        )
        dx, dy, ha = label_offsets[row["policy"]]
        ax.annotate(
            row["policy"],
            (row["missing_additional"], row["state_wind_spread"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.2,
            ha=ha,
            va="center",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.78),
        )
    ax.set_xlabel("Additional missingness (TEST dirty)")
    ax.set_ylabel("State-dependent spread (TEST dirty)")
    ax.axvline(0.05, color="#777777", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.text(
        0.052,
        0.186,
        "5% TRAIN budget",
        ha="left",
        va="top",
        fontsize=8.0,
        color="#555555",
    )
    ax.grid(True, color="#dddddd", linewidth=0.55, alpha=0.75)
    ax.set_xlim(-0.004, 0.49)
    ax.set_ylim(-0.004, 0.19)
    ax.text(
        0.965,
        0.055,
        "Marker area = u99 retention",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.0,
        color="#444444",
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=1.0)
    for path in [
        OUT / "F6_ci_policy_tradeoff.pdf",
        OUT / "F6_ci_policy_tradeoff.png",
        ROOT / "article" / "F6_ci_policy_tradeoff.pdf",
    ]:
        fig.savefig(path, dpi=300)
    plt.close(fig)


def write_architecture_figure() -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    edge = "#2f3437"
    arrow = "#3f4549"
    muted = "#60686f"
    blue = "#d9e8f6"
    green = "#dcefd4"
    amber = "#f7e0b8"
    red = "#f5d6d6"
    purple = "#eadff7"
    slate = "#e8edf2"

    def box(x, y, w, h, title, body, face, lw=1.0):
        rect = plt.Rectangle((x, y), w, h, facecolor=face, edgecolor=edge, linewidth=lw)
        ax.add_patch(rect)
        ax.text(x + 0.018, y + h - 0.055, title, ha="left", va="top",
                fontsize=9.2, fontweight="bold", color="#111111")
        ax.text(x + 0.018, y + h - 0.125, body, ha="left", va="top",
                fontsize=7.7, color="#222222", linespacing=1.15)

    def pill(x, y, w, h, text, face):
        rect = plt.Rectangle((x, y), w, h, facecolor=face, edgecolor=edge, linewidth=0.9)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=7.4, color="#111111")

    def connect(start, end, rad=0.0, lw=1.25):
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops=dict(
                arrowstyle="->",
                color=arrow,
                linewidth=lw,
                shrinkA=2,
                shrinkB=2,
                connectionstyle=f"arc3,rad={rad}",
            ),
        )

    ax.text(
        0.02,
        0.94,
        "Constraint-verified fault-to-action decision layer",
        ha="left",
        va="center",
        fontsize=10.6,
        fontweight="bold",
        color="#111111",
    )
    ax.text(
        0.02,
        0.885,
        "Detector evidence is not automatically converted into deletion; actions are selected after operational checks.",
        ha="left",
        va="center",
        fontsize=7.6,
        color=muted,
    )

    box(
        0.035,
        0.56,
        0.18,
        0.21,
        "Sensor data",
        "Raw streams\nstations, variables,\ntime stamps",
        blue,
    )
    box(
        0.275,
        0.56,
        0.22,
        0.21,
        "Fault evidence",
        "Rule checks\npersistence/coherence\nanomaly scores",
        green,
    )
    box(
        0.555,
        0.56,
        0.22,
        0.21,
        "Policy mapping",
        "Map evidence to\ndelete, retain+flag,\nquarantine, escalate",
        amber,
        lw=1.25,
    )
    box(
        0.835,
        0.56,
        0.14,
        0.21,
        "Actions",
        "Filtered data\noperator alert\naudit trail",
        green,
    )

    ax.text(0.035, 0.45, "Verification gate", ha="left", va="center",
            fontsize=8.4, fontweight="bold", color="#111111")
    ax.plot([0.035, 0.975], [0.425, 0.425], color="#c6cbd1", lw=0.9)
    pill(0.055, 0.285, 0.14, 0.075, "availability\nbudget", red)
    pill(0.225, 0.285, 0.14, 0.075, "MNAR / state\ndependence", red)
    pill(0.395, 0.285, 0.14, 0.075, "benchmark\npower", red)
    pill(0.565, 0.285, 0.14, 0.075, "spatial support\nfor extremes", red)
    pill(0.735, 0.285, 0.14, 0.075, "risk-weighted\ncosts", red)

    box(
        0.635,
        0.055,
        0.34,
        0.16,
        "Reported artefacts",
        "",
        purple,
        lw=0.9,
    )
    ax.text(
        0.653,
        0.125,
        "selected policy and cost grid\ndiagnostics, reproducible outputs",
        ha="left",
        va="top",
        fontsize=7.7,
        color="#222222",
        linespacing=1.12,
    )

    connect((0.215, 0.665), (0.275, 0.665))
    connect((0.495, 0.665), (0.555, 0.665))
    connect((0.775, 0.665), (0.835, 0.665))
    connect((0.665, 0.56), (0.665, 0.365))
    connect((0.805, 0.285), (0.835, 0.56), rad=-0.22)
    connect((0.805, 0.285), (0.72, 0.215))

    ax.add_patch(plt.Rectangle((0.035, 0.105), 0.51, 0.105,
                               facecolor=slate, edgecolor="#9ba4ad", linewidth=0.8))
    ax.text(
        0.055,
        0.158,
        "Decision rule: choose the least destructive action that passes the gate.",
        ha="left",
        va="center",
        fontsize=7.7,
        color="#222222",
    )

    fig.tight_layout(pad=0.35)
    for path in [
        OUT / "F0_hybrid_qc_architecture.pdf",
        OUT / "F0_hybrid_qc_architecture.png",
        ROOT / "article" / "F0_hybrid_qc_architecture.pdf",
        ROOT / "article" / "Figure_1_architecture.png",
    ]:
        fig.savefig(path, dpi=1000 if path.suffix.lower() == ".png" else 300)
    plt.close(fig)


def run_supervised_stuck_benchmark(df_features: pd.DataFrame) -> None:
    labels = pd.read_parquet(RESULTS / "benchmark_stuck" / "benchmark_labels.parquet")
    bench = labels.merge(
        df_features[
            [
                "station_id",
                "timestamp_utc",
                "wind_gust",
                "wind_mean",
                "pressure_hpa",
                "wind_dir",
            ]
        ],
        on=["station_id", "timestamp_utc"],
        how="left",
    )
    bench = add_features(bench)
    bench["split"] = split_phase(bench)
    train = bench[bench["split"] == "train"].copy()
    test = bench[bench["split"] == "test"].copy()
    x_train = train[FEATURES].replace([np.inf, -np.inf], np.nan)
    y_train = train["stuck_injected"].astype(int).to_numpy()
    x_test = test[FEATURES].replace([np.inf, -np.inf], np.nan)
    y_test = test["stuck_injected"].astype(int).to_numpy()

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.05,
            l2_regularization=0.01,
            random_state=RANDOM_STATE,
        ),
    )
    model.fit(x_train, y_train)
    p_train = model.predict_proba(x_train)[:, 1]
    p_test = model.predict_proba(x_test)[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_train, p_train)
    candidates = []
    for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
        if r >= 0.95:
            candidates.append((p, r, t))
    if candidates:
        _, _, threshold = max(candidates, key=lambda item: (item[0], item[2]))
    else:
        threshold = 0.5

    rows = []
    for split, y, prob in [("train", y_train, p_train), ("test", y_test, p_test)]:
        pred = prob >= threshold
        tp = int(np.sum((pred == 1) & (y == 1)))
        fp = int(np.sum((pred == 1) & (y == 0)))
        fn = int(np.sum((pred == 0) & (y == 1)))
        tn = int(np.sum((pred == 0) & (y == 0)))
        rows.append(
            {
                "model": "HGB-supervised",
                "split": split,
                "threshold": float(threshold),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "recall": tp / (tp + fn) if (tp + fn) else np.nan,
                "precision": tp / (tp + fp) if (tp + fp) else np.nan,
            }
        )
    pd.DataFrame(rows).to_csv(OUT / "ci_supervised_stuck_benchmark.csv", index=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(RESULTS / "products" / "P0" / "data.parquet")
    df = add_features(df)
    df["phase"] = split_phase(df)

    dirty = df[df["stratum"] == "dirty"].reset_index(drop=True)
    train_dirty = dirty[dirty["phase"] == "train"].reset_index(drop=True)
    test_dirty = dirty[dirty["phase"] == "test"].reset_index(drop=True)
    u95 = float(train_dirty["wind_gust"].quantile(0.95))
    u99 = float(train_dirty["wind_gust"].quantile(0.99))

    scorers = {
        "IF": score_isolation_forest,
        "kOCSVM": score_kernel_ocsvm,
        "MLP-AE": score_mlp_autoencoder,
        "robust-z": score_robust_distance,
    }

    records = []
    score_store = {}
    for name, scorer in scorers.items():
        scores = scorer(train_dirty, dirty)
        score_store[name] = scores
        train_scores = scores[dirty["phase"].to_numpy() == "train"]
        test_scores = scores[dirty["phase"].to_numpy() == "test"]
        for budget in BUDGETS:
            threshold = float(np.quantile(train_scores, 1.0 - budget))
            for phase, phase_df, phase_scores in [
                ("train", train_dirty, train_scores),
                ("test", test_dirty, test_scores),
            ]:
                removed = phase_scores >= threshold
                records.append(
                    {
                        "policy": name,
                        "budget_train_dirty": budget,
                        "phase": phase,
                        "scope": "dirty",
                        "missing_additional": float(np.mean(removed)),
                        "state_wind_spread": state_wind_spread(
                            phase_df["wind_gust"].reset_index(drop=True), removed
                        ),
                        "retention_u95": retention(
                            phase_df["wind_gust"].reset_index(drop=True), removed, u95
                        ),
                        "retention_u99": retention(
                            phase_df["wind_gust"].reset_index(drop=True), removed, u99
                        ),
                        "threshold_score": threshold,
                    }
                )

    summary = pd.DataFrame.from_records(records)
    summary.to_csv(OUT / "ci_policy_comparison.csv", index=False)

    # Compact manuscript table: compare the selected policy to CI policies
    # at the stricter 5% calibration budget on the held-out TEST dirty stratum.
    p3_missing = pd.read_csv(RESULTS / "metrics_missingness.csv")
    p3_mnar = pd.read_csv(RESULTS / "metrics_mnar.csv")
    p3_test_missing = float(
        p3_missing.query(
            "product == 'P3' and phase == 'test' and scope == 'dirty' and variable == 'wind_gust'"
        )["missing_additional"].iloc[0]
    )
    p3_test_mnar = float(
        p3_mnar.query("product == 'P3' and phase == 'test' and scope == 'dirty'")[
            "state_wind_spread"
        ].iloc[0]
    )
    compact = summary.query(
        "phase == 'test' and budget_train_dirty == 0.05"
    ).copy()
    compact = compact[
        [
            "policy",
            "missing_additional",
            "state_wind_spread",
            "retention_u95",
            "retention_u99",
        ]
    ]
    compact = pd.concat(
        [
            pd.DataFrame(
                [
                    {
                        "policy": "Constraint-selected P3",
                        "missing_additional": p3_test_missing,
                        "state_wind_spread": p3_test_mnar,
                        "retention_u95": 1.0,
                        "retention_u99": 1.0,
                    }
                ]
            ),
            compact,
        ],
        ignore_index=True,
    )
    compact.to_csv(OUT / "ci_policy_comparison_test_dirty_B05.csv", index=False)
    write_architecture_figure()
    write_ci_tradeoff_figure(compact)
    run_supervised_stuck_benchmark(df)


if __name__ == "__main__":
    main()
