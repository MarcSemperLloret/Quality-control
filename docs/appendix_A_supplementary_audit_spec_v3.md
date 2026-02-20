# Appendix A: Auditable QC Specification (Paper 2, v3)

This appendix is a reproducible, code-aligned specification of:
- `S1` rule catalog,
- `S2` full configuration grid,
- formal criteria `C1..C6`,
- anti-gaming benchmark protocol,
- artifact inventory and minimum regeneration command.

Primary implementation source:
- `paper 2/src/06_run_qc_with_guarantees_v3.py`

Primary generated artifacts:
- `results_v3/`

## 1) Reproducibility Snapshot

- Data window:
  - `START=2023-01-01 00:00:00 UTC`
  - `END=2026-01-01 00:00:00 UTC`
  - `TRAIN=[2023-01-01, 2025-01-01)`
  - `TEST=[2025-01-01, 2026-01-01)`
- Active budgets: `B_SET={0.05, 0.10}`.
- Config grid: `T_hours x tol_hpa x policy_suspect = 5 x 3 x 2 = 30`.
- Selected stations: `30` total (`10 clean + 10 medium + 10 dirty`).
- Selected P3 config in this run: `T3_tol0.05_KEEP`.
- Workspace git metadata:
  - `git rev-parse --show-toplevel` -> not a git repository in this workspace.
  - Fill commit/tag externally when exporting this appendix to the manuscript package.

## 2) S1 - QC Rule Catalog (Auditable)

### 2.1 Rule families and formal behavior

| Rule ID | Human name | Variable(s) | Signal type | Rule output | Parameters / units | Prerequisites | Where computed |
|---|---|---|---|---|---|---|---|
| `pressure_fail_zero` | Pressure zero gross error | `pressure_raw` | Univariate | `FAIL` (binary) | `pressure_raw == 0` | non-null pressure | `06_run_qc_with_guarantees_v3.py` (canonical table build) |
| `pressure_fail_out_range_after_norm` | Pressure out-of-range after normalization | `pressure_hpa` | Univariate + unit-normalized | `FAIL` (binary) | `pressure_hpa < 850` or `> 1100` hPa | station unit classification + conversion | same |
| `fail_wind_negative` | Negative wind values | `wind_mean`, `wind_gust` | Univariate / dual-variable | `FAIL` (binary) | any wind value `< 0` | non-null wind fields | same |
| `fail_wind_incoherent` | Internal wind consistency | `wind_mean`, `wind_gust` | Multivariable consistency | `FAIL` (binary) | `wind_mean > wind_gust + 1e-9` | both wind fields present | same |
| `fail_wind_dir_999` | Wind direction sentinel | `wind_dir` | Univariate | `FAIL` (binary) | `wind_dir == 999` | non-null wind direction | same |
| `fail_wind_dir_range` | Wind direction out-of-range | `wind_dir` | Univariate | `FAIL` (binary) | `wind_dir < 0` or `> 360` | non-null wind direction | same |
| `flag_unit_converted` | Unit/scale anomaly flag (station-level) | `pressure_raw -> pressure_hpa` | Station-level + metadata derived from TRAIN quantiles | `FLAG` (binary, not hard fail) | classify station as `Pa`, `mmHg`, or `hPa`; convert `Pa/100` or `mmHg*1.3332239` | sufficient TRAIN pressure history per station | same |
| `suspect_stuck_T{T}_tol{tol}` | Pressure persistence / stall detector | `pressure_hpa` + timestamp | Time-series univariate with temporal continuity | `SUSPECT` (binary) | run duration `>= T_hours`; continuity break if gap `> 2*median_train_delta_s` or `|delta pressure| > tol_hpa` | station-level median TRAIN sampling interval | same |

Notes:
- `suspect_stuck_*` is computed for all `T in {3,6,12,24,48}` and `tol in {0.05,0.10,0.20}` hPa.
- Direction failures (`fail_wind_dir_999`, `fail_wind_dir_range`) are applied to direction field sanitization (`wind_dir -> NaN`) and are not part of the main row-removal hard-fail mask for gust/pressure.

### 2.2 Rule-to-removal mapping (DROP vs KEEP)

Row-keep masks by product (`P0..P5`):

| Product | Policy | FAIL rows | SUSPECT(stuck) rows | `flag_unit_converted` rows | Practical meaning |
|---|---|---|---|---|---|
| `P0` | raw canonical | KEEP | KEEP | KEEP | reference stream |
| `P1` | strict censor | DROP | DROP (`T_ref=3`, `tol_ref=0.10`) | DROP | aggressive baseline |
| `P2` | convert+verify | DROP | KEEP | KEEP | fail-only curated baseline |
| `P3` | budgeted v3 | DROP | depends on config (`DROP` or `KEEP`) | KEEP | optimized policy under constraints |
| `P4` | policy baseline | DROP | DROP (`T_ref=3`, `tol_ref=0.10`) | KEEP | non-budgeted baseline |
| `P5` | tool baseline | in this run unavailable | unavailable | unavailable | `SaQC` not installed in this environment |

SaQC mapping status:
- No operational SaQC plugin mapping is active in this run.
- `results_v3/tool_baseline_unavailable.txt` and `results_v3/products/P5/manifest.json` explicitly document this.

## 3) S2 - Full Configuration Grid Manifest (30 configs)

Canonical grid definition:
- `T_hours = {3, 6, 12, 24, 48}`
- `tol_hpa = {0.05, 0.10, 0.20}`
- `policy_suspect = {KEEP, DROP}`
- Cartesian product => `30` configurations.

Canonical files:
- `results_v3/benchmark_stuck/benchmark_perf_by_config.csv` (exactly 30 rows, one per config).
- `results_v3/budget_search_grid_v3.csv` (config evaluation expanded over budgets and folds).

Full list:

| Config ID | T_hours | tol_hpa | policy_suspect | Package | Description |
|---|---:|---:|---|---|---|
| T3_tol0.05_DROP | 3 | 0.05 | DROP | FAIL+SUSPECT | Persistence 3h, tol=0.05 hPa, DROP suspect segments |
| T3_tol0.05_KEEP | 3 | 0.05 | KEEP | FAIL-only | Persistence 3h, tol=0.05 hPa, KEEP suspect segments |
| T3_tol0.10_DROP | 3 | 0.10 | DROP | FAIL+SUSPECT | Persistence 3h, tol=0.10 hPa, DROP suspect segments |
| T3_tol0.10_KEEP | 3 | 0.10 | KEEP | FAIL-only | Persistence 3h, tol=0.10 hPa, KEEP suspect segments |
| T3_tol0.20_DROP | 3 | 0.20 | DROP | FAIL+SUSPECT | Persistence 3h, tol=0.20 hPa, DROP suspect segments |
| T3_tol0.20_KEEP | 3 | 0.20 | KEEP | FAIL-only | Persistence 3h, tol=0.20 hPa, KEEP suspect segments |
| T6_tol0.05_DROP | 6 | 0.05 | DROP | FAIL+SUSPECT | Persistence 6h, tol=0.05 hPa, DROP suspect segments |
| T6_tol0.05_KEEP | 6 | 0.05 | KEEP | FAIL-only | Persistence 6h, tol=0.05 hPa, KEEP suspect segments |
| T6_tol0.10_DROP | 6 | 0.10 | DROP | FAIL+SUSPECT | Persistence 6h, tol=0.10 hPa, DROP suspect segments |
| T6_tol0.10_KEEP | 6 | 0.10 | KEEP | FAIL-only | Persistence 6h, tol=0.10 hPa, KEEP suspect segments |
| T6_tol0.20_DROP | 6 | 0.20 | DROP | FAIL+SUSPECT | Persistence 6h, tol=0.20 hPa, DROP suspect segments |
| T6_tol0.20_KEEP | 6 | 0.20 | KEEP | FAIL-only | Persistence 6h, tol=0.20 hPa, KEEP suspect segments |
| T12_tol0.05_DROP | 12 | 0.05 | DROP | FAIL+SUSPECT | Persistence 12h, tol=0.05 hPa, DROP suspect segments |
| T12_tol0.05_KEEP | 12 | 0.05 | KEEP | FAIL-only | Persistence 12h, tol=0.05 hPa, KEEP suspect segments |
| T12_tol0.10_DROP | 12 | 0.10 | DROP | FAIL+SUSPECT | Persistence 12h, tol=0.10 hPa, DROP suspect segments |
| T12_tol0.10_KEEP | 12 | 0.10 | KEEP | FAIL-only | Persistence 12h, tol=0.10 hPa, KEEP suspect segments |
| T12_tol0.20_DROP | 12 | 0.20 | DROP | FAIL+SUSPECT | Persistence 12h, tol=0.20 hPa, DROP suspect segments |
| T12_tol0.20_KEEP | 12 | 0.20 | KEEP | FAIL-only | Persistence 12h, tol=0.20 hPa, KEEP suspect segments |
| T24_tol0.05_DROP | 24 | 0.05 | DROP | FAIL+SUSPECT | Persistence 24h, tol=0.05 hPa, DROP suspect segments |
| T24_tol0.05_KEEP | 24 | 0.05 | KEEP | FAIL-only | Persistence 24h, tol=0.05 hPa, KEEP suspect segments |
| T24_tol0.10_DROP | 24 | 0.10 | DROP | FAIL+SUSPECT | Persistence 24h, tol=0.10 hPa, DROP suspect segments |
| T24_tol0.10_KEEP | 24 | 0.10 | KEEP | FAIL-only | Persistence 24h, tol=0.10 hPa, KEEP suspect segments |
| T24_tol0.20_DROP | 24 | 0.20 | DROP | FAIL+SUSPECT | Persistence 24h, tol=0.20 hPa, DROP suspect segments |
| T24_tol0.20_KEEP | 24 | 0.20 | KEEP | FAIL-only | Persistence 24h, tol=0.20 hPa, KEEP suspect segments |
| T48_tol0.05_DROP | 48 | 0.05 | DROP | FAIL+SUSPECT | Persistence 48h, tol=0.05 hPa, DROP suspect segments |
| T48_tol0.05_KEEP | 48 | 0.05 | KEEP | FAIL-only | Persistence 48h, tol=0.05 hPa, KEEP suspect segments |
| T48_tol0.10_DROP | 48 | 0.10 | DROP | FAIL+SUSPECT | Persistence 48h, tol=0.10 hPa, DROP suspect segments |
| T48_tol0.10_KEEP | 48 | 0.10 | KEEP | FAIL-only | Persistence 48h, tol=0.10 hPa, KEEP suspect segments |
| T48_tol0.20_DROP | 48 | 0.20 | DROP | FAIL+SUSPECT | Persistence 48h, tol=0.20 hPa, DROP suspect segments |
| T48_tol0.20_KEEP | 48 | 0.20 | KEEP | FAIL-only | Persistence 48h, tol=0.20 hPa, KEEP suspect segments |

## 4) C1-C6 Exact Criteria (Definitions, Thresholds, Splits)

### 4.1 Base per-config constraints used in budget search

For each config (TRAIN, `scope=dirty`):

- `constraint_missing`:
  - `missing_additional <= B`
  - `missing_additional = missing_rate_product - missing_rate_P0`
- `constraint_common`:
  - `|delta_common_p99| <= epsilon` and `|delta_common_p999| <= epsilon`
  - `epsilon = 0.005`
  - `delta_common_pq = (p_q(product_common) - p_q(P0_common)) / p_q(P0_common)`
- `constraint_mnar`:
  - `state_wind_spread <= tau`
  - `tau = 0.10`
  - `state_wind_spread = Q90(removal_rate_by_decile) - Q10(removal_rate_by_decile)` on P0 wind deciles.
- `constraint_benchmark` (C6 gate at config level when enabled):
  - `recall_stuck >= 0.95` and `precision_stuck >= 0.90`.

`compliant = constraint_missing AND constraint_common AND constraint_mnar AND constraint_benchmark`.

### 4.2 Protocol-level C1..C6 used for manuscript submission decision

| Criterion | Formal definition | Exact threshold(s) | Evaluation split/scope |
|---|---|---|---|
| `C1` | Holdout robustness: among 3 station-holdout folds, at least 2 folds satisfy all base constraints (`missing/common/mnar`) at checklist budget. | `sum(constraints_all over fold_0..fold_2) >= 2` | selection on TRAIN (dev), evaluation on TEST dirty holdout |
| `C2v2` | MNAR evidence package with subchecks; pass requires `C2v2.1` and `C2v2.2` (subcheck `C2v2.3` is reported). | see below | TRAIN+TEST dirty, product contrasts P1 vs P2 |
| `C2v2.1` | Spread-ratio evidence. | `spread_ratio_train >= 10` and `spread_ratio_test >= 10` | TRAIN dirty and TEST dirty |
| `C2v2.2` | Missingness-ratio evidence. | `missing_ratio_train >= 10` and `missing_ratio_test >= 10` | TRAIN dirty and TEST dirty |
| `C2v2.3` | Removed-vs-kept evidence. | TRAIN: `p99_gap_rel >= 0.05` OR `spread >= 0.01`; TEST: `p99_gap_rel >= 0.02` OR `spread >= 0.01` | TRAIN dirty and TEST dirty |
| `C3` | P3 superiority over P1/P4 with common invariance. | `spread_P3 < spread_P1 and spread_P4`; `missing_P3 < missing_P1 and missing_P4`; and `|delta_common_p99_P3|, |delta_common_p999_P3| <= 0.005` | TEST dirty |
| `C4` | Baseline/tool reproducibility availability check. | existence of required baseline artifacts (`P4`, comparison CSV, and tool artifact or unavailability note) | artifact-level |
| `C5` | Downstream cost-sensitive performance non-inferiority/superiority of P3. | using `model=ridge`, `scope=global`: `score_P3 <= score_P0` and `score_P3 <= score_P1` | TEST global |
| `C6` | Anti-gaming benchmark pass for selected config. | `recall_stuck >= 0.95` and `precision_stuck >= 0.90` | benchmark set (injected events, TRAIN+TEST event design) |

Current run status (`results_v3/checklist_acceptance_v4.json`):
- `C1=true`, `C2v2=true`, `C3=true`, `C4=true`, `C5=true`, `C6=true`
- `submit_status=SUBMIT`
- `selected_config (C6_detail)=T3_tol0.05_KEEP`
- `recall_stuck=1.0`, `precision_stuck=0.9606413994169096`

## 5) Anti-Gaming Benchmark Specification (No raw private data required)

Benchmark family:
- Synthetic stuck-event injection on selected dirty stations, evaluated with same detector family as in config grid.

Protocol (exact):
1. Select `5` dirty stations from selected set (`seed_synthetic=20260218`).
2. For each selected station, each phase (`train`, `test`), each duration (`3,6,12,24` h), inject one event:
   - find contiguous candidate runs with:
     - no large gap (`dt <= 2 * station_median_train_delta_s`),
     - non-null pressure,
     - not already failing hard pressure checks (`pressure_fail_zero=0`, `pressure_fail_out_range_after_norm=0`).
   - enforce event run duration `>= duration_h`.
3. Injection mechanism:
   - replace pressure within event with near-constant value:
     - `pressure_hpa = base_value + jitter`
     - `jitter` pattern in `{ -0.005, 0.0, +0.005 }` hPa cycling by index.
4. Precision denominator construction:
   - add a `±1h` temporal context around each injected event to benchmark rows.
5. Ground truth:
   - `stuck_injected = 1` only on injected rows; context rows are negatives.
6. Metrics:
   - `tp, fn, fp` from predicted stuck mask vs `stuck_injected`.
   - `recall_stuck = tp / (tp + fn)`; `precision_stuck = tp / (tp + fp)`.
7. C6 thresholds:
   - `recall_stuck >= 0.95`
   - `precision_stuck >= 0.90`

Current run benchmark manifest highlights:
- `results_v3/benchmark_stuck/benchmark_manifest.json`
  - `selected_dirty_stations`: `c25m059e03,c26m073e01,c27m096e08,c34m015e03,c34m113e04`
  - `events_n = 40` (`5 stations x 2 phases x 4 durations`)
  - `injection_tol_hpa = 0.05`
- `results_v3/benchmark_stuck/benchmark_labels.parquet`
  - benchmark rows: `3094`
  - GT positives (`stuck_injected=1`): `2636`

## 6) Artifact Inventory (Names + One-line content)

### 6.1 Core protocol and selection

| Artifact | Content |
|---|---|
| `results_v3/preregister_v3.json` | Frozen protocol constants, grids, seeds, formulas, thresholds. |
| `results_v3/selection_report.json` | Station selection diagnostics and anti-concentration checks. |
| `results_v3/selected_stations.csv` | Final 30 selected stations with stratum labels. |
| `results_v3/case_stations.csv` | Case-study station subset (if available). |
| `results_v3/P0_raw_canon.parquet` | Canonicalized raw dataset (post dedup and unit normalization). |
| `results_v3/flags_base.parquet` | Base QC flags per row (hard/soft-related fields). |

### 6.2 Grid, constraints, and decisions

| Artifact | Content |
|---|---|
| `results_v3/budget_search_grid_v3.csv` | Full config evaluation grid across budgets/folds with constraints and utilities. |
| `results_v3/budget_holdout_summary_v3.csv` | Holdout-fold validation summaries for selected configs. |
| `results_v3/budget_best_B05_v3.json` | Best compliant config for `B=0.05`. |
| `results_v3/budget_best_B10_v3.json` | Best compliant config for `B=0.10`. |
| `results_v3/budget_top3_stability.csv` | Top-3 utility stability under each budget. |
| `results_v3/checklist_acceptance_v4.json` | Final C1..C6 pass/fail and details. |
| `results_v3/decision_summary_v4.md` | Human-readable decision summary. |

### 6.3 Metrics

| Artifact | Content |
|---|---|
| `results_v3/metrics_missingness.csv` | Missingness metrics per product/scope/phase. |
| `results_v3/metrics_tails.csv` | Tail distortion metrics (`p99`, `p999`, common vs operational). |
| `results_v3/metrics_mnar.csv` | MNAR spread summaries by product/scope/phase. |
| `results_v3/metrics_removed_vs_kept.csv` | Removed-vs-kept diagnostics (`p99` gaps etc.). |
| `results_v3/metrics_removal_by_decile.csv` | Removal rate by wind decile (state dependence). |
| `results_v3/metrics_suspect_burden.csv` | Suspect burden summaries. |
| `results_v3/metrics_topk.csv` | Top-k overlap/coherence metrics. |
| `results_v3/pareto_drop_rules.csv` | Primary-rule Pareto decomposition of removals. |
| `results_v3/metrics_station_heterogeneity.csv` | Station-level heterogeneity metrics (dirty/test). |
| `results_v3/metrics_p1_vs_p4_comparison.csv` | Direct comparison P1 vs P4 behavior. |
| `results_v3/metrics_p3_vs_p5_tool.csv` | P3 vs tool baseline status/metric summary. |

### 6.4 Benchmark, downstream, synthetic

| Artifact | Content |
|---|---|
| `results_v3/benchmark_stuck/benchmark_manifest.json` | Injection protocol instance (chosen stations/events). |
| `results_v3/benchmark_stuck/benchmark_labels.parquet` | Benchmark labels (`stuck_injected`, event metadata). |
| `results_v3/benchmark_stuck/benchmark_perf_by_config.csv` | Recall/precision/tp/fp/fn for all 30 configs. |
| `results_v3/benchmark_stuck/benchmark_perf_selected_config.json` | Benchmark metrics for selected P3 config. |
| `results_v3/downstream/downstream_skill_vs_coverage_v3.csv` | Downstream RMSE/coverage metrics. |
| `results_v3/downstream/downstream_cost_sensitive_v3.csv` | Cost-sensitive downstream score table. |
| `results_v3/downstream/downstream_tail_metrics_v3.csv` | Tail RMSE downstream metrics. |
| `results_v3/synthetic/S1_data.parquet` | Synthetic stress-test product S1 data. |
| `results_v3/synthetic/synthetic_metrics.csv` | Synthetic MNAR and tail-shift metrics. |
| `results_v3/synthetic/synthetic_tailshape_metrics.csv` | Synthetic exceedance distortion metrics (u95/u99). |

### 6.5 Figures

| Artifact | Content |
|---|---|
| `results_v3/figures/F1_pareto.pdf` | Primary-rule removal Pareto. |
| `results_v3/figures/F2_common_vs_operational_tails.pdf` | Common vs operational tail deltas. |
| `results_v3/figures/F3_mnar_spread_deciles.pdf` | MNAR spread, deciles, removed-vs-kept. |
| `results_v3/figures/F4_tradeoff.pdf` | Train dirty trade-off and feasible region. |
| `results_v3/figures/F5_station_heterogeneity.pdf` | Station heterogeneity (dirty/test). |
| `results_v3/figures/F6_downstream_cost_tail.pdf` | Downstream cost-sensitive and tail performance. |
| `results_v3/figures/F7_synthetic_mnar_vs_tailshift.pdf` | Synthetic MNAR/tail-shape stress test. |
| `results_v3/figures/F8_benchmark_detector_power.pdf` | Benchmark detector power heatmap. |

## 7) Minimum Regeneration Command

From repository root:

```bash
python "paper 2/src/06_run_qc_with_guarantees_v3.py" \
  --input "avamet_all.parquet" \
  --workdir "." \
  --budgets "0.05,0.10" \
  --c6-enabled true \
  --selected-budget 0.10
```

Optional figure redraw (PNG + PDF versions with final styling):

```bash
python "paper 2/src/08_redraw_figures_v4_png.py"
```

## 8) Optional: gamma Sensitivity (mini-table)

Computed from `results_v3/downstream/downstream_cost_sensitive_v3.csv` by recomputing:
- `score_gamma = rmse_operational + gamma*(1 - coverage_operational)`

### 8.1 Ridge model (used in C5)

| gamma | Best product | Ranking (best -> worst) |
|---:|---|---|
| 0.0 | P2 | P2 > P3 > P1 > P4 > P0 |
| 1.0 | P2 | P2 > P3 > P1 > P4 > P0 |
| 2.9680 | P2 | P2 > P3 > P1 > P4 > P0 |
| 5.0 | P2 | P2 > P3 > P1 > P4 > P0 |

### 8.2 Persistence model (reference)

| gamma | Best product | Ranking (best -> worst) |
|---:|---|---|
| 0.0 | P2 | P2 > P3 > P1 > P4 > P0 |
| 1.0 | P2 | P2 > P3 > P0 > P1 > P4 |
| 2.9680 | P0 | P0 > P2 > P3 > P1 > P4 |
| 5.0 | P0 | P0 > P2 > P3 > P1 > P4 |

## 9) Code Anchors (for reviewer audit)

- Constants and grid: `paper 2/src/06_run_qc_with_guarantees_v3.py:25`
- Pre-registered payload writing: `paper 2/src/06_run_qc_with_guarantees_v3.py:103`
- Canonical flags/rules creation: `paper 2/src/06_run_qc_with_guarantees_v3.py:273`
- Stuck detector definition: `paper 2/src/06_run_qc_with_guarantees_v3.py:625`
- Benchmark generation: `paper 2/src/06_run_qc_with_guarantees_v3.py:678`
- Config grid and IDs: `paper 2/src/06_run_qc_with_guarantees_v3.py:1173`
- Per-config constraints: `paper 2/src/06_run_qc_with_guarantees_v3.py:1272`
- C1..C6 checklist construction: `paper 2/src/06_run_qc_with_guarantees_v3.py:2802`
- Required artifact validation list: `paper 2/src/06_run_qc_with_guarantees_v3.py:2330`

