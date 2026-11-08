# Workflow guide

This guide follows the order of the manuscript. Commands assume the repository root.

## Terms

| Term | Meaning |
|---|---|
| Quality control (QC) | Checks that look for faulty observations |
| Flag | A QC result attached to a record: FAIL or SUSPECT |
| FAIL | A value that cannot be correct, such as zero pressure |
| SUSPECT | A pattern that may be a fault, such as stuck pressure |
| Deletion | The value is removed |
| Quarantine | The value is withheld from automatic use, and kept with its evidence for review |
| Maintenance alert | The data are unchanged, and the operator is told that a sensor needs attention |
| Reference | A neighbouring station used for comparison |
| Effective redundancy depth, `m` | References that really have a value at a given moment |
| Hard-invalid | A dead sensor or sentinel value, such as 0 hPa |
| Guard | Excludes hard-invalid references and abstains below the quorum |
| Reference-health screen | Removes a reference far from the other references |
| False isolation | A healthy station quarantined because of failed references |
| Design and evaluation periods | 2023-2024 and 2025; called TRAIN and TEST in the code |

## Names in the code

The manuscript uses descriptive names. Some files keep internal names.

| Manuscript | Code |
|---|---|
| Raw | P0 |
| Delete-all | P1, Delete-all-flags |
| Base | P3, Constraint-selected P3 |
| Pressure quarantine | P4 in scripts 11, 12 and 13 |
| Wind quarantine | P5 in scripts 11, 12 and 13 |
| Wind deletion | P6 in scripts 11, 12 and 13 |
| Low-, medium- and high-flag groups | clean, medium, dirty |

In the main pipeline, product P2 is a FAIL-only reference, P4 deletes FAIL and SUSPECT values, and P5 is the fixed domain-rule baseline. These are different from P4 to P6 in the spatial scripts.

## Step 1. Data and local rules

```bash
python scripts/run_experiment.py --input avamet_all.parquet --workdir . --redraw-figures
```

Input contract: `schemas/input_schema.json`. Timestamps come from `data ini` inside `payload_json`, and `obs_time` is the fallback. AVAMET time is local civil time at the end of the interval; the column name `timestamp_utc` is historical.

Output: `results_v3/`, with `selected_stations.csv`, `products/P0`, `P1`, `P3`, the selected rule and the acceptance checklist.

## Step 2. Network pool

```bash
export WORK=runs_additional/network
python scripts/21_build_network_pool.py $WORK/pool
python scripts/22_make_neighbour_sets.py $WORK
```

The pool holds stations with valid pressure and gust in at least half of all hours and at most 1% invalid pressure. Script 22 builds references for other `K` values and for distance rings. Use `RESULTS_DIR` and `RAW_PARQUET` if your files are elsewhere.

## Step 3. Neighbour failure

```bash
NB_MODE=pool NB_POOL_DIR=$WORK/pool python scripts/20_neighbor_failure_experiment.py --out-dir runs_additional/neighbor_failure --article-dir $WORK/article --avamet-product results_v3/products/P0/data.parquet
```

Output: false-quarantine rates by failure type, number of failed references and `m`.

## Step 4. Injected faults, archive tests and loss scores

```bash
WORK=$WORK RAWP=avamet_all.parquet bash scripts/run_variant.sh V10_ini_pool14_q3_rh results_v3 pool 14
```

This runs scripts 11, 13 and 12 in order. Script 12 does not mark a rule as best if it fails the retention test. Output for each configuration is in `$WORK/var/NAME`.

Baselines: `scripts/14_saqc_selection_stress.py` (needs `requirements-optional.txt`), `scripts/09_run_ci_policy_comparison.py`, `scripts/10_spatial_extreme_retention.py`.

## Step 5. Sensitivity

```bash
WORK=$WORK RAWP=avamet_all.parquet bash scripts/run_sensitivity.sh results_v3
python scripts/26_sensitivity_summary.py $WORK paper_artifacts/revision_r1
```

This covers seeds, fault size, `K`, `q`, `tau`, distance rings and time tolerance.

## Step 6. Whole network

```bash
python scripts/23_network_analysis.py $WORK $WORK/network
python scripts/24_verify_episodes.py $WORK $WORK/network
python scripts/25_wind_stations.py $WORK $WORK/network
python scripts/27_pressure_release_tests.py $WORK $WORK/network
python scripts/28_pressure_extreme_trace.py $WORK
```

These give decorrelation by distance, injected faults at stations outside the sample, pressure episodes, the wind rule over the network, and release tests on pressure itself. Station-level files are ignored by git, because the candidates they list are unconfirmed.

## Step 7. Second network

See [DWD_OPEN_REPRODUCTION.md](DWD_OPEN_REPRODUCTION.md).

## Adapting the workflow to another network

Provide a table with station, timestamp, pressure, mean wind and gust, plus station coordinates. Then:

1. Check how often stations report, and choose a time tolerance above the common interval.
2. Measure `m` on aligned data before enabling any spatial action.
3. Check that the variable is coherent at the distances between stations.
4. Set `NB_WSCALE=3.6` if wind is in m/s.
5. Prefer quarantine to deletion, so that decisions can be reviewed.

## Interpreting the pressure diagnostics

Script 27 reports pressure-state spread and raw-extreme retention across the network. Script 28 exports full-pool retention and exploratory influence subsets:

```bash
python scripts/28_pressure_extreme_trace.py $WORK $WORK/network
```

The below-1% subset uses flags over the full archive, and excluding its largest high-tail contributor is a post-hoc influence check. Neither changes the release limits or confirms the truth of removed extremes. Main spatial release tables use station-specific TRAIN gust thresholds; auxiliary common-threshold diagnostics use TRAIN values pooled across evaluated stations.
