#!/bin/bash
# Run the fault suite, the archive tests and the loss scores for one configuration.
# usage: WORK=/some/dir bash scripts/run_variant.sh NAME RESULTS_DIR MODE TOLERANCE_MIN
# RESULTS_DIR is a pipeline results folder holding products/P0, P1 and P3.
# Optional: PYTHON, RAWP (parquet with coordinates), POOLDIR, SKIP13, and the NB_* settings
# documented in scripts/neighbour_reference.py.
set -u
R="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python}"
RAW="${RAWP:-$R/avamet_all.parquet}"
V="${WORK:?set WORK to an output folder}/var/$1"
mkdir -p "$V/results_v3/products"
for p in P0 P1 P3; do cp -r "$2/products/$p" "$V/results_v3/products/"; done
cp "$2"/*.csv "$2"/*.json "$V/results_v3/" 2>/dev/null
export NB_MODE="$3" NB_TOL_MIN="$4" NB_POOL_DIR="${POOLDIR:-$WORK/pool}"
"$PY" "$R/scripts/11_measurement_fault_suite.py" --results-dir "$V/results_v3" --raw-parquet "$RAW" --article-dir "$V/article" > "$V/log11.txt" 2>&1 || echo "script 11 failed, see $V/log11.txt"
[ -n "${SKIP13:-}" ] || "$PY" "$R/scripts/13_spatial_quarantine_archive.py" --results-dir "$V/results_v3" --raw-parquet "$RAW" --article-dir "$V/article" > "$V/log13.txt" 2>&1 || echo "script 13 failed, see $V/log13.txt"
[ -n "${SKIP13:-}" ] || "$PY" "$R/scripts/12_operational_risk_analysis.py" --results-dir "$V/results_v3" --article-dir "$V/article" > "$V/log12.txt" 2>&1 || echo "script 12 failed, see $V/log12.txt"
echo "done $1"
