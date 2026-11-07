#!/bin/bash
# Sensitivity battery of the manuscript. usage: WORK=/some/dir bash scripts/run_sensitivity.sh RESULTS_DIR
# Needs $WORK/pool from scripts/21_build_network_pool.py and $WORK/pool_* from scripts/22_make_neighbour_sets.py.
R="$(cd "$(dirname "$0")" && pwd)"
B="$1"
export NB_QUORUM=3 NB_REFHEALTH=1
bash "$R/run_variant.sh" V10_ini_pool14_q3_rh "$B" pool 14
for s in 1 2 3 4; do SKIP13=1 NB_SEED=$s bash "$R/run_variant.sh" Bs_seed$s "$B" pool 14; done
for f in 5 8 10 12 15; do SKIP13=1 NB_FAULT_HPA=$f bash "$R/run_variant.sh" Bf_fault$f "$B" pool 14; done
for k in K3 K7 K10; do POOLDIR="$WORK/pool_$k" bash "$R/run_variant.sh" Bk_$k "$B" pool 14; done
for t in 5 12; do NB_TAU=$t bash "$R/run_variant.sh" Bt_tau$t "$B" pool 14; done
for r in R0_10 R10_25 R25_50 R50_100; do POOLDIR="$WORK/pool_$r" bash "$R/run_variant.sh" Br_$r "$B" pool 14; done
for q in 2 4 5; do NB_QUORUM=$q bash "$R/run_variant.sh" Bq_q$q "$B" pool 14; done
for m in 7 20; do bash "$R/run_variant.sh" V_tol$m "$B" pool $m; done
