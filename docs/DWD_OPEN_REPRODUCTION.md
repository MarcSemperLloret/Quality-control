# Public DWD Reproduction

This repository separates the licensed AVAMET case from the fully public Deutscher Wetterdienst (DWD) reproduction. AVAMET raw records are not redistributed. The DWD side can be regenerated from public DWD Climate Data Center files and is the public audit path for the cross-network and neighbour-failure claims.

## What Is Included

- DWD station metadata used for the station search: `data/dwd_10min/meta/`.
- DWD reconstruction scripts: `scripts/16_dwd_build_multivariate_product.py`, `scripts/17_dwd_stratify.py`, and `scripts/18_dwd_derive_products.py`.
- Shared evaluation scripts used by the manuscript: `scripts/11_measurement_fault_suite.py`, `scripts/13_spatial_quarantine_archive.py`, `scripts/19_cross_network_table.py`, and `scripts/20_neighbor_failure_experiment.py`.
- Small derived DWD audit tables: `paper_artifacts/dwd_open_demo/`.
- Neighbour-failure summary outputs used in the manuscript: `runs_additional/neighbor_failure/`.

Large downloaded DWD zip files and generated parquet products are not committed. They are regenerated locally under `data/dwd_10min/raw/` and `runs_additional/DWD/`.

## End-to-End DWD Commands

From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python scripts/16_dwd_build_multivariate_product.py
python scripts/17_dwd_stratify.py
python scripts/18_dwd_derive_products.py

python scripts/11_measurement_fault_suite.py --results-dir runs_additional/DWD/results_v3 --article-dir runs_additional/DWD/article_dwd
python scripts/13_spatial_quarantine_archive.py --results-dir runs_additional/DWD/results_v3 --article-dir runs_additional/DWD/article_dwd
python scripts/20_neighbor_failure_experiment.py --out-dir runs_additional/neighbor_failure --article-dir article
python scripts/19_cross_network_table.py
```

The first command downloads public DWD CDC zip files as needed. Re-runs use the local cache.

## Expected Public Outputs

The key small outputs are:

- `runs_additional/DWD/results_v3/products/P0/data.parquet`
- `runs_additional/DWD/results_v3/products/P1/data.parquet`
- `runs_additional/DWD/results_v3/products/P3/data.parquet`
- `runs_additional/DWD/results_v3/measurement_fault_suite/measurement_fault_suite_summary.csv`
- `runs_additional/DWD/results_v3/spatial_quarantine_archive/archive_retention.csv`
- `runs_additional/neighbor_failure/neighbor_failure_marginal.csv`
- `runs_additional/neighbor_failure/m_distribution.csv`

Reference copies of the small DWD outputs used for manuscript audit are stored in `paper_artifacts/dwd_open_demo/`. Minor floating-point differences can occur across numerical backends; the qualitative checks are the rule-missed pressure offset, pressure drift, and wind spike/burst families recovered by spatial evidence, and near-zero DWD high-state censoring under the guarded archive action.

## Scope

This public reproduction does not regenerate AVAMET-only tables because AVAMET observations require permission from AVAMET. It regenerates the open-network side of the cross-network result and the DWD half of the neighbour-failure experiment.
