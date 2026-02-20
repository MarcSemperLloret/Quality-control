# Paper 2 Reproducibility Package (QC with Guarantees v3)

This repository is a clean package to reproduce the Paper 2 pipeline outputs.

Main pipeline:
- `src/06_run_qc_with_guarantees_v3.py`

Figure redraw script:
- `src/08_redraw_figures_v4_png.py`

Supplementary audit appendix:
- `docs/appendix_A_supplementary_audit_spec_v3.md`

## Data Access

The original AVAMET raw dataset is licensed and is not redistributed here.

- Expected real input filename: `avamet_all.parquet`
- Input contract and schema: `schemas/input_schema.json`
- Validator: `scripts/validate_input.py`
- Public toy sample (schema/CI only): `data/sample_public.parquet`

## Quickstart (real licensed data)

1. Create environment (Python `3.13.2` tested).
2. Install dependencies.
3. Validate input contract.
4. Run full pipeline.
5. Optionally redraw figures.
6. Verify outputs.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python scripts/validate_input.py --input avamet_all.parquet
python scripts/run_experiment.py --input avamet_all.parquet --workdir . --redraw-figures
python scripts/verify_outputs.py --results-dir results_v3
```

Windows PowerShell activation:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Expected Outputs

The pipeline writes to `results_v3/`, including:
- config/grid artifacts (`budget_search_grid_v3.csv`, `budget_best_B*.json`)
- acceptance checklist (`checklist_acceptance_v4.json`)
- benchmark artifacts (`benchmark_stuck/*`)
- downstream/synthetic metrics (`downstream/*`, `synthetic/*`)
- final figures (`results_v3/figures/F1..F8`).

## Reproducibility and Determinism

- Seeds are fixed in `protocol/preregister_v3.json`.
- CLI supports deterministic budget and C6 flags via `scripts/run_experiment.py`.
- Known non-determinism sources: BLAS/linear algebra backend implementation details across platforms may introduce tiny floating point differences.

## Machine-readable Protocol + Configs

- Full grid manifest (30 configs): `configs/manifest.csv`
- Frozen protocol constants: `protocol/preregister_v3.json`
- Input schema: `schemas/input_schema.json`

## Paper Artifact Verification

- Reference checks (selected metrics): `paper_artifacts/expected_reference.json`
- Optional hashes of generated manuscript artifacts: `paper_artifacts/checksums.txt`

## Citation and License

- Citation metadata: `CITATION.cff`
- License: `LICENSE` (MIT)

## Release/DOI

When publishing:
1. Tag a release (e.g. `v1.0.0`).
2. Archive in Zenodo.
3. Add DOI + tag + commit hash to manuscript and supplement.

