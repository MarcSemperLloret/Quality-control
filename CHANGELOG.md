# Changelog

## ems-revision-r1

Revision of the manuscript after peer review. The reviewed version remains available as the tag `ems-review-2026-07-07`.

### Corrections

- Stations were compared on exact timestamps although they report at different intervals. References are now aligned in time. The share of healthy records with at most two references falls from 11.4% to 0.06%.
- Neighbours came only from the 30-station sample, and in the fault suite only from the ten high-flag stations. References can now come from a filtered network pool.
- Scripts 11 and 13 required two references although the manuscript stated a quorum of three. The quorum and the reference-health screen now apply.
- The wind rule used km/h constants on DWD data in m/s, so it could not fire. Use `NB_WSCALE=3.6`.
- Timestamps now come from `data ini`, because `obs_time` had day and month exchanged in 0.14% of rows. AVAMET time is local civil time, not UTC.
- Script 12 now applies the retention test before marking a rule as best.
- Default paths no longer assume a parent project folder.

### Added

- `scripts/neighbour_reference.py`: alignment, guard and reference-health screen.
- Scripts missing from the reviewed release: 09, 10, 12 (both) and 14.
- Network analyses: scripts 21 to 28, `run_variant.sh` and `run_sensitivity.sh`.
- `examples/quickstart.py`, unit tests, `docs/WORKFLOW.md`.
- Aggregated results in `paper_artifacts/revision_r1/`.

### Not included

Manuscript figure scripts for the new figures depend on a local plotting style. The data behind those figures are in `paper_artifacts/revision_r1/`.
