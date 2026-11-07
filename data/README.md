# Data Contract and Access

## Real dataset (licensed, not redistributed)

This project expects an input parquet file with this name by default:
- `avamet_all.parquet`

You must provide it locally under your license terms.

Expected columns:
- `station_id`
- `obs_time` (fallback timestamp; day and month are exchanged in some rows, so `data ini` is used when present)
- `payload_json` (JSON string with keys):
  - `data ini` (observation time, local civil time at the end of the interval)
  - `pres mit_mit`
  - `vent vel_mit`
  - `vent vel_max`
  - `vent gra_mit`

Validate before running:

```bash
python scripts/validate_input.py --input avamet_all.parquet
```

## Public sample included

Included file:
- `data/sample_public.parquet`

Purpose:
- schema validation,
- CI smoke checks,
- not intended to reproduce manuscript-level metrics.

