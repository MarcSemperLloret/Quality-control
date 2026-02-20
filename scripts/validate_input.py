from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["station_id", "obs_time", "payload_json"]
REQUIRED_PAYLOAD_KEYS = ['pres mit_mit', 'vent vel_mit', 'vent vel_max', 'vent gra_mit']


def _parse_num(raw: object) -> float | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if s == "" or s.lower() == "none":
        return None
    s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate input parquet against the pipeline data contract.")
    parser.add_argument("--input", required=True, help="Path to input parquet.")
    parser.add_argument("--sample-rows", type=int, default=50000, help="Rows to inspect (head).")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"[FAIL] Input file not found: {input_path}")

    df = pd.read_parquet(input_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"[FAIL] Missing required columns: {missing}")

    n_total = int(df.shape[0])
    if n_total == 0:
        raise SystemExit("[FAIL] Input parquet has zero rows.")

    view = df.head(max(1, int(args.sample_rows))).copy()
    n_view = int(view.shape[0])

    ts = pd.to_datetime(view["obs_time"], errors="coerce", utc=True)
    ts_ok = int(ts.notna().sum())
    if ts_ok == 0:
        raise SystemExit("[FAIL] Could not parse any obs_time values as datetime.")

    station_ok = int(view["station_id"].notna().sum())
    if station_ok == 0:
        raise SystemExit("[FAIL] station_id is null in all inspected rows.")

    payload_ok = 0
    payload_missing_keys = 0
    numeric_pressure = 0
    numeric_wind_mean = 0
    numeric_wind_gust = 0
    numeric_wind_dir = 0

    for raw in view["payload_json"]:
        try:
            payload = json.loads(raw) if isinstance(raw, str) else {}
        except Exception:
            payload = {}
        if all(k in payload for k in REQUIRED_PAYLOAD_KEYS):
            payload_ok += 1
        else:
            payload_missing_keys += 1
        if _parse_num(payload.get('pres mit_mit')) is not None:
            numeric_pressure += 1
        if _parse_num(payload.get('vent vel_mit')) is not None:
            numeric_wind_mean += 1
        if _parse_num(payload.get('vent vel_max')) is not None:
            numeric_wind_gust += 1
        if _parse_num(payload.get('vent gra_mit')) is not None:
            numeric_wind_dir += 1

    if payload_ok == 0:
        raise SystemExit("[FAIL] No inspected rows contained all required payload_json keys.")

    print("[OK] Input contract validation passed")
    print(f"- file: {input_path}")
    print(f"- rows_total: {n_total}")
    print(f"- rows_inspected: {n_view}")
    print(f"- parseable_obs_time: {ts_ok}/{n_view}")
    print(f"- non_null_station_id: {station_ok}/{n_view}")
    print(f"- payload_with_all_required_keys: {payload_ok}/{n_view}")
    print(f"- numeric pressure samples: {numeric_pressure}/{n_view}")
    print(f"- numeric wind_mean samples: {numeric_wind_mean}/{n_view}")
    print(f"- numeric wind_gust samples: {numeric_wind_gust}/{n_view}")
    print(f"- numeric wind_dir samples: {numeric_wind_dir}/{n_view}")
    if payload_missing_keys > 0:
        print(f"- warning_missing_payload_keys_rows: {payload_missing_keys}/{n_view}")


if __name__ == "__main__":
    main()

