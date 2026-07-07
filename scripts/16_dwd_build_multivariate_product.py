"""Build a canonical multivariate DWD 10-minute product for the open-network demo.

Downloads DWD CDC 10-minute observations for a compact cluster of German
stations that carry pressure (air_temperature product: PP_10), mean wind and
direction (wind product: FF_10, DD_10), and wind gust (extreme_wind product:
FX_10). Produces a canonical parquet with the SAME schema as the AVAMET P0
product so the existing action-verification scripts (11/13/15) can run on it.

Output columns: station_id, timestamp_utc, stratum, pressure_hpa, wind_mean,
wind_gust, wind_dir, temp_c, qn_pressure, qn_wind. DWD 10-minute MESS_DATUM is
in UTC. Missing sentinel -999 is mapped to NaN.

All data are public (https://opendata.dwd.de). Raw zips are cached under
data/dwd_10min/raw so re-runs are offline.
"""

from __future__ import annotations

import argparse
import io
import math
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

CDC = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes"

PRODUCTS = {
    # key: (subdir, recent_prefix, historical_prefix)
    "tu": ("air_temperature", "10minutenwerte_TU_", "10minutenwerte_TU_"),
    "wind": ("wind", "10minutenwerte_wind_", "10minutenwerte_wind_"),
    "fx": ("extreme_wind", "10minutenwerte_extrema_wind_", "10minutenwerte_extrema_wind_"),
}

# Canonical column <- DWD column
COLMAP = {
    "pressure_hpa": "PP_10",
    "temp_c": "TT_10",
    "wind_mean": "FF_10",
    "wind_dir": "DD_10",
    "wind_gust": "FX_10",
}


def http_get(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "research-qc/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def list_dir(url: str) -> list[str]:
    html = http_get(url).decode("latin-1", errors="ignore")
    import re

    return re.findall(r'href="([^"?][^"]*\.zip)"', html)


def parse_station_meta(path: Path) -> dict[str, tuple[int, int, float, float, float]]:
    rows: dict[str, tuple[int, int, float, float, float]] = {}
    lines = path.read_text(encoding="latin-1").splitlines()
    for ln in lines[2:]:
        if not ln.strip():
            continue
        try:
            sid = ln[:5].strip()
            von = int(ln[6:14])
            bis = int(ln[15:23])
            alt = float(ln[24:38])
            lat = float(ln[38:50])
            lon = float(ln[50:60])
        except ValueError:
            continue
        rows[sid] = (von, bis, alt, lat, lon)
    return rows


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def select_cluster(tu_meta, ff_meta, n_stations, win_start, win_end) -> pd.DataFrame:
    common = [
        s
        for s in tu_meta
        if s in ff_meta
        and tu_meta[s][0] <= win_start
        and tu_meta[s][1] >= win_end
        and ff_meta[s][0] <= win_start
        and ff_meta[s][1] >= win_end
    ]
    coords = {s: (tu_meta[s][3], tu_meta[s][4], tu_meta[s][2]) for s in common}
    # densest center: station with most neighbours within 60 km
    best, best_cnt = None, -1
    for s in common:
        la, lo, _ = coords[s]
        cnt = sum(1 for t in common if t != s and haversine_km(la, lo, *coords[t][:2]) <= 60.0)
        if cnt > best_cnt:
            best, best_cnt = s, cnt
    cla, clo, _ = coords[best]
    ranked = sorted(common, key=lambda s: haversine_km(cla, clo, coords[s][0], coords[s][1]))
    chosen = ranked[:n_stations]
    return pd.DataFrame(
        [{"station_id": s, "lat": coords[s][0], "lon": coords[s][1], "alt": coords[s][2]} for s in chosen]
    )


def download_product_files(product_key: str, station: str, cache: Path) -> list[Path]:
    subdir, rec_pre, hist_pre = PRODUCTS[product_key]
    out: list[Path] = []
    # recent
    rec_url = f"{CDC}/{subdir}/recent/{rec_pre}{station}_akt.zip"
    rec_path = cache / f"{product_key}_{station}_akt.zip"
    if not rec_path.exists():
        try:
            rec_path.write_bytes(http_get(rec_url))
        except Exception:
            rec_path = None  # type: ignore
    if rec_path and rec_path.exists():
        out.append(rec_path)
    # historical: list dir, match this station
    try:
        hist_files = [f for f in list_dir(f"{CDC}/{subdir}/historical/") if f"_{station}_" in f]
    except Exception:
        hist_files = []
    for f in hist_files:
        hp = cache / f"{product_key}_{f}"
        if not hp.exists():
            try:
                hp.write_bytes(http_get(f"{CDC}/{subdir}/historical/{f}"))
            except Exception:
                continue
        out.append(hp)
    return out


def read_produkt_zip(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        name = next((n for n in z.namelist() if n.startswith("produkt_")), None)
        if name is None:
            return pd.DataFrame()
        raw = z.read(name).decode("latin-1")
    df = pd.read_csv(io.StringIO(raw), sep=";", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    return df


def extract_canonical(frames: list[pd.DataFrame], product_key: str) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["MESS_DATUM"] = df["MESS_DATUM"].str.strip()
    keep = {"MESS_DATUM": "MESS_DATUM"}
    if "QN" in df.columns:
        keep["QN"] = "qn"
    for canon, dwd in COLMAP.items():
        if dwd in df.columns:
            keep[dwd] = canon
    sub = df[list(keep)].rename(columns=keep)
    for c in sub.columns:
        if c == "MESS_DATUM":
            continue
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
        sub.loc[sub[c] <= -999, c] = np.nan
    sub = sub.drop_duplicates("MESS_DATUM", keep="last")
    return sub


def build_station(station: str, cache: Path, win_start, win_end) -> pd.DataFrame:
    parts: dict[str, pd.DataFrame] = {}
    qn: dict[str, pd.DataFrame] = {}
    for key in PRODUCTS:
        files = download_product_files(key, station, cache)
        frames = [read_produkt_zip(p) for p in files]
        frames = [f for f in frames if not f.empty]
        canon = extract_canonical(frames, key)
        if canon.empty:
            continue
        if "qn" in canon.columns:
            qn[key] = canon[["MESS_DATUM", "qn"]].rename(columns={"qn": f"qn_{key}"})
            canon = canon.drop(columns=["qn"])
        parts[key] = canon
    if "tu" not in parts:
        return pd.DataFrame()
    merged = parts["tu"]
    for key in ("wind", "fx"):
        if key in parts:
            merged = merged.merge(parts[key], on="MESS_DATUM", how="outer")
    for key in ("tu", "wind"):
        if key in qn:
            merged = merged.merge(qn[key], on="MESS_DATUM", how="left")
    merged["station_id"] = station
    ts = pd.to_datetime(merged["MESS_DATUM"], format="%Y%m%d%H%M", utc=True, errors="coerce")
    merged["timestamp_utc"] = ts
    merged = merged.drop(columns=["MESS_DATUM"])
    win0 = pd.Timestamp(win_start, tz="UTC")
    win1 = pd.Timestamp(win_end, tz="UTC")
    merged = merged[(merged["timestamp_utc"] >= win0) & (merged["timestamp_utc"] < win1)]
    return merged


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-dir", type=Path, default=repo_root / "data" / "dwd_10min" / "meta")
    ap.add_argument("--cache", type=Path, default=repo_root / "data" / "dwd_10min" / "raw")
    ap.add_argument("--out-dir", type=Path, default=repo_root / "runs_additional" / "DWD" / "results_v3" / "products" / "P0")
    ap.add_argument("--n-stations", type=int, default=30)
    ap.add_argument("--win-start", type=str, default="20230101")
    ap.add_argument("--win-end", type=str, default="20260101")
    ap.add_argument("--limit", type=int, default=0, help="if >0, only build this many stations (vertical slice)")
    ap.add_argument("--candidate-pool", type=int, default=70, help="rank depth to search for good-pressure stations")
    ap.add_argument("--min-pressure-cov", type=float, default=0.90, help="require this pressure availability to accept a station")
    args = ap.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tu_meta = parse_station_meta(args.meta_dir / "tu_stations.txt")
    ff_meta = parse_station_meta(args.meta_dir / "ff_stations.txt")
    candidates = select_cluster(tu_meta, ff_meta, args.candidate_pool, int(args.win_start), int(args.win_end))
    if args.limit > 0:
        candidates = candidates.head(args.limit)
    print(f"Searching up to {len(candidates)} candidates for {args.n_stations} stations "
          f"with pressure availability >= {args.min_pressure_cov}")

    win_start = pd.Timestamp(args.win_start)
    win_end = pd.Timestamp(args.win_end)
    win0 = pd.Timestamp(args.win_start, tz="UTC")
    win1 = pd.Timestamp(args.win_end, tz="UTC")
    expected_rows = int((win1 - win0).total_seconds() // 600)  # 10-min slots in window
    all_frames: list[pd.DataFrame] = []
    accepted: list[str] = []
    for _, row in candidates.iterrows():
        if len(accepted) >= args.n_stations and args.limit == 0:
            break
        st = row["station_id"]
        df = build_station(st, args.cache, win_start, win_end)
        if df.empty:
            print(f"  {st}: NO DATA  (skip)")
            continue
        pcov = df["pressure_hpa"].notna().sum() / max(expected_rows, 1)
        gcov = df["wind_gust"].notna().sum() / max(expected_rows, 1) if "wind_gust" in df else 0.0
        if pcov < args.min_pressure_cov and args.limit == 0:
            print(f"  {st}: pressure_cov={pcov:.2f} < {args.min_pressure_cov}  (skip, no barometer)")
            continue
        print(f"  {st}: {len(df):>7d} rows  pressure_cov={pcov:.2f}  gust_cov={gcov:.2f}  [accepted #{len(accepted)+1}]")
        accepted.append(st)
        all_frames.append(df)

    sel = candidates[candidates["station_id"].isin(accepted)].copy()
    product = pd.concat(all_frames, ignore_index=True)
    coords = sel.set_index("station_id")[["lat", "lon", "alt"]]
    product = product.merge(coords, left_on="station_id", right_index=True, how="left")
    # placeholder stratum (assigned later by selection script); keep column for schema parity
    product["stratum"] = "unassigned"
    product = product.sort_values(["station_id", "timestamp_utc"]).reset_index(drop=True)
    out_path = args.out_dir / "data.parquet"
    product.to_parquet(out_path, index=False)
    sel.to_csv(args.out_dir.parent.parent / "dwd_selected_stations.csv", index=False)
    print(f"\nWrote {out_path}  ({len(product)} rows, {product['station_id'].nunique()} stations)")
    print(f"Columns: {list(product.columns)}")


if __name__ == "__main__":
    main()
