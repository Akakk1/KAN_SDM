#!/usr/bin/env python3
"""
Step 03b: Clip global background points to strict Chinese administrative boundary
and save with coordinates (fills the pipeline gap between 03_prepare_data.py and blockCV).

Pipeline position:
  1. scripts/01_download_gbif.py              → ginkgo_occurrence.csv
  2. scripts/02_extract_env.py                → ginkgo_envdata.csv
  3. scripts/utils/clip_occurrence_to_china.py → ginkgo_occurrence_china.csv + china_admin_union.geojson
  4. scripts/03_prepare_data.py               → variable screening + ginkgo_training_data.csv
  5. THIS SCRIPT                              → ginkgo_background_china.csv  ← NEW
  6. R blockCV (or utils/generate_spatial_folds.py) → fold assignment
  7. Combine occurrence_china + background_china + folds → ginkgo_training_with_coords.csv

Usage:
  python scripts/utils/clip_background_to_china.py

Requires:
  - KAN_GINKGO_DATA env var set (or data_external/ sibling) pointing to WorldClim tifs
  - utils/clip_occurrence_to_china.py already run (for china_admin_union.geojson)
  - 03_prepare_data.py already run (for variable_screening_report_v1.0.json)

Outputs:
  - Data/ginkgo_background_china.csv
"""
import json, os, sys
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol
from shapely.geometry import shape, Point

# Use the repo's central path configuration
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

DATA_DIR = config.DATA_DIR
HISTORY_DIR = config.HISTORY_DIR
BOUNDARY_GEOJSON = config.BOUNDARY_GEOJSON

# ──────────────────────────────────────────────────────────────────────
# 1. Check prerequisites
# ──────────────────────────────────────────────────────────────────────
if not HISTORY_DIR or not os.path.isdir(HISTORY_DIR):
    print("ERROR: WorldClim data not found.", file=sys.stderr)
    print(f"  HISTORY_DIR resolved to: {HISTORY_DIR or '(empty)'}", file=sys.stderr)
    print("  Set KAN_GINKGO_DATA env var or create data_external/ next to the repo.", file=sys.stderr)
    sys.exit(1)

BIO_DIR = os.path.join(HISTORY_DIR, "wc2.1_10m_bio")
ELEV_DIR = os.path.join(HISTORY_DIR, "wc2.1_10m_elev")
ELEV_FILE = os.path.join(ELEV_DIR, "wc2.1_10m_elev.tif")

for d in [BIO_DIR, ELEV_DIR]:
    if not os.path.isdir(d):
        print(f"ERROR: {d} not found.", file=sys.stderr)
        sys.exit(1)
if not os.path.isfile(ELEV_FILE):
    print(f"ERROR: {ELEV_FILE} not found.", file=sys.stderr)
    sys.exit(1)

# Screening report
SCREENING_REPORT = os.path.join(DATA_DIR, "variable_screening_report_v1.0.json")
if not os.path.exists(SCREENING_REPORT):
    print(f"ERROR: Screening report not found at {SCREENING_REPORT}", file=sys.stderr)
    print("Run 03_prepare_data.py first.", file=sys.stderr)
    sys.exit(1)

# China boundary
if not os.path.exists(BOUNDARY_GEOJSON):
    print(f"ERROR: China union polygon not found at {BOUNDARY_GEOJSON}", file=sys.stderr)
    print("Run utils/clip_occurrence_to_china.py first.", file=sys.stderr)
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────
# 2. Load screened variables & China polygon
# ──────────────────────────────────────────────────────────────────────
with open(SCREENING_REPORT) as f:
    screening = json.load(f)
KEEP = screening["final_vars"]
N_LITERATURE_FORCED = len(screening["literature_forced"])
print(f"Loaded {len(KEEP)} screened variables ({N_LITERATURE_FORCED} literature-forced):")
print(f"  {KEEP}")

with open(BOUNDARY_GEOJSON) as f:
    union_fc = json.load(f)
if union_fc["type"] == "FeatureCollection":
    china_geom = shape(union_fc["features"][0]["geometry"])
else:
    china_geom = shape(union_fc)

minx, miny, maxx, maxy = china_geom.bounds
print(f"China union polygon bound box: lon [{minx:.2f}, {maxx:.2f}], lat [{miny:.2f}, {maxy:.2f}]")

# ──────────────────────────────────────────────────────────────────────
# 3. Generate background points within China
# ──────────────────────────────────────────────────────────────────────
N_TARGET = 10000
N_CANDIDATES = N_TARGET * 10
np.random.seed(42)

print(f"\nGenerating {N_TARGET} background points within China polygon...")

candidates_lon = np.random.uniform(minx, maxx, N_CANDIDATES)
candidates_lat = np.random.uniform(miny, maxy, N_CANDIDATES)

bg_lats, bg_lons = [], []
with rasterio.open(ELEV_FILE) as elev_src:
    nd = elev_src.nodata
    for i in range(len(candidates_lat)):
        r, c = rowcol(elev_src.transform, candidates_lon[i], candidates_lat[i])
        if 0 <= r < elev_src.height and 0 <= c < elev_src.width:
            v = elev_src.read(1)[r, c]
            if v != nd and v > -1e20 and v > -500:
                pt = Point(candidates_lon[i], candidates_lat[i])
                if pt.within(china_geom):
                    bg_lats.append(candidates_lat[i])
                    bg_lons.append(candidates_lon[i])
                    if len(bg_lats) >= N_TARGET:
                        break

print(f"  Generated: {len(bg_lats):,} points")
if len(bg_lats) < N_TARGET:
    print(f"  WARNING: only {len(bg_lats)} points (target {N_TARGET})")

# ──────────────────────────────────────────────────────────────────────
# 4. Extract environmental values
# ──────────────────────────────────────────────────────────────────────
print(f"\nExtracting {len(KEEP)} environmental variables...")

bg_env = {}
for v in KEEP:
    tif = os.path.join(BIO_DIR, f"{v}.tif")
    vals = []
    with rasterio.open(tif) as src:
        nd = src.nodata
        for lon, lat in zip(bg_lons, bg_lats):
            r, c = rowcol(src.transform, lon, lat)
            val = src.read(1)[r, c]
            if np.isnan(val) or val == nd or abs(val) > 1e20:
                vals.append(np.nan)
            else:
                vals.append(float(val))
        bg_env[v] = vals
    valid = sum(1 for vv in vals if not np.isnan(vv))
    print(f"  {v:6s}: {valid}/{len(vals)} valid")

# ──────────────────────────────────────────────────────────────────────
# 5. Build DataFrame, drop NaN rows, save
# ──────────────────────────────────────────────────────────────────────
bg_df = pd.DataFrame(bg_env)
bg_df["decimalLongitude"] = bg_lons[:len(bg_df)]
bg_df["decimalLatitude"] = bg_lats[:len(bg_df)]
bg_df["label"] = 0

n_before = len(bg_df)
bg_df = bg_df.dropna()
n_after = len(bg_df)
print(f"\nNaN drop: {n_before} → {n_after} (dropped {n_before - n_after})")

# Consistency fields
bg_df["species"] = "Ginkgo biloba"
bg_df["year"] = 0
bg_df["country"] = "China"
bg_df["basisOfRecord"] = "BACKGROUND"

# Column order matching old background_china.csv convention
out_cols = (["decimalLongitude", "decimalLatitude", "label"] +
            KEEP +
            ["species", "year", "country", "basisOfRecord"])
out_df = bg_df[out_cols]

outfile = os.path.join(DATA_DIR, "ginkgo_background_china.csv")
out_df.to_csv(outfile, index=False)
print(f"\n✅ Saved: {outfile}")
print(f"   Shape: {out_df.shape}")
print(f"   Lat range: {bg_df['decimalLatitude'].min():.2f} – {bg_df['decimalLatitude'].max():.2f}")
print(f"   Lon range: {bg_df['decimalLongitude'].min():.2f} – {bg_df['decimalLongitude'].max():.2f}")

print(f"\nDone. Next step: assign fold labels via R blockCV or")
print(f"  python scripts/utils/generate_spatial_folds.py")
print(f"Then combine ginkgo_occurrence_china.csv + ginkgo_background_china.csv + folds")
print(f"to create ginkgo_training_with_coords.csv.")
