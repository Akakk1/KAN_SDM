#!/usr/bin/env python3
"""Extract environmental variables from WorldClim tifs for occurrence points"""
import rasterio
import numpy as np
import csv, os, glob
from rasterio.transform import rowcol
from config import DATA_DIR, HISTORY_DIR

# ====== Configuration ======
OCCURRENCE = os.path.join(DATA_DIR, "ginkgo_occurrence.csv")

# Variables: BIO (19) + elevation (1) = 20 total
BIO_DIR = os.path.join(HISTORY_DIR, "wc2.1_10m_bio")
ELEV_FILE = os.path.join(HISTORY_DIR, "wc2.1_10m_elev/wc2.1_10m_elev.tif")

def read_points(csv_path):
    """Read occurrence CSV, return [(lat, lon, meta), ...]"""
    points = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            lat = float(row["decimalLatitude"])
            lon = float(row["decimalLongitude"])
            points.append((lat, lon, row))
    return points

def extract_values(points, tif_path, band=1):
    """Extract values from a single-band tif for all points"""
    with rasterio.open(tif_path) as src:
        values = []
        nodata = src.nodata
        for lat, lon, _ in points:
            # rasterio expects (row, col)
            r, c = rowcol(src.transform, lon, lat)  # note: lon, lat order
            # Bounds check
            if 0 <= r < src.height and 0 <= c < src.width:
                v = src.read(band)[r, c]
                if v == nodata or v < -1e20:
                    values.append(None)
                else:
                    values.append(float(v))
            else:
                values.append(None)
    return values

def extract_multiband(points, tif_path):
    """Extract all bands from a multi-band tif for all points"""
    with rasterio.open(tif_path) as src:
        n_bands = src.count
        all_values = [[] for _ in range(n_bands)]
        nodata = src.nodata if src.nodata is not None else -3.4e38
        for lat, lon, _ in points:
            r, c = rowcol(src.transform, lon, lat)
            if 0 <= r < src.height and 0 <= c < src.width:
                row_data = src.read(window=((r, r+1), (c, c+1)))
                for b in range(n_bands):
                    v = row_data[b, 0, 0]
                    if abs(v) > 1e20 or v < -1e20:
                        all_values[b].append(None)
                    else:
                        all_values[b].append(float(v))
            else:
                for b in range(n_bands):
                    all_values[b].append(None)
    return all_values

# ====== Main ======
print("=== Loading occurrence points ===")
points = read_points(OCCURRENCE)
print(f"Loaded {len(points):,} points")

print("\n=== Extracting BIO variables ===")
bio_files = sorted(glob.glob(f"{BIO_DIR}/bio*.tif"))
# Filter out .ovr overview files
bio_files = [f for f in bio_files if not f.endswith('.ovr')]
print(f"BIO variables: {len(bio_files)}")

bio_data = {}
for bf in bio_files:
    name = os.path.basename(bf).replace('.tif', '')
    vals = extract_values(points, bf)
    bio_data[name] = vals
    valid = sum(1 for v in vals if v is not None)
    print(f"  {name}: {valid}/{len(points)} valid")

print("\n=== Extracting elevation ===")
elev_vals = extract_values(points, ELEV_FILE)
valid = sum(1 for v in elev_vals if v is not None)
print(f"  elev: {valid}/{len(points)} valid")

# ====== Merge and save ======
print("\n=== Saving output ===")
os.makedirs(DATA_DIR, exist_ok=True)
outfile = f"{DATA_DIR}/ginkgo_envdata.csv"

# Fields: original metadata + all environmental variables
with open(outfile, "w", newline="") as f:
    fields = ["gbif_id", "species", "decimalLatitude", "decimalLongitude", 
              "year", "country", "basisOfRecord"] + list(bio_data.keys()) + ["elev"]
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    
    for i, (lat, lon, meta) in enumerate(points):
        row = {k: meta.get(k, "") for k in ["gbif_id", "species", "decimalLatitude", "decimalLongitude", "year", "country", "basisOfRecord"]}
        for name in bio_data:
            row[name] = bio_data[name][i] if bio_data[name][i] is not None else ""
        row["elev"] = elev_vals[i] if elev_vals[i] is not None else ""
        writer.writerow(row)

print(f"✅ Saved: {outfile}")
print(f"   Total rows: {len(points):,}")
print(f"   Variables: {len(bio_data) + 1}")

# Quick summary
print("\n=== Missing value overview ===")
for name in bio_data:
    missing = sum(1 for v in bio_data[name] if v is None)
    if missing > 0:
        print(f"  {name}: {missing} missing")
missing_elev = sum(1 for v in elev_vals if v is None)
if missing_elev > 0:
    print(f"  elev: {missing_elev} missing")
else:
    print("  All variables: no missing values")
