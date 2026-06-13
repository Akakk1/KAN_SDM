#!/usr/bin/env python3
"""GBIF batch download via occurrence/search API with auto-retry"""
import requests
import json, csv, os, time, subprocess, sys
from config import DATA_DIR

TAXON_KEY = 2687885
OUTDIR = DATA_DIR

# China scope (strict administrative boundaries, including HK, MO, TW)
# Use a generous bbox covering China + Taiwan for initial download,
# then apply strict polygon filter in R (blockCV + sf) for final points.
CHINA_BBOX = "73,18,135,54"  # minLon,minLat,maxLon,maxLat (covers mainland + Taiwan)

# GBIF occurrence/search API — batches with pagination and retry

LIMIT = 300
MAX_PAGES = 300  # safety cap

all_records = []
failed_pages = 0

for page_idx in range(MAX_PAGES):
    offset = page_idx * LIMIT
    url = f"https://api.gbif.org/v1/occurrence/search?taxonKey={TAXON_KEY}&limit={LIMIT}&offset={offset}&decimalLongitude=73,135&decimalLatitude=18,54"
    
    for attempt in range(3):
        r = subprocess.run(["curl", "-s", "--max-time", "30", url], capture_output=True, text=True)
        if r.stdout.strip():
            try:
                data = json.loads(r.stdout)
                results = data.get("results", [])
                if results:
                    all_records.extend(results)
                    break
            except json.JSONDecodeError:
                pass
        time.sleep(1 + attempt)  # incremental delay
    else:
        failed_pages += 1
    
    if page_idx % 20 == 0:
        print(f"  Progress: {len(all_records):,} records / offset={offset}", flush=True)
    
    if len(results) < LIMIT:
        break  # last page
    
    time.sleep(0.5)  # GBIF API friendly interval

print(f"\n=== Download complete ===")
print(f"Total downloaded: {len(all_records):,} records")
print(f"Failed pages: {failed_pages}")

# ====== Cleaning ======
clean = []
bad = {"no_coordinates": 0, "bad_coordinates": 0, "null_island": 0}

for r in all_records:
    lat = r.get("decimalLatitude")
    lon = r.get("decimalLongitude")
    
    if lat is None or lon is None:
        bad["no_coordinates"] += 1
        continue
    if abs(lat) > 90 or abs(lon) > 180:
        bad["bad_coordinates"] += 1
        continue
    if lat == 0.0 and lon == 0.0:
        bad["null_island"] += 1
        continue
    
    clean.append({
        "gbif_id": r.get("key"),
        "species": "Ginkgo biloba",
        "decimalLatitude": lat,
        "decimalLongitude": lon,
        "year": r.get("year", ""),
        "country": r.get("country", ""),
        "basisOfRecord": r.get("basisOfRecord", ""),
    })

print(f"Valid records: {len(clean):,}  (filtered: {bad})")

# ====== Spatial thinning ======
grid_size = 0.15  # ~16.7 km, matching 10 arc-min
grid = set()
thinned = []

for r in clean:
    gx = int(r["decimalLongitude"] / grid_size)
    gy = int(r["decimalLatitude"] / grid_size)
    key = (gx, gy)
    if key not in grid:
        grid.add(key)
        thinned.append(r)

print(f"After thinning: {len(thinned):,} records")

# ====== Save ======
os.makedirs(OUTDIR, exist_ok=True)
outfile = os.path.join(OUTDIR, "ginkgo_occurrence_raw.csv")
with open(outfile, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["gbif_id", "species", "decimalLatitude", "decimalLongitude", "year", "country", "basisOfRecord"])
    w.writeheader()
    w.writerows(clean)

outfile2 = os.path.join(OUTDIR, "ginkgo_occurrence.csv")
with open(outfile2, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["gbif_id", "species", "decimalLatitude", "decimalLongitude", "year", "country", "basisOfRecord"])
    w.writeheader()
    w.writerows(thinned)

print(f"✅ Saved: {outfile2}")
print(f"   Raw records: {len(clean):,}")
print(f"   After thinning: {len(thinned):,}")
