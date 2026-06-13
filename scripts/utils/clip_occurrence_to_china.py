#!/usr/bin/env python3
"""
Clip GBIF occurrence points to strict Chinese administrative boundary
(including Hong Kong, Macao, and Taiwan as required for contest eligibility).

Uses GADM level-0 geojson for CHN + TWN, unioned with shapely.
No heavy geopandas required — only shapely (now installed) + stdlib json.

Usage:
  python Program/utils/clip_occurrence_to_china.py

Outputs:
  - Data/ginkgo_occurrence_china.csv   (the clipped strict-China presence points)
  - Data/boundaries/china_admin_union.geojson (the polygon used, for reuse e.g. background sampling)
"""
import json
import os
from shapely.geometry import shape, Point, mapping
from shapely.ops import unary_union

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "Data")
BOUNDARIES_DIR = os.path.join(DATA_DIR, "boundaries")
INPUT_CSV = os.path.join(DATA_DIR, "ginkgo_occurrence.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "ginkgo_occurrence_china.csv")
UNION_GEOJSON = os.path.join(BOUNDARIES_DIR, "china_admin_union.geojson")

def main():
    print("Loading GADM boundaries...")
    with open(os.path.join(BOUNDARIES_DIR, "gadm41_CHN_0.json")) as f:
        chn = json.load(f)
    with open(os.path.join(BOUNDARIES_DIR, "gadm41_TWN_0.json")) as f:
        twn = json.load(f)

    # GADM json is a FeatureCollection; take the first (only) feature's geometry
    chn_geom = shape(chn["features"][0]["geometry"])
    twn_geom = shape(twn["features"][0]["geometry"])

    # Union to get one geometry representing China (claimed) + Taiwan + HK/MO (inside CHN)
    china_union = unary_union([chn_geom, twn_geom])
    print(f"China admin union geometry ready (type: {china_union.geom_type}, parts: {len(china_union.geoms) if hasattr(china_union, 'geoms') else 1})")

    # Save the union polygon for later reuse (background points, masks, etc.)
    union_fc = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {"name": "China_strict_incl_TW_HK_MO", "source": "GADM4.1 CHN_0 + TWN_0 union"},
            "geometry": mapping(china_union)
        }]
    }
    with open(UNION_GEOJSON, "w") as f:
        json.dump(union_fc, f)
    print(f"Saved union polygon to {UNION_GEOJSON}")

    # Load occurrence
    import pandas as pd
    df = pd.read_csv(INPUT_CSV)
    print(f"\nOriginal points: {len(df)}")

    # Clip
    kept = []
    for _, row in df.iterrows():
        pt = Point(row["decimalLongitude"], row["decimalLatitude"])
        if pt.within(china_union):
            kept.append(row)

    df_china = pd.DataFrame(kept)
    print(f"Kept after strict China clip (incl. TW/HK/MO): {len(df_china)}")
    print(f"Removed: {len(df) - len(df_china)}")

    if "country" in df.columns:
        removed = df[~df.index.isin(df_china.index)]
        print("\nRemoved by country (top):")
        print(removed["country"].value_counts().head(10))

    # Save
    df_china.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved strict China occurrence to {OUTPUT_CSV}")

    # Quick sanity: show country breakdown in the kept data
    if "country" in df_china.columns:
        print("\nKept points by country field (should be almost only China + HK + Taiwan/Chinese Taipei):")
        print(df_china["country"].value_counts())

if __name__ == "__main__":
    main()
