#!/usr/bin/env python3
"""Figure 1: Study area, presence points, and spatial blocking design.

Generates a publication-ready map showing:
- China strict administrative boundary (GADM union incl. HK/MO/TW)
- Ocean background (pale blue) behind the land
- 238 Ginkgo biloba presence points colored by canonical blockCV fold (5 folds)
- No title on the image (detailed caption in Figure Legend)
- Nine-dash line inset (South China Sea, bottom-right corner)
  with colors matching the main map

Data source: ginkgo_training_with_coords.csv (canonical 150 km R blockCV folds)
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys, os

_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(_PROJ_ROOT, 'scripts'))
from utils.nine_dash import add_nine_dash_inset

# --- Paths ---
BOUNDARY_PATH = os.path.join(_PROJ_ROOT, 'data', 'boundaries', 'china_admin_union.geojson')
TRAINING_PATH = os.path.join(_PROJ_ROOT, 'data', 'ginkgo_training_with_coords.csv')
OUTPUT_DIR = os.path.join(_PROJ_ROOT, 'results', 'figures')
OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'fig1_study_area_folds.png')

# --- Colors (used by both main map and inset) ---
LAND_COLOR = '#f0f0f0'
OCEAN_COLOR = '#dce8f0'

# --- Font ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Liberation Serif', 'Times New Roman', 'serif']
plt.rcParams['font.size'] = 11

# ============================================================
# 1. Load China boundary
# ============================================================
with open(BOUNDARY_PATH) as f:
    china_raw = json.load(f)

from shapely.geometry import shape
polys = []
if china_raw['type'] == 'FeatureCollection':
    for feat in china_raw['features']:
        g = shape(feat['geometry'])
        if g.geom_type == 'Polygon':
            polys.append(g)
        elif g.geom_type == 'MultiPolygon':
            polys.extend(list(g.geoms))
else:
    print(f"Unexpected GeoJSON type: {china_raw['type']}")
    sys.exit(1)

# Filter out tiny islands
min_area_deg2 = 0.5
mainland = [p for p in polys if p.area > min_area_deg2]
print(f"China boundary: {len(mainland)} polygons kept from {len(polys)} total")

# ============================================================
# 2. Load presence points with canonical blockCV folds (0-based)
# ============================================================
df = pd.read_csv(TRAINING_PATH)
pres = df[df['label'] == 1].copy()
print(f"Presence points: {len(pres)}")
print(f"Fold distribution: {pres['fold'].value_counts().sort_index().to_dict()}")

# ============================================================
# 3. Plot
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(8, 7))

# -- Map extent (full China view, so the SCS inset fits cleanly in bottom-right) --
ax.set_xlim(72, 138)
ax.set_ylim(16, 55)

# -- Ocean background (pale blue) --
ax.set_facecolor(OCEAN_COLOR)

# -- China land boundary --
for poly in mainland:
    if poly.geom_type == 'Polygon':
        xs, ys = poly.exterior.xy
        ax.fill(xs, ys,
                facecolor=LAND_COLOR, edgecolor='#333333',
                linewidth=0.8, zorder=1)

# -- Fold colors (ColorBrewer Set1, 5-class) --
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

# -- Presence points by fold --
for f in range(5):
    subset = pres[pres['fold'] == f]
    ax.scatter(subset['decimalLongitude'], subset['decimalLatitude'],
               c=colors[f], s=5, edgecolors='white', linewidth=0.2,
               label=f'Fold {f} (n={len(subset)})', zorder=3, alpha=0.85)

# ============================================================
# 4. Legend (bottom-right, above the nine-dash inset)
# ============================================================
ax.legend(loc='lower right', fontsize=8, markerscale=1.5,
          framealpha=0.9, edgecolor='#666666',
          bbox_to_anchor=(0.96, 0.30))

# ============================================================
# 5. Labels (no title — caption is in Figure Legend)
# ============================================================
ax.set_xlabel('Longitude (°E)', fontsize=11)
ax.set_ylabel('Latitude (°N)', fontsize=11)
# Correct aspect for mid-latitude (~35°N): 1° lat / 1° lon = 1/cos(35°)
ax.set_aspect(1.0 / np.cos(np.radians(35)))

# ============================================================
# 6. Nine-dash line inset (colors match main map)
# ============================================================
# Nine-dash line inset (inside main axes, bottom-right corner)
add_nine_dash_inset(ax, box=[0.735, 0.01, 0.26, 0.26],
                    land_color=LAND_COLOR, sea_color=OCEAN_COLOR)

# ============================================================
# 7. Save
# ============================================================
plt.savefig(OUTPUT_PATH, dpi=600, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_PATH}")
