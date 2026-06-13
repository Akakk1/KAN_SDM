#!/usr/bin/env python3
"""

from config import BOUNDARY_GEOJSON, RESULTS_DIR
Plotting script for the NEW Figure 5: Calibrated suitability maps.

This figure shows the post-hoc Platt-scaled versions for KAN and MLP,
together with the native (already well-behaved) maps for MaxEnt, RF, and XGB.

All panels are clipped to the strict Chinese administrative boundary (incl. HK, Macao, Taiwan)
and use a unified 0-1 color scale for direct visual comparison.

Output:
  manuscript/figures/figS4_calibrated_suitability_maps.png

This is intentionally a separate new figure and does not overwrite the native maps figure.
"""

import os
import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import geopandas as gpd

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Liberation Serif', 'Times New Roman', 'serif']
plt.rcParams['font.size'] = 8

# SCRIPT_DIR — path resolution now handled by config.py
MAPS_DIR = os.path.join(RESULTS_DIR, "maps")
OUT_DIR = os.path.join(SCRIPT_DIR, "..", "manuscript", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# Calibrated or native tifs (use calibrated for KAN and MLP)
TIF_FILES = {
    "maxent": os.path.join(MAPS_DIR, "maxent_current_suitability.tif"),
    "rf":     os.path.join(MAPS_DIR, "rf_current_suitability.tif"),
    "xgb":    os.path.join(MAPS_DIR, "xgb_current_suitability.tif"),
    "mlp":    os.path.join(MAPS_DIR, "mlp_calibrated_current_suitability.tif"),
    "kan":    os.path.join(MAPS_DIR, "kan_calibrated_current_suitability.tif"),
}

LABELS = {
    "maxent": "R maxnet (cloglog, native)",
    "rf":     "Random Forest (native)",
    "xgb":    "XGBoost (native)",
    "mlp":    "MLP (Platt calibrated)",
    "kan":    "KAN (Platt calibrated)",
}

BOUNDARY = BOUNDARY_GEOJSON

def load_and_clip(path, geom):
    with rasterio.open(path) as src:
        out_image, out_transform = rio_mask(src, [geom], crop=True, nodata=np.nan, filled=False)
        arr = out_image[0]
        h, w = arr.shape
        left = out_transform[2]
        top = out_transform[5]
        xres = out_transform[0]
        yres = out_transform[4]
        extent = (left, left + xres * w, top + yres * h, top)
        arr = np.ma.masked_invalid(arr)
        return arr, extent

def main():
    print("Loading boundary for clipping...")
    gdf = gpd.read_file(BOUNDARY)
    geom = gdf.geometry.iloc[0]

    print("Loading and clipping all calibrated/native tifs...")
    data = {}
    for key, tif in TIF_FILES.items():
        arr, ext = load_and_clip(tif, geom)
        data[key] = (arr, ext)
        print(f"  {key}: range after clip [{np.ma.min(arr):.4f}, {np.ma.max(arr):.4f}]")

    # 2x3 layout (5 panels + one empty or title space)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes = axes.flatten()

    for idx, (key, (arr, ext)) in enumerate(data.items()):
        ax = axes[idx]
        im = ax.imshow(arr, extent=ext, cmap='YlGnBu', vmin=0, vmax=1, origin='upper')
        ax.set_title(LABELS[key], fontsize=9)
        ax.set_xlabel("Longitude", fontsize=7)
        ax.set_ylabel("Latitude", fontsize=7)
        ax.set_aspect('equal')
        gdf.boundary.plot(ax=ax, color='black', linewidth=0.4, alpha=0.7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label='Calibrated suitability (0-1)')

    # Hide the last empty axis
    axes[5].axis('off')

    plt.suptitle(
        "Supplementary Figure S4. Post-hoc calibrated habitat suitability maps for all models\n"
        "(Platt scaling applied to KAN and MLP; all other models shown with native outputs; "
        "strict Chinese administrative boundary incl. HK/Macao/Taiwan)",
        fontsize=11, fontweight='bold', y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    outpath = os.path.join(OUT_DIR, "figS4_calibrated_suitability_maps.png")
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ New Supplementary Figure S4 saved → {outpath}")

if __name__ == "__main__":
    main()