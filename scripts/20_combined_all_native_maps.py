#!/usr/bin/env python3
"""

from config import BOUNDARY_GEOJSON, RESULTS_DIR
Combined Figure 4: All native suitability maps for the 5 models
(KAN, MaxEnt, RF, XGB, MLP).

This stitches the content of (the 1x2 native KAN/MaxEnt) and S3 (RF/XGB/MLP)
into one main figure (5 panels).

All clipped to China boundary (incl. HK/Macao/Taiwan), same style.

Difference module is kept separately in the native maps script.

Output: manuscript/figures/fig4_all_models_suitability_maps.png
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

# All 5 native tifs (KAN and MaxEnt from the native generator, others from other_baselines)
TIFS = {
    "kan":    os.path.join(MAPS_DIR, "kan_current_suitability.tif"),
    "maxent": os.path.join(MAPS_DIR, "maxent_current_suitability.tif"),
    "rf":     os.path.join(MAPS_DIR, "rf_current_suitability.tif"),
    "xgb":    os.path.join(MAPS_DIR, "xgb_current_suitability.tif"),
    "mlp":    os.path.join(MAPS_DIR, "mlp_current_suitability.tif"),
}

LABELS = {
    "kan":    "KAN (width=[10,20,10,1], 300 steps)\ndata 0–0.16, color scale 0–1",
    "maxent": "R maxnet (regmult=1.0, cloglog)",
    "rf":     "Random Forest (n_est=200, max_depth=10)\nAUC 0.8919 ± 0.0143",
    "xgb":    "XGBoost (n_est=200, max_depth=6, lr=0.1)\nAUC 0.8822 ± 0.0116",
    "mlp":    "MLP (hidden=[32,16], Adam lr=0.01)\nAUC 0.9047 ± 0.0118",
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
        extent = (left, left + xres*w, top + yres*h, top)
        arr = np.ma.masked_invalid(arr)
        return arr, extent

def main():
    print("Loading China boundary for clipping...")
    gdf = gpd.read_file(BOUNDARY)
    geom = gdf.geometry.iloc[0]

    print("Loading + clipping the 5 tifs...")
    data = {}
    for name, tif in TIFS.items():
        arr, ext = load_and_clip(tif, geom)
        data[name] = (arr, ext)
        print(f"  {name}: shape={arr.shape}, range=[{np.ma.min(arr):.4f}, {np.ma.max(arr):.4f}]")

    # 2x3 layout: top 2 (MaxEnt + KAN), bottom 3 (RF + XGB + MLP)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # Top row: MaxEnt (left), KAN (center), hide right
    top_names = ['maxent', 'kan']
    for i, name in enumerate(top_names):
        arr, ext = data[name]
        im = axes[0, i].imshow(arr, extent=ext, cmap='YlGnBu', vmin=0, vmax=1, origin='upper')
        axes[0, i].set_title(LABELS[name], fontsize=9)
        axes[0, i].set_xlabel("Longitude", fontsize=7)
        axes[0, i].set_ylabel("Latitude", fontsize=7)
        axes[0, i].set_aspect('equal', adjustable='box')
        gdf.boundary.plot(ax=axes[0, i], color='black', linewidth=0.4, alpha=0.7)
        plt.colorbar(im, ax=axes[0, i], fraction=0.046, pad=0.04, label='Suitability (0-1)')

    # Hide top-right
    axes[0, 2].axis('off')

    # Bottom row: RF, XGB, MLP
    bottom_names = ['rf', 'xgb', 'mlp']
    for i, name in enumerate(bottom_names):
        arr, ext = data[name]
        im = axes[1, i].imshow(arr, extent=ext, cmap='YlGnBu', vmin=0, vmax=1, origin='upper')
        axes[1, i].set_title(LABELS[name], fontsize=9)
        axes[1, i].set_xlabel("Longitude", fontsize=7)
        axes[1, i].set_ylabel("Latitude", fontsize=7)
        axes[1, i].set_aspect('equal', adjustable='box')
        gdf.boundary.plot(ax=axes[1, i], color='black', linewidth=0.4, alpha=0.7)
        plt.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04, label='Suitability (0-1)')

    plt.suptitle("Figure 4. Predicted habitat suitability maps for all models (native outputs)\n(all clipped to strict Chinese administrative boundary incl. HK/Macao/Taiwan)", 
                 fontsize=11, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out = os.path.join(OUT_DIR, "fig4_all_models_suitability_maps.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Combined Figure 4 saved → {out}")

    print("Done. The previous 1x2 (KAN/MaxEnt) and 1x3 (others) are superseded by this combined view for main text.")
    print("Individuals and the old combined S3 remain available if needed.")

if __name__ == "__main__":
    main()