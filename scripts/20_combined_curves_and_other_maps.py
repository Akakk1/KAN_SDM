#!/usr/bin/env python3
"""

from config import BOUNDARY_GEOJSON, DATA_DIR, RESULTS_DIR, TRAIN_CSV
Combined figure: key response curves (bio13, bio11) + suitability maps for additional baselines (RF, XGB, MLP).

This combines the content of (old) Figure 2 / current Figure 3 curves and S3 other baselines maps into one main figure.

Layout:
- Top row: 2 columns - the two key PDP curves
- Bottom row: 3 columns - the three maps for RF, XGB, MLP

All maps clipped to China boundary, same style as other figures.

Output: manuscript/figures/fig3_curves_and_other_baselines_maps.png

This replaces the previous separate curves and S3 for a more compact presentation.

Difference module remains in the native maps script as requested.
"""

import os
import numpy as np
import pandas as pd
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
DATA_DIR = os.path.join(RESULTS_DIR, "kan_response_data")  # for curves CSV
MAPS_DIR = os.path.join(RESULTS_DIR, "maps")
OUT_DIR = os.path.join(SCRIPT_DIR, "..", "manuscript", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# Curves data (from 10_export)
CURVE_VARS = ['bio13', 'bio11']
VAR_LABELS = {
    'bio13': 'Precipitation of wettest month (mm)',
    'bio11': 'Mean temperature of coldest quarter (°C)',
}
TITLES = {'bio13': 'BIO13', 'bio11': 'BIO11'}
TRIM_EDGE = 2
X_PAD = 0.05

# Maps tifs for other baselines (native, as in S3)
MAP_TIFS = {
    "rf": os.path.join(MAPS_DIR, "rf_current_suitability.tif"),
    "xgb": os.path.join(MAPS_DIR, "xgb_current_suitability.tif"),
    "mlp": os.path.join(MAPS_DIR, "mlp_current_suitability.tif"),
}
MAP_TITLES = {
    "rf": "Random Forest\nAUC 0.8919 ± 0.0143",
    "xgb": "XGBoost\nAUC 0.8822 ± 0.0116",
    "mlp": "MLP\nAUC 0.9047 ± 0.0118",
}

BOUNDARY = BOUNDARY_GEOJSON

def plot_curves_to_axes(ax1, ax2):
    """Plot the two key curves into the provided axes (recreates logic from curves script)."""
    for ax, vn in [(ax1, 'bio13'), (ax2, 'bio11')]:
        d = pd.read_csv(os.path.join(DATA_DIR, f"{vn}_response_pdp.csv"))
        n = len(d)
        d = d.iloc[TRIM_EDGE:n - TRIM_EDGE].copy()
        x = d["var_value"].values
        y = d["prob_pdp"].values

        ax.plot(x, y, '-', color='#2ca02c', lw=2.0)

        x_range = x.max() - x.min()
        ax.set_xlim(x.min() - x_range * X_PAD, x.max() + x_range * X_PAD)
        y_range = y.max() - y.min()
        ax.set_ylim(max(0, y.min() - y_range * X_PAD), y.max() + y_range * X_PAD)

        ax.set_xlabel(VAR_LABELS[vn], fontsize=9)
        ax.set_ylabel('P(presence)', fontsize=9)
        ax.set_title(TITLES[vn], fontweight='bold', fontsize=10)
        ax.grid(alpha=0.2)

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
    print("Loading boundary...")
    gdf = gpd.read_file(BOUNDARY)
    geom = gdf.geometry.iloc[0]

    print("Creating combined figure...")
    # Layout: 2 rows, top 2 cols for curves (wider), bottom 3 cols for maps
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.3], hspace=0.25, wspace=0.2)

    # Top: curves (span or two axes)
    ax_curve1 = fig.add_subplot(gs[0, 0])
    ax_curve2 = fig.add_subplot(gs[0, 1])
    plot_curves_to_axes(ax_curve1, ax_curve2)

    # Bottom: maps
    ax_maps = [fig.add_subplot(gs[1, i]) for i in range(3)]

    print("Loading and plotting maps...")
    map_names = list(MAP_TIFS.keys())
    for i, (name, ax) in enumerate(zip(map_names, ax_maps)):
        arr, ext = load_and_clip(MAP_TIFS[name], geom)
        im = ax.imshow(arr, extent=ext, cmap='YlGnBu', vmin=0, vmax=1, origin='upper')
        ax.set_title(MAP_TITLES[name], fontsize=9)
        ax.set_xlabel("Longitude", fontsize=7)
        ax.set_ylabel("Latitude", fontsize=7)
        ax.set_aspect('equal')
        gdf.boundary.plot(ax=ax, color='black', linewidth=0.5, alpha=0.8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Suitability (0-1)')

    plt.suptitle("Figure 3. KAN key response curves and suitability maps for additional baselines\n(all maps clipped to strict Chinese administrative boundary incl. HK/Macao/Taiwan)", 
                 fontsize=11, fontweight='bold', y=0.98)

    outpath = os.path.join(OUT_DIR, "fig3_curves_and_other_baselines_maps.png")
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Combined figure saved → {outpath}")

if __name__ == "__main__":
    main()