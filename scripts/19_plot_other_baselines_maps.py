#!/usr/bin/env python3
"""

from config import BOUNDARY_GEOJSON, RESULTS_DIR
Plot suitability maps for the other baselines (RF, XGB, MLP).

Produces a clean 1x3 (or 2x2) figure using the same China clipping + style
as Figure 3.

Output:
  manuscript/figures/figS3_other_baselines_suitability.png

Run under kan_spe (has geopandas, rasterio, matplotlib).
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
plt.rcParams['font.size'] = 9

# SCRIPT_DIR — path resolution now handled by config.py
MAPS_DIR = os.path.join(RESULTS_DIR, "maps")
OUT_DIR = os.path.join(SCRIPT_DIR, "..", "manuscript", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

TIFS = {
    "rf":   os.path.join(MAPS_DIR, "rf_current_suitability.tif"),
    "xgb":  os.path.join(MAPS_DIR, "xgb_current_suitability.tif"),
    "mlp":  os.path.join(MAPS_DIR, "mlp_current_suitability.tif"),
}

# Same boundary as Figure 3
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

    print("Loading + clipping the three tifs...")
    data = {}
    for name, tif in TIFS.items():
        arr, ext = load_and_clip(tif, geom)
        data[name] = (arr, ext)
        print(f"  {name}: shape={arr.shape}, range=[{np.ma.min(arr):.4f}, {np.ma.max(arr):.4f}]")

    # 1x3 figure
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))

    titles = {
        "rf":  "Random Forest (n_est=200, max_depth=10)\nAUC 0.8919 ± 0.0143",
        "xgb": "XGBoost (n_est=200, max_depth=6, lr=0.1)\nAUC 0.8822 ± 0.0116",
        "mlp": "MLP (hidden=[32,16], Adam lr=0.01)\nAUC 0.9047 ± 0.0118",
    }

    for ax, (name, (arr, ext)) in zip(axes, data.items()):
        im = ax.imshow(arr, extent=ext, cmap='YlGnBu', vmin=0, vmax=1, origin='upper')
        ax.set_title(titles[name], fontsize=9)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect('equal', adjustable='box')
        # draw boundary
        gdf.boundary.plot(ax=ax, color='black', linewidth=0.5, alpha=0.8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Suitability (0-1)')

    plt.suptitle("Supplementary Figure S3. Predicted suitability maps for additional baselines\n(all clipped to strict Chinese administrative boundary incl. HK/Macao/Taiwan)", 
                 fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out = os.path.join(OUT_DIR, "figS3_other_baselines_suitability.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved {out}")

    # Save individual subplots for flexible combining (as requested)
    for name, (arr, ext) in data.items():
        fig_i, ax_i = plt.subplots(1, 1, figsize=(5, 4.5))
        im_i = ax_i.imshow(arr, extent=ext, cmap='YlGnBu', vmin=0, vmax=1, origin='upper')
        ax_i.set_title(titles[name], fontsize=9)
        ax_i.set_xlabel("Longitude")
        ax_i.set_ylabel("Latitude")
        ax_i.set_aspect('equal', adjustable='box')
        gdf.boundary.plot(ax=ax_i, color='black', linewidth=0.5, alpha=0.8)
        plt.colorbar(im_i, ax=ax_i, fraction=0.046, pad=0.04, label='Suitability (0-1)')
        plt.tight_layout()
        ind_out = os.path.join(OUT_DIR, f"figS3_{name}_map.png")
        plt.savefig(ind_out, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Individual saved → {ind_out}")

    # Also a small combined preview with KAN/Maxnet if the user wants a big 6-panel later
    print("Done. You can now combine with the existing KAN/Maxnet maps (Fig 4) if desired for a full comparison figure.")

if __name__ == "__main__":
    main()