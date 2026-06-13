#!/usr/bin/env python3
"""

from config import BOUNDARY_GEOJSON, RESULTS_DIR
Produce publication-quality Figure 3: side-by-side current suitability maps
for the canonical R maxnet and the best KAN model (10 screened variables).

Requires the two GeoTIFFs produced by:
  - 16_generate_current_suitability_maps.py  (KAN)
  - 16b_maxent_current_suitability.R         (R maxnet)

Output:
  manuscript/figures/fig3_suitability_maps.png  (high-res for the paper)

Also produces a difference map (KAN - maxnet) for supplementary if desired.
"""

import os
import numpy as np
import rasterio
from rasterio.plot import plotting_extent
from rasterio.mask import mask as rio_mask
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.colors import Normalize
# Boundary overlay is optional (geopandas may not be installed in all environments).
try:
    import geopandas as gpd
    HAS_GPD = True
except ImportError:
    HAS_GPD = False
    gpd = None

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Liberation Serif', 'Times New Roman', 'serif']
plt.rcParams['font.size'] = 9

# SCRIPT_DIR — path resolution now handled by config.py
MAPS_DIR = os.path.join(RESULTS_DIR, "maps")
OUT_DIR = os.path.join(SCRIPT_DIR, "..", "manuscript", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

KAN_TIF = os.path.join(MAPS_DIR, "kan_current_suitability.tif")
MAXENT_TIF = os.path.join(MAPS_DIR, "maxent_current_suitability.tif")

# Boundary (use the same as Figure 1). The file lives in the parent project because
# The data paths use the project directory structure (../data/...).
BOUNDARY = BOUNDARY_GEOJSON
# Fallback search if the above is not present
if not os.path.exists(BOUNDARY):
    for cand in [
        os.path.join(SCRIPT_DIR, "Data", "boundaries", "china_admin_union.geojson"),
        BOUNDARY_GEOJSON,
    ]:
        if os.path.exists(cand):
            BOUNDARY = cand
            break

def load_raster(path):
    """Load full raster (fallback if no clipping)."""
    with rasterio.open(path) as src:
        arr = src.read(1)
        extent = plotting_extent(src)
        crs = src.crs
    arr = np.ma.masked_where(~np.isfinite(arr), arr)
    return arr, extent, crs

def load_and_clip_raster(path, geom):
    """
    Clip the raster to the given geometry (e.g. China boundary) using rasterio.mask.
    Returns masked array (outside geometry is masked) and the corresponding plotting extent.
    This ensures the figure ONLY shows data inside China (incl. HK, Macao, Taiwan).
    """
    with rasterio.open(path) as src:
        # rio_mask crops to the geometry bbox and masks pixels outside the polygon
        out_image, out_transform = rio_mask(
            src, [geom], crop=True, nodata=np.nan, filled=False
        )
        arr = out_image[0]  # single band
        # Build extent from the clipped transform
        height, width = arr.shape
        left = out_transform[2]
        top = out_transform[5]
        xres = out_transform[0]
        yres = out_transform[4]
        right = left + xres * width
        bottom = top + yres * height
        extent = (left, right, bottom, top)
        arr = np.ma.masked_invalid(arr)
        return arr, extent

def main():
    print("Loading rasters for Figure 3...")
    # Load boundary first (we need its geometry for clipping)
    boundary = None
    china_geom = None
    if HAS_GPD:
        try:
            boundary = gpd.read_file(BOUNDARY)
            china_geom = boundary.geometry.iloc[0]  # single union polygon (mainland + Taiwan + HK/Macao)
        except Exception as e:
            print("Warning: could not load boundary geojson:", e)
            boundary = None
    else:
        print("Note: geopandas not available — plotting rasters without boundary overlay (still publication-usable).")

    if china_geom is not None:
        print("Clipping rasters to China boundary (including HK, Macao, Taiwan) ...")
        kan_arr, kan_ext = load_and_clip_raster(KAN_TIF, china_geom)
        max_arr, max_ext = load_and_clip_raster(MAXENT_TIF, china_geom)
    else:
        kan_arr, kan_ext, _ = load_raster(KAN_TIF)
        max_arr, max_ext, _ = load_raster(MAXENT_TIF)

    # Difference on the raw (clipped) arrays.
    # KAN values remain the original low ones (~0-0.16); we only change the *color scale upper limit* to 1 for visual matching.
    diff_arr = kan_arr - max_arr

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Color scaling as requested:
    # - Both colorbars go 0-1 (unified upper limit)
    # - KAN data remains its original 0-0.16 values (its highest areas will only use the lower ~16% of the color ramp)
    # - MaxEnt uses full 0-1
    cmap = 'YlGnBu'
    vmin_kan, vmax_kan = 0, 1.0   # color scale upper limit = 1, but data max is still ~0.16
    vmin_mx, vmax_mx = 0, 1.0

    # Panel 1: KAN (clipped to China) - original low data, color scale stretched to 0-1
    im1 = axes[0].imshow(kan_arr, extent=kan_ext, cmap=cmap, vmin=vmin_kan, vmax=vmax_kan, origin='upper')
    axes[0].set_title('(a) KAN (width=[10,20,10,1], 300 steps)\ndata 0–0.16, color scale 0–1', fontsize=9)
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04, label='Suitability (KAN data max ~0.16)')

    # Panel 2: R maxnet (clipped to China)
    im2 = axes[1].imshow(max_arr, extent=max_ext, cmap=cmap, vmin=vmin_mx, vmax=vmax_mx, origin='upper')
    axes[1].set_title('(b) R maxnet (regmult=1.0, cloglog)', fontsize=10)
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label='Suitability (cloglog 0-1)')

    for ax in axes:
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        if boundary is not None and HAS_GPD:
            boundary.boundary.plot(ax=ax, color='black', linewidth=0.6, alpha=0.7)
        ax.set_aspect('equal', adjustable='box')
        # Tighten view to the clipped China extent
        if kan_ext:
            ax.set_xlim(kan_ext[0], kan_ext[1])
            ax.set_ylim(kan_ext[2], kan_ext[3])

    plt.suptitle('Figure 4. Predicted habitat suitability — current climate (screened 10-variable set)', 
                 fontsize=12, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_png = os.path.join(OUT_DIR, 'fig4_suitability_maps.png')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Figure 4 saved → {out_png}")

    # Save individual subplots for KAN and MaxEnt (for flexible combining)
    for name, (arr, ext), title in [
        ("kan", (kan_arr, kan_ext), "KAN (width=[10,20,10,1], 300 steps)\ndata 0–0.16, color scale 0–1"),
        ("maxent", (max_arr, max_ext), "R maxnet (regmult=1.0, cloglog)"),
    ]:
        fig_i, ax_i = plt.subplots(1, 1, figsize=(5, 4.5))
        im_i = ax_i.imshow(arr, extent=ext, cmap='YlGnBu', vmin=0, vmax=1, origin='upper')
        ax_i.set_title(title, fontsize=9)
        ax_i.set_xlabel("Longitude")
        ax_i.set_ylabel("Latitude")
        ax_i.set_aspect('equal', adjustable='box')
        if boundary is not None and HAS_GPD:
            boundary.boundary.plot(ax=ax_i, color='black', linewidth=0.5, alpha=0.8)
        plt.colorbar(im_i, ax=ax_i, fraction=0.046, pad=0.04, label='Suitability (0-1)')
        plt.tight_layout()
        ind_out = os.path.join(OUT_DIR, f"fig4_{name}_map.png")
        plt.savefig(ind_out, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Individual saved → {ind_out}")

    # Keep the difference module: compute and save separate difference figure (as requested, submodule retained)
    # Difference on the raw (clipped) arrays.
    diff_arr = kan_arr - max_arr
    diff_png = os.path.join(OUT_DIR, 'fig4_difference_kan_minus_maxnet.png')
    fig_d, ax_d = plt.subplots(1, 1, figsize=(7, 6))
    dmin = float(np.ma.min(diff_arr))
    dmax = float(np.ma.max(diff_arr))
    abs_max = max(abs(dmin), abs(dmax))
    imd = ax_d.imshow(diff_arr, extent=kan_ext, cmap='RdBu_r', vmin=-abs_max, vmax=abs_max, origin='upper')
    if boundary is not None and HAS_GPD:
        boundary.boundary.plot(ax=ax_d, color='black', linewidth=0.7)
    ax_d.set_title('KAN − R maxnet (current climate, clipped to China incl. HK/Macao/Taiwan)\nKAN color scale stretched to 0-1 for comparison (data still 0-0.16)')
    plt.colorbar(imd, ax=ax_d, label='Difference')
    ax_d.set_aspect('equal')
    if kan_ext:
        ax_d.set_xlim(kan_ext[0], kan_ext[1])
        ax_d.set_ylim(kan_ext[2], kan_ext[3])
    plt.savefig(diff_png, dpi=250, bbox_inches='tight')
    plt.close()
    print(f"  Supplementary difference map saved → {diff_png} (module retained)")

if __name__ == "__main__":
    main()