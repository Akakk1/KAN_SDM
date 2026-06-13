#!/usr/bin/env python3
"""Regenerate all 12 future projection combined maps + main text Figure 5.

Loads existing future TIFs → clips to China boundary (shapely + rasterio.mask)
→ 2×3 grayscale layout → nine-dash inset → single colorbar.

Does NOT run model predictions — only plots from existing TIFs.
"""

import os, sys, json, numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapely.geometry import shape

_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(_PROJ_ROOT, 'scripts'))
from utils.nine_dash import add_nine_dash_inset

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Liberation Serif', 'Times New Roman', 'serif']

FUTURE_DIR = os.path.join(_PROJ_ROOT, 'results', 'maps', 'future')
OUT_DIR = os.path.join(_PROJ_ROOT, 'results', 'figures')
BOUNDARY = os.path.join(_PROJ_ROOT, 'data', 'boundaries', 'china_admin_union.geojson')

SSPS = ['ssp126', 'ssp370', 'ssp585']
PERIODS = ['2021-2040', '2041-2060', '2061-2080', '2081-2100']

LABELS = {
    "maxent": "R maxnet",
    "kan":    "KAN",
    "rf":     "Random Forest",
    "xgb":    "XGBoost",
    "mlp":    "MLP",
}

LAND_COLOR = '#e0e0e0'
SEA_COLOR = 'white'
INSET_BOX = [0.76, 0.01, 0.23, 0.22]


def load_boundary(path):
    """Load China boundary, return union_geom (for clipping) + list of large rings (for plotting)."""
    with open(path) as f:
        raw = json.load(f)
    feat = raw['features'][0]
    union_geom = shape(feat['geometry'])

    rings = []
    for feat in raw['features']:
        g = shape(feat['geometry'])
        if g.geom_type == 'Polygon':
            if g.area > 0.5:
                rings.append(g)
        elif g.geom_type == 'MultiPolygon':
            for p in list(g.geoms):
                if p.area > 0.5:
                    rings.append(p)
    return union_geom, rings


def load_and_clip(tif_path, geom):
    """Load a TIF and clip to the given geometry, returning (masked_array, extent)."""
    with rasterio.open(tif_path) as src:
        out_image, out_transform = rio_mask(
            src, [geom], crop=True, nodata=np.nan, filled=False
        )
        arr = out_image[0]
        h, w = arr.shape
        left = out_transform[2]
        top = out_transform[5]
        xres = out_transform[0]
        yres = out_transform[4]
        extent = (left, left + xres * w, top + yres * h, top)
        arr = np.ma.masked_invalid(arr)
        return arr, extent


def make_plot(ssp, per, output_name=None):
    """Generate one 2×3 future projection plot."""
    ssp_short = ssp.replace("ssp", "")
    if output_name is None:
        output_name = f"figS5_{ssp}_{per}_2x3.png"
    outpath = os.path.join(OUT_DIR, output_name)

    tifs = {
        "maxent": os.path.join(FUTURE_DIR, f"maxent_ssp{ssp_short}_{per}.tif"),
        "kan":    os.path.join(FUTURE_DIR, f"kan_ssp{ssp_short}_{per}.tif"),
        "rf":     os.path.join(FUTURE_DIR, f"rf_ssp{ssp_short}_{per}.tif"),
        "xgb":    os.path.join(FUTURE_DIR, f"xgb_ssp{ssp_short}_{per}.tif"),
        "mlp":    os.path.join(FUTURE_DIR, f"mlp_ssp{ssp_short}_{per}.tif"),
    }

    # Load boundary + clip geometry
    union_geom, boundary_rings = load_boundary(BOUNDARY)
    clip_geom = union_geom.__geo_interface__

    # Load and clip each TIF
    data = {}
    for name, tif_path in tifs.items():
        if not os.path.exists(tif_path):
            print(f"  MISSING: {tif_path}")
            continue
        arr, ext = load_and_clip(tif_path, clip_geom)
        data[name] = (arr, ext)
        valid = np.sum(~arr.mask)
        print(f"  {name}: shape={arr.shape}, valid={valid}, range=[{arr.min():.4f}, {arr.max():.4f}]")

    if len(data) < 5:
        print(f"  Only {len(data)}/5 TIFs available, skipping")
        return

    # --- Plot ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))

    def draw_panel(ax, arr, ext, label):
        ax.set_facecolor(SEA_COLOR)
        im = ax.imshow(arr, extent=ext, cmap='gray_r',
                       vmin=0, vmax=1, origin='upper')
        for poly in boundary_rings:
            xs, ys = poly.exterior.xy
            ax.plot(xs, ys, color='#333333', linewidth=0.8, alpha=0.8, zorder=2)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Longitude", fontsize=11)
        ax.set_ylabel("Latitude", fontsize=11)
        ax.tick_params(labelsize=10)
        ax.set_aspect(1.0 / np.cos(np.radians(35)))
        add_nine_dash_inset(ax, box=INSET_BOX,
                            land_color=LAND_COLOR, sea_color=SEA_COLOR)
        return im

    # Top row
    im_last = None
    for i, name in enumerate(["maxent", "kan"]):
        if name in data:
            arr, ext = data[name]
            im_last = draw_panel(axes[0, i], arr, ext, LABELS[name])
    axes[0, 2].axis('off')

    # Bottom row
    for i, name in enumerate(["rf", "xgb", "mlp"]):
        if name in data:
            arr, ext = data[name]
            im_last = draw_panel(axes[1, i], arr, ext, LABELS[name])

    # Single colorbar on bottom-right panel
    if im_last is not None:
        plt.colorbar(im_last, ax=axes[1, 2], fraction=0.046, pad=0.04,
                     label='Suitability (0-1)')

    # Suptitle (keep for scenario identification)
    ssp_label = f"SSP {ssp_short}"
    plt.suptitle(f"Future climate projections: {ssp_label}, {per}",
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(outpath, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_name}")


def main():
    # All 12 supplementary maps
    for ssp in SSPS:
        for per in PERIODS:
            print(f"\n=== {ssp} {per} ===")
            make_plot(ssp, per)

    print("\nDone.")


if __name__ == "__main__":
    main()
