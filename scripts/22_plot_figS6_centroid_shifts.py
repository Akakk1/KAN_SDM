import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch
import os

_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

from config import BOUNDARY_GEOJSON

# =========================
# Load data
# =========================

df = pd.read_csv(os.path.join(_PROJ_ROOT, 'results', 'future_centroids.csv'))

boundary_path = os.path.join(_PROJ_ROOT, 'data', 'boundaries', 'china_admin_union.geojson')
gdf = gpd.read_file(boundary_path)

periods = ["2021-2040", "2041-2060", "2061-2080", "2081-2100"]
ssps = ["126", "370", "585"]
models = ["maxent", "kan"]

ssp_colors = {
    "126": "#2E7D32",
    "370": "#F57C00",
    "585": "#C62828"
}

model_markers = {
    "maxent": "s",
    "kan": "^"
}

# =========================
# Calculate centroid extent
# =========================

all_lon = []
all_lat = []

for ssp in ssps:
    for model in models:

        rows = df[
            (df.model == model)
            & (df.scenario.str.contains(f"ssp{ssp}"))
        ]

        all_lon.extend(rows["centroid_lon"].tolist())
        all_lat.extend(rows["centroid_lat"].tolist())

margin = 0.3

zoom_xmin = min(all_lon) - margin
zoom_xmax = max(all_lon) + margin

zoom_ymin = min(all_lat) - margin
zoom_ymax = max(all_lat) + margin

# =========================
# Figure layout - correct way for equal aspect + same physical height for A/B
# Reference from S5 scripts (21_future_projections.py, 17_...): figsize~(13-14,8),
# set_aspect('equal') on each map, tight proportions.
# Key fix: compute box width for each panel from its data aspect ratio so that
# when we allocate the *same height*, the box w/h exactly matches data lon/lat span.
# This prevents aspect from resizing the boxes (the bug that made previous "same height"
# allocations get overridden, causing misalignment and flat A).
# =========================

fig_width = 14.0
fig_height = 8.0
fig = plt.figure(figsize=(fig_width, fig_height))

# Common physical height for the two map areas (in figure fraction).
# This guarantees A and B are exactly the same height (aligned).
common_h = 0.80
bottom = 0.08

# Data aspect for A (national overview): use the limits we actually set
# Narrower lon span to make China fill the panel better, less empty west/east.
# Extended north to 54.5 to stop clipping the northernmost areas.
a_lon = 136 - 73
a_lat = 54.5 - 18
a_ratio = a_lon / a_lat   # ~1.726 (narrower than before)

# Data aspect for B (auto zoom from centroids)
b_lon = zoom_xmax - zoom_xmin
b_lat = zoom_ymax - zoom_ymin
b_ratio = b_lon / b_lat

# Compute widths so box w/h == data ratio (perfect equal aspect, no squashing)
a_w = common_h * a_ratio / (fig_width / fig_height)   # convert to figure fraction
b_w = common_h * b_ratio / (fig_width / fig_height)

# Place A left, B right with small gap
gap = 0.025
left_a = 0.03
left_b = left_a + a_w + gap

axA = fig.add_axes([left_a, bottom, a_w, common_h])
axB = fig.add_axes([left_b, bottom, b_w, common_h])

# =========================
# Panel A
# =========================

gdf.plot(
    ax=axA,
    facecolor="#f5f5f0",
    edgecolor="#333333",
    linewidth=0.7,
    alpha=0.6,
    zorder=1
)

axA.set_facecolor("#e8f0f5")

for ssp in ssps:

    color = ssp_colors[ssp]

    for model in models:

        marker = model_markers[model]

        lons = []
        lats = []

        for per in periods:

            row = df[
                (df.scenario == f"ssp{ssp}_{per}")
                & (df.model == model)
            ]

            if len(row) == 0:
                continue

            lons.append(row["centroid_lon"].values[0])
            lats.append(row["centroid_lat"].values[0])

        if len(lons) < 2:
            continue

        axA.plot(
            lons,
            lats,
            color=color,
            linewidth=0.8,
            alpha=0.75,
            zorder=2
        )

        axA.annotate(
            '',
            xy=(lons[-1], lats[-1]),
            xytext=(lons[-2], lats[-2]),
            arrowprops=dict(
                arrowstyle='->',
                color=color,
                lw=0.8,
                mutation_scale=6
            ),
            zorder=2
        )

        for lon, lat in zip(lons, lats):

            axA.plot(
                lon,
                lat,
                marker=marker,
                color=color,
                markersize=4.5,
                markeredgecolor='white',
                markeredgewidth=0.3,
                zorder=3
            )

# zoom rectangle

rect = mpatches.Rectangle(
    (zoom_xmin, zoom_ymin),
    zoom_xmax - zoom_xmin,
    zoom_ymax - zoom_ymin,
    linewidth=1.6,
    edgecolor='black',
    facecolor='none',
    linestyle='--',
    zorder=5
)

axA.add_patch(rect)

axA.text(
    (zoom_xmin + zoom_xmax) / 2,
    zoom_ymax + 0.4,
    "Zoom region",
    ha='center',
    fontsize=8,
    fontweight='bold',
    bbox=dict(
        boxstyle='round,pad=0.2',
        facecolor='white',
        alpha=0.9
    )
)

axA.set_xlim(73, 136)
axA.set_ylim(18, 54.5)  # narrower lon to focus China; extended north to prevent clippingernmost areas (matches typical S5 raster extents after China clip)

axA.set_xlabel("Longitude (°E)")
axA.set_ylabel("Latitude (°N)")

axA.set_title(
    "(A) National overview of centroid shifts",
    fontsize=12,
    fontweight='bold'
)

axA.grid(True, linestyle=':', alpha=0.2)
axA.set_aspect('equal', adjustable='box')  # lock outer axes box so allocated height is respected; data scales inside with correct geo proportions (matches S5 style)

# =========================
# Panel B
# =========================

gdf.plot(
    ax=axB,
    facecolor="#f5f5f0",
    edgecolor="#333333",
    linewidth=0.5,
    alpha=0.5,
    zorder=1
)

axB.set_facecolor("#e8f0f5")

for ssp in ssps:

    color = ssp_colors[ssp]

    for model in models:

        marker = model_markers[model]

        lons = []
        lats = []

        for per in periods:

            row = df[
                (df.scenario == f"ssp{ssp}_{per}")
                & (df.model == model)
            ]

            if len(row) == 0:
                continue

            lons.append(row["centroid_lon"].values[0])
            lats.append(row["centroid_lat"].values[0])

        if len(lons) < 2:
            continue

        axB.plot(
            lons,
            lats,
            color=color,
            linewidth=2.0,
            alpha=0.95,
            zorder=2
        )

        axB.annotate(
            '',
            xy=(lons[-1], lats[-1]),
            xytext=(lons[-2], lats[-2]),
            arrowprops=dict(
                arrowstyle='->',
                color=color,
                lw=2.0,
                mutation_scale=12
            ),
            zorder=2
        )

        for lon, lat in zip(lons, lats):

            axB.plot(
                lon,
                lat,
                marker=marker,
                color=color,
                markersize=9,
                markeredgecolor='white',
                markeredgewidth=0.5,
                zorder=3
            )

        # Annotate start (first period) and end (last period) centroids with coordinates
        # only on the enlarged Panel B, small font + white bbox to stay readable
        for i in [0, -1]:
            lon = lons[i]
            lat = lats[i]
            axB.annotate(
                f"{lon:.2f}°E\n{lat:.2f}°N",
                xy=(lon, lat),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=5.5,
                color=color,
                ha="left",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.12", facecolor="white", alpha=0.85, edgecolor="none"),
                zorder=5
            )

# Auto zoom

axB.set_xlim(zoom_xmin, zoom_xmax)
axB.set_ylim(zoom_ymin, zoom_ymax)

axB.set_xlabel("Longitude (°E)")
axB.set_ylabel("Latitude (°N)")

axB.set_title(
    "(B) Enlarged centroid shifts",
    fontsize=12,
    fontweight='bold'
)

axB.grid(True, linestyle=':', alpha=0.3)
axB.set_aspect('equal', adjustable='box')  # lock outer box for alignment with A

# =========================
# Connection arrow
# =========================

con = ConnectionPatch(
    xyA=(zoom_xmax, (zoom_ymin + zoom_ymax) / 2),
    coordsA=axA.transData,
    xyB=(0.02, 0.5),
    coordsB=axB.transAxes,
    arrowstyle='->',
    lw=1.2,
    color='black'
)

fig.add_artist(con)

# =========================
# Legend
# =========================

legend_elements = [
    Line2D(
        [0],
        [0],
        color=ssp_colors[s],
        linewidth=2,
        label=f"SSP{s}"
    )
    for s in ssps
]

legend_elements += [
    Line2D(
        [0],
        [0],
        marker='s',
        color='w',
        markerfacecolor='gray',
        markersize=7,
        markeredgecolor='white',
        label='MaxEnt'
    ),
    Line2D(
        [0],
        [0],
        marker='^',
        color='w',
        markerfacecolor='gray',
        markersize=7,
        markeredgecolor='white',
        label='KAN'
    )
]

axA.legend(
    handles=legend_elements,
    loc='upper left',
    fontsize=8,
    framealpha=0.95
)

# =========================
# Save
# =========================

plt.savefig(
    "manuscript/figures/figS6_centroid_shifts.png",  # canonical name for manuscript
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.close()

print("Finished: Panel A/B version exported.")