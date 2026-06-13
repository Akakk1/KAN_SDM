"""Nine-dash line (South China Sea) inset for China maps.

Reads the nine-dash line GeoJSON (MultiPolygon boundaries drawn as
dashed lines) and places a small inset in the bottom-right corner
of the main axes.  Uses axes coordinates so the inset stays inside
the figure regardless of savefig bbox cropping.

Usage:
    from utils.nine_dash import add_nine_dash_inset
    add_nine_dash_inset(ax)          # ax = main map Axes
"""

import json
import os
import matplotlib.pyplot as plt

_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_NINE_DASH_CANDIDATES = [
    os.path.join(_PROJ_ROOT, 'data', 'boundaries', '九段线', '九段线.geojson'),
]
_CHINA_CANDIDATES = [
    os.path.join(_PROJ_ROOT, 'data', 'boundaries', 'china_admin_union.geojson'),
]
_SCS_LON = (105, 125)
_SCS_LAT = (2, 25)


def _find_file(candidates):
    for p in candidates:
        try:
            with open(p):
                return p
        except (FileNotFoundError, IOError):
            continue
    raise FileNotFoundError(f"None found: {candidates}")


def _load_ring_polys(path):
    with open(path) as f:
        data = json.load(f)
    features = data['features'] if data['type'] == 'FeatureCollection' else [data]
    rings = []
    for feat in features:
        geom = feat.get('geometry', feat) if isinstance(feat, dict) else feat
        gtype, coords = geom['type'], geom['coordinates']
        if gtype == 'MultiPolygon':
            for poly in coords:
                rings.append(([p[0] for p in poly[0]], [p[1] for p in poly[0]]))
        elif gtype == 'Polygon':
            rings.append(([p[0] for p in coords[0]], [p[1] for p in coords[0]]))
        elif gtype in ('MultiLineString',):
            for line in coords:
                rings.append(([p[0] for p in line], [p[1] for p in line]))
        elif gtype == 'LineString':
            rings.append(([p[0] for p in coords], [p[1] for p in coords]))
    return rings


def add_nine_dash_inset(parent_ax, box=None, land_color='#f0f0f0',
                        sea_color='#dce8f0'):
    """Add nine-dash line inset (South China Sea) inside the main axes.

    Parameters
    ----------
    parent_ax : matplotlib.axes.Axes
        The main map axes (inset is placed relative to this).
    box : [left, bottom, width, height] in axes coords (0-1).
        Default: [0.73, 0.01, 0.20, 0.18]
    land_color, sea_color : colors for land/ocean in the inset.
    """
    if box is None:
        box = [0.73, 0.01, 0.20, 0.18]

    ax = parent_ax.inset_axes(box)

    try:
        dash_rings = _load_ring_polys(_find_file(_NINE_DASH_CANDIDATES))
    except FileNotFoundError:
        print("[nine_dash] WARNING: nine-dash data not found, skipping")
        ax.set_visible(False)
        return ax

    try:
        china_rings = _load_ring_polys(_find_file(_CHINA_CANDIDATES))
    except FileNotFoundError:
        china_rings = []

    ax.set_xlim(*_SCS_LON)
    ax.set_ylim(*_SCS_LAT)
    ax.set_facecolor(sea_color)

    # Land (intersecting SCS)
    for lons, lats in china_rings:
        if (max(lons) > _SCS_LON[0] and min(lons) < _SCS_LON[1]
                and max(lats) > _SCS_LAT[0] and min(lats) < _SCS_LAT[1]):
            ax.fill(lons, lats, facecolor=land_color,
                    edgecolor='#999999', linewidth=0.4, zorder=2)

    # Nine-dash line (red dashed)
    for lons, lats in dash_rings:
        ax.plot(lons, lats, color='#cc0000', linewidth=1.0,
                linestyle='--', dashes=(5, 3), zorder=3)

    # Label removed (clean inset, caption in figure legend)

    ax.set_aspect(1.05)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color('#444444')
        spine.set_linewidth(0.8)

    return ax
