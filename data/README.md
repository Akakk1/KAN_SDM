# Data

Small input files for the pipeline. Large external data sources must be downloaded separately.

## Included

| File | Description |
|:-----|:------------|
| `ginkgo_training_with_coords.csv` | 8,948 points (238 pres + 8,710 bg), 10 env vars + label + fold + coords |
| `variable_screening_report_v1.0.json` | Variable screening method & final 10-variable list |
| `variable_screening_summary.json` | Machine-readable selected vars |
| `boundaries/china_admin_union.geojson` | GADM China + Taiwan union polygon |

## Required External Data

Not included (too large for GitHub). Download separately:

| Source | URL |
|:-------|:----|
| WorldClim 2.1 (10 arc-min) | <https://www.worldclim.org/data/worldclim21.html> |
| GBIF occurrence | Auto-downloaded by `01_download_gbif.py` |

Place WorldClim `.tif` files under `scripts/Data/History/` (path convention in scripts).
