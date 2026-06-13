# Scripts Directory

All Python and R analysis scripts for the KAN-SDM pipeline. Run from this directory.

## Path Configuration

All paths are centralized in `config.py`. Import it to get:

| Variable | Points to |
|:---------|:----------|
| `DATA_DIR`, `TRAIN_CSV`, `BOUNDARY_GEOJSON` | `../data/` (repo data) |
| `MODEL_DIR` | `../model/` (canonical checkpoint) |
| `RESULTS_DIR` | `../results/` (metrics) |
| `FIGURES_DIR`, `MAPS_DIR`, `OUTPUT_DIR` | `../output/` (generated outputs) |
| `HISTORY_DIR`, `FUTURE_DIR` | External data (`$KAN_GINKGO_DATA` or `../data_external/`) |

Run `python config.py` to validate your setup.

## Pipeline Order

```
01_download_gbif.py                  GBIF occurrence download
02_extract_env.py                     WorldClim variable extraction
03_prepare_data.py                    Data preparation & variable screening
04_maxent_baseline.py                 MaxEnt (Python LR approx — internal only)
04b_maxent_R.R                        MaxEnt (R maxnet, canonical)
05_rf_xgb.py                          Random Forest + XGBoost
06_kan_baseline.py                    KAN CV baseline
07_mlp_baseline.py                    MLP baseline
08_train_kan_full_interpret.py        Full-data KAN + interpretability
09_significance_test.R                DeLong pairwise significance
10_export_response_data.py            Export PDP response CSVs
12_plot_figureS1.py                   Fig. S1 — all response curves
14_figure2_model_performance.py       Fig. 2 — model comparison
15_figure3_key_response_curves.py     Fig. 3 — key response curves
15_plot_supplement_variants.py        Fig. S2 — PDP/global/presence variants
16_figure4_suitability_maps_native.py Fig. 4 — native suitability maps
16_generate_current_suitability_maps.py   Generate current suitability rasters
16b_maxent_current_suitability.R          MaxEnt current suitability
17_figure5_calibrated_suitability_maps.py Fig. S4 — calibrated maps
18_generate_other_baseline_maps.py    RF/XGB/MLP suitability rasters
19_plot_other_baselines_maps.py       Fig. S3 — other baseline maps
20_combined_all_native_maps.py        Combined native maps
20_combined_curves_and_other_maps.py  Combined curves + maps
21_future_projections.py              Future climate projections (SSP126/370/585)
21b_maxent_future.R                   MaxEnt future projections
22_plot_figS6_centroid_shifts.py      Fig. S6 — centroid shifts
validate_interpretability.py          One-shot validation
```

## External Data

The pipeline needs WorldClim 2.1 BIO tifs (10 arc-min) and future climate layers,
which are too large for GitHub. Set the `KAN_GINKGO_DATA` environment variable:

```bash
export KAN_GINKGO_DATA=/path/to/your/data
```

Expected structure:
```
$KAN_GINKGO_DATA/
├── History/
│   └── wc2.1_10m_bio/
│       ├── bio1.tif
│       ├── bio2.tif
│       └── ... (bio1–bio19)
└── Future/
    ├── ssp126/bio/
    ├── ssp370/bio/
    └── ssp585/bio/
```

Alternatively, create a `data_external/` directory alongside the repo root.

## Quick Start

```bash
# Setup
pip install -r requirements.txt
pip install git+https://github.com/KindXiaoming/pykan.git
export KAN_GINKGO_DATA=/path/to/WorldClim+tifs

# Verify
python config.py

# Run baselines
python 05_rf_xgb.py
python 06_kan_baseline.py
python 07_mlp_baseline.py
Rscript 04b_maxent_R.R

# Interpretability
python 08_train_kan_full_interpret.py
python 10_export_response_data.py
```
