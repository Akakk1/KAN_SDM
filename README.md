# KAN-SDM: Kolmogorov-Arnold Networks for Species Distribution Modeling

A comparison of Kolmogorov-Arnold Networks (KAN) against canonical MaxEnt and other
methods for presence-only species distribution modeling under strict spatial
cross-validation.

Case study species: ***Ginkgo biloba* L.**, a Mesozoic relict plant, within Chinese
administrative boundaries.

---

## Results

**blockCV 150 km external folds (5-fold), 10 screened variables**

| Model | AUC (mean ± SD) |
|:------|:---------------:|
| MaxEnt (R maxnet) | **0.9168 ± 0.0131** |
| MLP | **0.9047 ± 0.0118** |
| Random Forest | 0.8919 ± 0.0143 |
| KAN ([10,20,10,1], 300 steps) | **0.8832 ± 0.0278** |
| XGBoost | 0.8822 ± 0.0116 |

KAN is statistically indistinguishable from Random Forest and XGBoost (DeLong p > 0.05).
Its core differentiator is **B-spline interpretable response curves** via partial dependence.

---

## Directory Structure

```
KAN_Ginkgo/
├── README.md
├── .gitignore
├── LICENSE
│
├── scripts/                 ← All Python + R scripts
│   ├── config.py                Central path configuration
│   ├── requirements.txt
│   ├── 01_download_gbif.py      1. GBIF download
│   ├── 02_extract_env.py        2. WorldClim extraction
│   ├── 03_prepare_data.py       3. Data prep & variable screening
│   ├── 04_maxent_baseline.py    4a. MaxEnt (Python approx, internal only)
│   ├── 04b_maxent_R.R           4b. MaxEnt (R maxnet, canonical)
│   ├── 05_rf_xgb.py             5. Random Forest + XGBoost
│   ├── 06_kan_baseline.py       6. KAN CV baseline
│   ├── 07_mlp_baseline.py       7. MLP baseline
│   ├── 08_train_kan_full_interpret.py  8. Full-data KAN + interpretability
│   ├── 09_significance_test.R       9. DeLong pairwise tests
│   ├── 10_export_response_data.py  10. Export PDP response CSVs
│   ├── 12–24                      Figure generation scripts
│   ├── 21_future_projections.py   Future climate projections
│   ├── 21b_maxent_future.R        MaxEnt future projections
│   └── utils/                     Helper modules
│
├── data/                     ← Input data (small, committed)
│   ├── ginkgo_training_with_coords.csv    238 pres + 8710 bg points
│   ├── variable_screening_report_v1.0.json
│   └── boundaries/china_admin_union.geojson
│
├── model/                    ← Canonical KAN checkpoint
│   ├── model_best_config.yml
│   ├── model_best_state
│   └── model_best_cache_data
│
└── results/                  ← Key metrics
    ├── significance_tests.json    DeLong test matrix
    └── maxent_metrics.json        MaxEnt (R maxnet) metrics
```

---

## Environment Variables

10 screened WorldClim bioclimatic variables:
`bio6, bio11, bio12, bio13, bio2, bio3, bio4, bio5, bio14, bio15`

Screening: literature-priority (winter cold: bio6/bio11; precipitation: bio12/bio13)
+ Pearson |r| < 0.8.

---

## Reproducing Results

### Requirements

```bash
# Python
pip install -r scripts/requirements.txt
pip install git+https://github.com/KindXiaoming/pykan.git

# R
install.packages(c("maxnet", "blockCV", "dismo", "pROC", "terra", "sf"))
```

### External Data

WorldClim 2.1 tifs and future climate layers (~2 GB) are not included in this repo.
Download them from the sources below, then set `KAN_GINKGO_DATA`:

**Historical climate (WorldClim 2.1, 10 arc-min):**
- https://www.worldclim.org/data/worldclim21.html
- Download "Historical climate data" → 10m resolution → all BIO variables + elevation

**Future climate (CMIP6 downscaled, 10 arc-min):**
- https://www.worldclim.org/data/cmip6/cmip6climate.html
- Download for SSP126, SSP370, SSP585 (all time slices: 2021–2040, 2041–2060, 2061–2080, 2081–2100)

**GBIF occurrence data:**
- https://www.gbif.org/species/2687884 (*Ginkgo biloba*)
- Downloaded programmatically by `01_download_gbif.py`

```bash
export KAN_GINKGO_DATA=/path/to/your/data
```

Expected structure under `$KAN_GINKGO_DATA`:
```
History/
  wc2.1_10m_bio/   ← bio1.tif … bio19.tif
  wc2.1_10m_elev/  ← elevation tif
Future/
  ssp126/bio/
  ssp370/bio/
  ssp585/bio/
```

Alternatively, create `data_external/` alongside the repo root.

### Verify Setup

```bash
cd scripts && python config.py
```

### Run Pipeline

```bash
cd scripts

# Baseline models (CV evaluation)
python 05_rf_xgb.py
python 06_kan_baseline.py
python 07_mlp_baseline.py
Rscript 04b_maxent_R.R

# Significance tests
Rscript 09_significance_test.R

# KAN interpretability (full-data train → PDP export)
python 08_train_kan_full_interpret.py
python 10_export_response_data.py

# Generate figures
python 14_figure2_model_performance.py
python 15_figure3_key_response_curves.py
python 12_plot_figureS1.py
python 16_figure4_suitability_maps_native.py
```

---

## Key Methods

- **Spatial CV**: R `blockCV`, cv_spatial(), 150 km checkerboard, 5-fold external CV
- **Study area**: China GADM strict admin boundary (incl. HK/Macao/Taiwan)
- **Background points**: ~8,710 (randomly sampled within study area)
- **KAN best config**: width=[10,20,10,1], grid=10, k=3, steps=300, LBFGS, λ=0.005, λ_entropy=0.5
- **Interpretability**: PDP (Partial Dependence Plots), averaging over the empirical covariate distribution
- **KAN**: Liu et al. (2024). arXiv:2404.19756
- **MaxEnt**: Phillips et al. (2006). *Ecol. Model.* 190, 231–259

---

## License

MIT — see [LICENSE](LICENSE)
