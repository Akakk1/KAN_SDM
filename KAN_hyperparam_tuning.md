# KAN Hyperparameter Tuning Log

> **Canonical result**: KAN `wider20_steps300_10var` — AUC **0.8832 ± 0.0278** (5-fold, blockCV 150 km)
> All folds healthy: [0.8578, 0.8434, 0.8946, 0.9157, 0.9044]
> Config: steps=300, grid=10, k=3, width=[10,20,10,1], opt=LBFGS, lamb=0.005, lamb_entropy=0.5

## 1. Data scope

- **Occurrence**: 238 points (strict GADM Chinese administrative boundary, incl. HK/MO/TW)
- **Background**: 8,710 valid points
- **Environmental variables**: 10 (bio2, bio3, bio4, bio5, bio6, bio11, bio12, bio13, bio14, bio15)
  — selected via iterative screening with literature priority on winter cold (bio6, bio11) + precipitation (bio12+)
- **Spatial CV**: R blockCV 150 km external folds, k=5

## 2. Model comparison (blockCV 150 km)

| Model | AUC mean ± std | Folds |
|-------|---------------|-------|
| MaxEnt (R maxnet) | **0.9168 ± 0.0131** | [0.9022, 0.9298, 0.9039, 0.9208, 0.9276] |
| MLP | 0.9047 ± 0.0118 | [0.8962, 0.9024, 0.9072, 0.9259, 0.8921] |
| Random Forest | 0.8919 ± 0.0143 | [0.8926, 0.9018, 0.8643, 0.8977, 0.9030] |
| **KAN (best)** | **0.8832 ± 0.0278** | [0.8578, 0.8434, 0.8946, 0.9157, 0.9044] |
| XGBoost | 0.8822 ± 0.0116 | [0.8886, 0.8977, 0.8632, 0.8847, 0.8768] |

**DeLong significance test** (vs KAN best):
- KAN vs MaxEnt: p = 0.0004 (✗ significant)
- KAN vs MLP: p = 0.0026 (✗ significant)
- KAN vs RF: p = 0.3458 (not significant)
- KAN vs XGBoost: p = 0.9777 (not significant)

KAN performs comparably to RF and XGBoost, but lags behind MaxEnt and MLP on this dataset.

## 3. KAN hyperparameter tuning (10 variables, blockCV 150 km folds)

### 3.1 Run log

| tag | steps | grid | k | width | lamb | lamb_entropy | AUC mean | std | Folds | Notes |
|-----|-------|------|---|-------|------|--------------|----------|-----|-------|-------|
| default_10var | 80 | 5 | 3 | 10,8,4,1 | 0.01 | 1.0 | 0.4612 | 0.0408 | [0.4423, 0.4507, 0.4943, 0.5174, 0.4011] | Defaults → all folds near-random |
| wider16_lowreg_10var | 200 | 10 | 3 | 10,16,8,1 | 0.005 | 0.5 | 0.8100 | 0.1183 | [0.5774, 0.8792, 0.8588, 0.9000, 0.8345] | One weak fold (0.577) drags mean |
| wider16_finetune_reg_10var | 200 | **12** | 3 | 10,16,8,1 | **0.003** | **0.3** | 0.8770 | 0.0273 | [0.8340, 0.8985, 0.8765, 0.9121, 0.8638] | Grid 12 + fine-tuned reg: strong, all folds healthy |
| **wider20_steps300_10var** | **300** | **10** | **3** | **10,20,10,1** | **0.005** | **0.5** | **0.8832** | **0.0278** | [0.8578, 0.8434, 0.8946, 0.9157, 0.9044] | **Best**: wider20 + 300 steps, all folds healthy |

### 3.2 Tuning observations

- Default pykan config collapsed to ~0.46 AUC (near-random) on the 10-variable data
- Wider architecture + lower regularization was essential: from 0.46 → 0.81
- Fine-tuning grid to 12 and further lowering reg (lamb=0.003) improved stability (std 0.0273)
- Widening to [10,20,10,1] + 300 steps gave the best overall result (0.8832)
- All folds healthy in both top configurations (no 0.5 collapse)
- Lowering regularization (lamb=0.005, lamb_entropy=0.5) was the single most important adjustment vs defaults

## 4. Final model

**Best config**: steps=300, grid=10, k=3, width=[10,20,10,1], opt=LBFGS, lamb=0.005, lamb_entropy=0.5
→ **AUC 0.8832 ± 0.0278**

Models saved at `Program/model_best/` (full checkpoint + config).

## 5. Interpretability (response curves)

PDP response curves generated for all 10 variables using the final model (exported via `10_export_response_data.py`). Key findings:

- Precipitation variables (bio13, bio14, bio12) showed the clearest univariate PDP signals — consistent with literature identifying precipitation as the dominant driver for *Ginkgo biloba* distribution
- Temperature variables (bio5, bio6, bio11) showed weak univariate effects — multivariate interactions dominate
- Marginal response flatness for temperature variables is expected for narrow-niche relict species (environmental signal embedded in joint/multi-variable space)
- Data: `Results/kan_response_data/` (100-point PDP sweeps per variable)

## 6. Historical notes (pre-variable-screening runs)

Earlier tuning (June 2026, prior to variable screening) used 13 variables at global extent. Those runs informed the general approach (LBFGS + wider architecture + low reg) but are superseded by the 10-variable China-strict blockCV results above. Archived results in `results/_archive/`.
