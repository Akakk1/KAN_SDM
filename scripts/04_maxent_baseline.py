#!/usr/bin/env python3
"""
INTERNAL / HISTORICAL ONLY — Python equivalent MaxEnt baseline

This file contains a simplified Python re-implementation (L1-regularized
logistic regression with manually constructed linear/quadratic/hinge
features) that was developed as an early approximation to MaxEnt.

It has NO role in the final scientific work or manuscript.

Rationale for keeping (but not using):
- Historical record of development work done during the "openclaw" phase.
- May be useful for internal debugging or future reference.
- Should NOT be cited, run for paper results, or mentioned in any publication.

All official MaxEnt results in this project use the real R implementation:
  Program/04b_maxent_R.R   (maxnet package + same blockCV folds)

This script and its outputs (metrics.json / predictions.csv) are for
internal use only.
"""
import pandas as pd, numpy as np, os, json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from config import TRAIN_CSV, RESULTS_DIR
df = pd.read_csv(TRAIN_CSV)
# Enforce the 13 non-collinear variables (consistent with data prep and docs)
KEEP = ['bio1','bio2','bio3','bio5','bio7','bio8','bio9','bio12','bio14','bio15','bio18','bio19','elev']
df = df[KEEP + ['decimalLatitude', 'decimalLongitude', 'label', 'fold']].copy()
env_cols = KEEP
X = df[env_cols].values.astype(np.float64)
y = df['label'].values
n_folds = len(np.unique(df['fold']))

print(f"=== MaxEnt Baseline (external blockCV folds) ===")
print(f"Samples={len(X):,} vars={len(env_cols)} presence={y.sum():,} folds={n_folds}")

# MaxEnt-equivalent feature transform
# 1) linear + 2) quadratic + 3) hinge (5 quantiles × 2 directions)
X_feat = [X]  # linear
X_feat.append(X**2)  # quadratic
for j in range(X.shape[1]):
    pts = np.percentile(X[:, j], [20, 40, 60, 80])
    for p in pts:
        X_feat.append(np.maximum(X[:, j:j+1] - p, 0))
        X_feat.append(np.maximum(p - X[:, j:j+1], 0))

X_all = np.column_stack(X_feat)
n_feat = X_all.shape[1]
print(f"Features: {n_feat} (linear:{len(env_cols)} + quad:{len(env_cols)} + hinge:{n_feat - 2*len(env_cols)})")

# Standardization
X_all = StandardScaler().fit_transform(X_all)

# Use external blockCV folds from CSV
fold_ids = df['fold'].values

aucs, nzs = [], []
preds = np.zeros(len(y))
for f in range(n_folds):
    te = fold_ids == f; tr = ~te
    model = LogisticRegression(penalty='l1', C=1.0, solver='saga', max_iter=2000, random_state=42)
    model.fit(X_all[tr], y[tr])
    prob = model.predict_proba(X_all[te])[:, 1]
    auc = roc_auc_score(y[te], prob)
    nz = (model.coef_[0] != 0).sum()
    aucs.append(auc); nzs.append(nz)
    preds[te] = prob
    print(f"  Fold {f}: AUC={auc:.4f}  nonzero_features={nz}/{n_feat}")

# Save consistent with other baselines
OUTDIR = os.path.join(RESULTS_DIR, "maxent_cb")
os.makedirs(OUTDIR, exist_ok=True)
with open(os.path.join(OUTDIR, "metrics.json"), "w") as f:
    json.dump({
        "model": "MaxEnt",
        "cv": "blockCV 150km external folds",
        "auc_mean": round(np.mean(aucs), 4),
        "auc_std": round(np.std(aucs), 4),
        "folds": [round(a, 4) for a in aucs]
    }, f, indent=2)
pd.DataFrame({"label": y, "pred_prob": preds}).to_csv(os.path.join(OUTDIR, "predictions.csv"), index=False)

print(f"\nAUC (blockCV 150km external folds): {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
print(f"Mean nonzero features: {np.mean(nzs):.0f} / {n_feat}")
print(f"✅ Saved to {OUTDIR}")
