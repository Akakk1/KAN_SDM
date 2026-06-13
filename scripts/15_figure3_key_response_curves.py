#!/usr/bin/env python3
"""Figure 3: KAN partial dependence curves for key variables.

from config import RESULTS_DIR

Reads response curve CSV data (pre-computed by 10_export_response_data.py).
No model training or data computation — pure visualization.

Output: figures/fig3_response_curves.png
"""
import pandas as pd, os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Liberation Serif', 'Times New Roman', 'serif']
plt.rcParams['font.size'] = 10

# SCRIPT_DIR — path resolution now handled by config.py
DATA = os.path.join(RESULTS_DIR, "kan_response_data")
OUT = os.path.join(SCRIPT_DIR, "..", "manuscript", "figures")
os.makedirs(OUT, exist_ok=True)

VARS = ['bio13', 'bio11']   # strongest signals from current 10-var PDP (bio13 dominant, bio11 also notable)
VAR_LABELS = {
    'bio13': 'Precipitation of wettest month (mm)',
    'bio11': 'Mean temperature of coldest quarter (°C)',
}
TITLES = {'bio13': 'BIO13', 'bio11': 'BIO11'}

TRIM_EDGE = 2   # skip first N and last N points to avoid edge artifacts
X_PAD = 0.05    # fraction of data range to pad on each side

fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))

for ax, vn in zip(axes, VARS):
    d = pd.read_csv(os.path.join(DATA, f"{vn}_response_pdp.csv"))
    n = len(d)

    # Trim edge artifacts
    d = d.iloc[TRIM_EDGE:n - TRIM_EDGE].copy()
    x = d["var_value"].values
    y = d["prob_pdp"].values

    ax.plot(x, y, '-', color='#2ca02c', lw=2.0)

    # If ensemble data is present (from multi-seed run in 08_train with >1 seed),
    # the CSV can be extended with 'prob_pdp_mean' and 'prob_pdp_std' columns.
    # Example (uncomment when available):
    # if 'prob_pdp_mean' in d.columns:
    #     ymean = d['prob_pdp_mean'].values
    #     ystd = d.get('prob_pdp_std', np.zeros_like(ymean)).values
    #     ax.fill_between(x, ymean - ystd, ymean + ystd, alpha=0.25, color='#2ca02c')

    # Axis limits with padding
    x_range = x.max() - x.min()
    ax.set_xlim(x.min() - x_range * X_PAD, x.max() + x_range * X_PAD)
    y_range = y.max() - y.min()
    ax.set_ylim(max(0, y.min() - y_range * X_PAD), y.max() + y_range * X_PAD)

    ax.set_xlabel(VAR_LABELS[vn], fontsize=10)
    ax.set_ylabel('P(presence)', fontsize=10)
    ax.set_title(TITLES[vn], fontweight='bold', fontsize=11)
    ax.grid(alpha=0.2)

plt.tight_layout()
outpath = os.path.join(OUT, 'fig3_response_curves.png')
plt.savefig(outpath, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Figure 3 saved → {outpath}")
