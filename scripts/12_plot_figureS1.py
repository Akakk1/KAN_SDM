#!/usr/bin/env python3
"""Figure S1: KAN partial dependence profiles for all 10 screened environmental variables.

from config import RESULTS_DIR

Reads response curve CSV data (pre-computed by 10_export_response_data.py).
No model training or data computation — pure visualization.

Y-axes are per-variable (standard in SDM response curves to show functional shape).
Each panel is annotated with its actual PDP range (Δ = max - min) to avoid
misinterpretation of tiny fluctuations (e.g. Δ=0.0001) as strong trends when
auto-scaled. Strong effects (bio6/11/13, Δ~0.09-0.098) are visually dominant;
others are essentially flat at the 10^-3 or smaller level.

Output: figures/figS1_all_response_curves.png
"""
import pandas as pd, os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Liberation Serif', 'Times New Roman', 'serif']
plt.rcParams['font.size'] = 7

# SCRIPT_DIR — path resolution now handled by config.py
DATA = os.path.join(RESULTS_DIR, "kan_response_data")
OUT = os.path.join(SCRIPT_DIR, "..", "manuscript", "figures")
os.makedirs(OUT, exist_ok=True)

ENV = ['bio6','bio11','bio12','bio13','bio2','bio3','bio4','bio5','bio14','bio15']  # current 10 screened vars (no elev in final set)

VAR_LABELS = {
    'bio6': 'Min temp coldest month (°C)', 'bio11': 'Mean temp coldest qtr (°C)',
    'bio12': 'Annual precipitation (mm)',  'bio13': 'Precip wettest month (mm)',
    'bio2': 'Mean diurnal range (°C)',     'bio3': 'Isothermality',
    'bio4': 'Temperature seasonality',     'bio5': 'Max temp warmest mth (°C)',
    'bio14': 'Precip driest month (mm)',   'bio15': 'Precipitation seasonality',
}

TRIM_EDGE = 2
X_PAD = 0.05

n_cols, n_rows = 4, 4
fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 10))

for idx, vn in enumerate(ENV):
    ax = axes.flatten()[idx]
    d = pd.read_csv(os.path.join(DATA, f"{vn}_response_pdp.csv"))
    n = len(d)

    # Trim edge artifacts
    d = d.iloc[TRIM_EDGE:n - TRIM_EDGE].copy()
    x = d["var_value"].values
    y = d["prob_pdp"].values

    ax.plot(x, y, '-', color='#2ca02c', lw=1.3)

    # Axis limits with padding
    x_range = x.max() - x.min() if x.max() > x.min() else 1.0
    ax.set_xlim(x.min() - x_range * X_PAD, x.max() + x_range * X_PAD)
    y_range = y.max() - y.min() if y.max() > y.min() else 1.0
    ax.set_ylim(y.min() - y_range * X_PAD, y.max() + y_range * X_PAD)

    # Annotate actual PDP range (Δ) inside panel — critical for tiny-effect vars
    # so that auto-scaled plots with Δ~0.0001 do not visually imply strong trends.
    rng = y.max() - y.min()
    if rng >= 0.05:
        boxcolor, txtcolor = '#d4edda', '#155724'  # green for strong
    elif rng >= 0.001:
        boxcolor, txtcolor = '#fff3cd', '#856404'  # amber for weak-but-visible
    else:
        boxcolor, txtcolor = '#f0f0f0', '#666666'  # gray for near-flat
    ax.text(0.97, 0.03, f'Δ={rng:.4f}' if rng < 0.01 else f'Δ={rng:.3f}',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=5.5,
            color=txtcolor,
            bbox=dict(boxstyle='round,pad=0.15', facecolor=boxcolor, alpha=0.85, edgecolor='none', linewidth=0))

    ax.set_xlabel(VAR_LABELS[vn], fontsize=6)
    ax.set_ylabel('P(presence)', fontsize=6)
    ax.tick_params(labelsize=5)
    ax.grid(alpha=0.15)

# Hide unused panels
for j in range(idx + 1, n_rows * n_cols):
    axes.flatten()[j].axis('off')

plt.suptitle('Figure S1: KAN Partial Dependence — All 10 Screened Variables',
             fontsize=11, fontweight='bold')
plt.tight_layout()
outpath = os.path.join(OUT, 'figS1_all_response_curves.png')
plt.savefig(outpath, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Figure S1 saved → {outpath}")
