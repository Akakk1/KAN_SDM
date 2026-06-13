#!/usr/bin/env python3
"""Supplementary Figure S2: KAN response curves under three conditioning strategies.

from config import RESULTS_DIR

For the 10 screened variables in KAN_SPE_Ginkgo_Reboot (strict 10-var project).

Reads the rich *_response.csv (produced by 10_export_response_data.py) which contains:
  prob_pdp, prob_global_mean, prob_presence_mean

Generates a comparison figure showing how fixing at the global (background-heavy) mean
often produces flatter or shifted curves compared with proper PDP or presence-mean
conditioning. This material supports the methodological discussion in the main text
(Section 4.1) without polluting the primary PDP-focused Figure 2 and Figure S1.

Output: manuscript/figures/figS2_response_variants_pdp_global_presence.png

This script is part of the clean 10-variable Reboot pipeline only.
"""
import pandas as pd, os, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Liberation Serif', 'Times New Roman', 'serif']
plt.rcParams['font.size'] = 8

# SCRIPT_DIR — path resolution now handled by config.py
DATA = os.path.join(RESULTS_DIR, "kan_response_data")
OUT = os.path.join(SCRIPT_DIR, "..", "manuscript", "figures")
os.makedirs(OUT, exist_ok=True)

# Exact 10 screened variables (order matches manuscript/methods preference)
ENV = ['bio6','bio11','bio12','bio13','bio2','bio3','bio4','bio5','bio14','bio15']

VAR_LABELS = {
    'bio6': 'Min temp coldest month (°C)',
    'bio11': 'Mean temp coldest qtr (°C)',
    'bio12': 'Annual precipitation (mm)',
    'bio13': 'Precip wettest month (mm)',
    'bio2': 'Mean diurnal range (°C)',
    'bio3': 'Isothermality',
    'bio4': 'Temperature seasonality',
    'bio5': 'Max temp warmest mth (°C)',
    'bio14': 'Precip driest month (mm)',
    'bio15': 'Precipitation seasonality',
}

TRIM_EDGE = 2
X_PAD = 0.04

fig, axes = plt.subplots(2, 5, figsize=(16, 8.0))

for idx, vn in enumerate(ENV):
    ax = axes.flatten()[idx]
    d = pd.read_csv(os.path.join(DATA, f"{vn}_response.csv"))
    n = len(d)

    # Trim edge artifacts for cleaner curves
    d = d.iloc[TRIM_EDGE:n - TRIM_EDGE].copy()
    x = d["var_value"].values
    y_pdp = d["prob_pdp"].values
    y_glob = d["prob_global_mean"].values
    y_pres = d["prob_presence_mean"].values

    ax.plot(x, y_pdp,  '-',  color='#2ca02c', lw=1.6, label='PDP' if idx == 0 else None)
    ax.plot(x, y_glob, '--',  color='#1f77b4', lw=1.1, label='Global mean' if idx == 0 else None)
    ax.plot(x, y_pres, ':',  color='#d62728', lw=1.1, label='Presence mean' if idx == 0 else None)

    # Axis padding
    x_range = x.max() - x.min() if x.max() > x.min() else 1.0
    ax.set_xlim(x.min() - x_range * X_PAD, x.max() + x_range * X_PAD)

    # Use a common-ish y view but allow per-panel breathing room
    all_y = np.concatenate([y_pdp, y_glob, y_pres])
    y_min, y_max = all_y.min(), all_y.max()
    y_range = y_max - y_min if y_max > y_min else 0.01
    ax.set_ylim(max(0, y_min - y_range * 0.15), y_max + y_range * 0.15)

    ax.set_xlabel(VAR_LABELS[vn], fontsize=7)
    ax.set_ylabel('P(presence)', fontsize=7)
    ax.tick_params(labelsize=5.5)
    ax.grid(alpha=0.12)

    # Small per-panel PDP range annotation (main method)
    rng_pdp = y_pdp.max() - y_pdp.min()
    ax.text(0.98, 0.97, f'PDP Δ={rng_pdp:.4f}', transform=ax.transAxes,
            ha='right', va='top', fontsize=5.5,
            bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.75, edgecolor='none'))

# Shared legend
legend_elements = [
    Line2D([0], [0], color='#2ca02c', lw=1.6, linestyle='-', label='PDP (marginalized)'),
    Line2D([0], [0], color='#1f77b4', lw=1.1, linestyle='--', label='Fixed at global (bg-weighted) mean'),
    Line2D([0], [0], color='#d62728', lw=1.1, linestyle=':', label='Fixed at presence-only mean'),
]
fig.legend(handles=legend_elements, loc='upper center', ncol=3, fontsize=8,
           frameon=True, fancybox=True, framealpha=0.9, bbox_to_anchor=(0.5, 0.935))

plt.suptitle('Supplementary Figure S2: KAN response curves — PDP vs. fixed global mean vs. fixed presence mean\n'
             '(10 screened variables; KAN_SPE_Ginkgo_Reboot)',
             fontsize=11, fontweight='bold', y=0.98)

plt.subplots_adjust(top=0.87)
plt.tight_layout(rect=[0, 0, 1, 0.87])
outpath = os.path.join(OUT, 'figS2_response_variants_pdp_global_presence.png')
plt.savefig(outpath, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Supplementary variants figure saved → {outpath}")
print("  This figure supports the methodological discussion (global vs. presence mean vs. proper PDP).")