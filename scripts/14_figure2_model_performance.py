#!/usr/bin/env python3
"""Figure 2: Model performance comparison bar plot.

AUC mean +/- sd bar chart with Times New Roman, black-bordered bars,
light yellow background. Publication-ready style.
"""
import json, os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from config import RESULTS_DIR, FIGURES_DIR

# Font setup (global, covers all text elements)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Liberation Serif', 'Times New Roman', 'serif']
plt.rcParams['font.size'] = 11
plt.rcParams['mathtext.fontset'] = 'stix'

RESULTS = RESULTS_DIR
OUT_DIR = FIGURES_DIR
os.makedirs(OUT_DIR, exist_ok=True)

models = [
    ('MaxEnt (R maxnet)', f'{RESULTS}/maxent_cb/metrics_R.json', '#e41a1c'),
    ('MLP',                f'{RESULTS}/mlp_cb/metrics.json',     '#377eb8'),
    ('KAN (best)',         f'{RESULTS}/kan_cb_wider20_steps300_10var/metrics.json', '#4daf4a'),
    ('XGBoost',            f'{RESULTS}/xgb_cb/metrics.json',    '#984ea3'),
    ('Random Forest',      f'{RESULTS}/rf_cb/metrics.json',     '#ff7f00'),
]

entries = []
for name, path, color in models:
    with open(path) as f:
        d = json.load(f)
    entries.append((name, d['auc_mean'], d['auc_std'], color))

# Sort descending by AUC
entries.sort(key=lambda e: e[1], reverse=True)
names = [e[0] for e in entries]
aucs  = [e[1] for e in entries]
stds  = [e[2] for e in entries]
colors = [e[3] for e in entries]

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.set_facecolor('#fffff0')  # light yellow

x = np.arange(len(names))
bars = ax.bar(x, aucs, yerr=stds, capsize=5, color=colors,
              edgecolor='black', linewidth=1.0, width=0.5,
              error_kw={'linewidth': 1.0})

# Annotate bars
for i, (bar, auc, std) in enumerate(zip(bars, aucs, stds)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012,
            f'{auc:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    label_color = 'white' if i < 2 else '#333333'
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.03,
            rf'$\pm${std:.4f}', ha='center', va='bottom', fontsize=8, color=label_color)

ax.set_ylabel('AUC', fontsize=13)
ax.set_ylim(0.72, 0.98)
ax.set_yticks(np.arange(0.75, 0.98, 0.05))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.2f}'))
ax.axhline(0.9, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)

ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=10, rotation=12, ha='right')

ax.set_title('Model Performance Under Spatial Block Cross-Validation\n'
             '(blockCV 150 km, 5-fold external folds)',
             fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.25, linestyle='--', color='#999999')
ax.set_axisbelow(True)

for spine in ax.spines.values():
    spine.set_linewidth(1.0)

plt.tight_layout()
outpath = f'{OUT_DIR}/fig2_model_comparison.png'
plt.savefig(outpath, dpi=300, bbox_inches='tight')
plt.close()
print(f'Saved: {outpath}')
