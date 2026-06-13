#!/usr/bin/env python3
"""Export KAN response curve data from saved best model checkpoint.

from config import DATA_DIR, MODEL_DIR, RESULTS_DIR, TRAIN_CSV

Loads a pre-trained full-data KAN (from 08_train_kan_full_interpret.py)
and computes three families of response curves for the 10 screened variables:
  - proper PDP (average prediction while marginalizing over the empirical distribution of other vars)  ← used for main figures
  - fixed at global (background-weighted) mean
  - fixed at presence mean

Saves:
  - *_response_pdp.csv   (PDP only — consumed by 11_plot_figure2.py and 12_plot_figureS1.py)
  - *_response.csv       (rich: contains prob_pdp + prob_global_mean + prob_presence_mean)
  - means.json           (global_mean and presence_mean per variable on original scale)
  - summary.json

This step is *deterministic* given a fixed checkpoint. No training happens here.

Reproduction:
    python 08_train_kan_full_interpret.py          # (re)train + save ckpt
    python 10_export_response_data.py              # this script → fresh CSVs
    python 11_plot_figure2.py ; python 12_plot_figureS1.py   # main PDP figures
    python 15_plot_supplement_variants.py          # supplementary comparison of the three approaches

KAN_SPE_Ginkgo_Reboot is strictly the 10-variable screened project. No 13-variable
or bio18/bio14 content is part of this clean workspace.
"""
import pandas as pd, numpy as np, json, os, sys, time
from datetime import datetime
import torch
from sklearn.preprocessing import StandardScaler
import scipy.linalg as sla

# Full MKL/scipy lstsq patch (consistent with 08_train_kan_full_interpret.py)
_orig_lstsq = sla.lstsq

def _patched_lstsq(a, b, cond=None, overwrite_a=False, overwrite_b=False,
                   check_finite=True, lapack_driver='gelsy'):
    return _orig_lstsq(a, b, cond=cond, overwrite_a=overwrite_a,
                       overwrite_b=overwrite_b, check_finite=check_finite,
                       lapack_driver=lapack_driver)

sla.lstsq = _patched_lstsq

from kan import KAN
import torch.nn.functional as F

# DATA_DIR provided by config.py
OUTDIR = os.path.join(RESULTS_DIR, "kan_response_data")
os.makedirs(OUTDIR, exist_ok=True)

N_SWEEP = 100

# ── CLI ──
import argparse
parser = argparse.ArgumentParser(description="Export deterministic response data from a saved KAN interpretability checkpoint.")
parser.add_argument("--ckpt", type=str, default="model_best",
                    help="Relative path (from Program/) or full path to pass to KAN.loadckpt. "
                         "Default 'model_best' (the base dir with versioned files) is recommended after using 08_train. "
                         "Other common values: 'model_best/0.1' (for older convention) or full path.")
parser.add_argument("--verify", action="store_true", default=True,
                    help="Run basic sanity checks on PDP ranges for the 10 screened variables (dominant vars vs. flat others).")
args = parser.parse_args()

CKPT_ARG = args.ckpt

# ── Data loading ──
print("Loading data...")
df = pd.read_csv(TRAIN_CSV)
non_env = ['label', 'fold', 'decimalLatitude', 'decimalLongitude']
env_cols = [c for c in df.columns if c not in non_env]
X_orig = df[env_cols].values.astype(np.float64)        # original scale
X_std = StandardScaler().fit_transform(X_orig).astype(np.float64)  # model input
y = df["label"].values.astype(np.float32)
N_pres = int(y.sum())
N_total = len(y)
print(f"Data: {N_total} rows ({N_pres} presence, {N_total - N_pres} background)")
print(f"Env cols ({len(env_cols)}): {env_cols}")

# ── Load pre-trained model ──
print(f"Loading model from checkpoint arg: {CKPT_ARG} ...")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Robust load: prefer the base "model_best" dir (new canonical after 08_train),
# then the "0.1" convention for backward compat, then the passed arg.
candidates = []
if os.path.isabs(CKPT_ARG):
    candidates.append(CKPT_ARG)
else:
    base = os.path.join(SCRIPT_DIR, "model_best")
    candidates.extend([
        os.path.join(SCRIPT_DIR, CKPT_ARG),   # what user passed
        base,                                 # preferred new default
        os.path.join(base, "0.1"),            # old convention that many runs used
        os.path.join(RESULTS_DIR, "best_model"),
    ])

model = None
used_ckpt = None
for cand in candidates:
    try:
        model = KAN.loadckpt(cand)
        used_ckpt = cand
        break
    except Exception:
        continue

if model is None:
    print("ERROR: Could not load checkpoint. Tried:", candidates)
    print("Run with --ckpt pointing to a valid pykan checkpoint dir (the one containing 0.0_*/0.1_* files or history.txt).")
    sys.exit(1)

model.eval()
model.to(DEVICE)
print(f"Model loaded: {type(model).__name__} on {DEVICE} from {used_ckpt}")

# ── Compute response data ──
print(f"\nComputing response curves ({len(env_cols)} vars × {N_SWEEP} pts × 3 strategies)...")
X_base = torch.tensor(X_std, dtype=torch.float32, device=DEVICE)  # safest cross-version tensor creation
gm_std = X_std.mean(axis=0)
pm_std = X_std[y == 1].mean(axis=0)
gm_orig = X_orig.mean(axis=0)
pm_orig = X_orig[y == 1].mean(axis=0)
means = {}

for i, vn in enumerate(env_cols):
    # Sweep in standardized space for model
    x_std_sweep = np.linspace(X_std[:, i].min(), X_std[:, i].max(), N_SWEEP)
    # Original-scale values for CSV
    x_orig_sweep = np.linspace(X_orig[:, i].min(), X_orig[:, i].max(), N_SWEEP)

    pdp_vals = np.zeros(N_SWEEP)
    global_vals = np.zeros(N_SWEEP)
    pres_vals = np.zeros(N_SWEEP)

    for j in range(N_SWEEP):
        v = float(x_std_sweep[j])

        # PDP (Partial Dependence): the proper way to visualize marginal effect.
        # For a candidate value v of variable i, we REPLACE the i-th column
        # in *every* row of the dataset with v, then average the model's
        # predictions across the entire (modified) dataset.
        # This approximates the expectation E[ f(v, X_{-i}) ] where the
        # expectation is over the *empirical marginal distribution* of all
        # other variables X_{-i} (both presence and background points).
        # This is the definition from Friedman (2001) "Greedy function
        # approximation: a gradient boosting machine".
        X_w = X_base.clone()
        X_w[:, i] = v
        with torch.no_grad():
            pdp_vals[j] = torch.sigmoid(model(X_w)).cpu().numpy().mean()

        # Fixed at global mean
        X_fix = torch.tensor(gm_std.astype(np.float32)).unsqueeze(0).to(DEVICE)
        X_fix[0, i] = v
        with torch.no_grad():
            global_vals[j] = torch.sigmoid(model(X_fix)).cpu().numpy().item()

        # Fixed at presence mean
        X_fix = torch.tensor(pm_std.astype(np.float32)).unsqueeze(0).to(DEVICE)
        X_fix[0, i] = v
        with torch.no_grad():
            pres_vals[j] = torch.sigmoid(model(X_fix)).cpu().numpy().item()

    # Save with ORIGINAL-scale x-axis values
    pd.DataFrame({"var_value": x_orig_sweep, "prob_pdp": pdp_vals}).to_csv(
        os.path.join(OUTDIR, f"{vn}_response_pdp.csv"), index=False)

    pd.DataFrame({
        "var_value": x_orig_sweep,
        "prob_pdp": pdp_vals,
        "prob_global_mean": global_vals,
        "prob_presence_mean": pres_vals,
    }).to_csv(os.path.join(OUTDIR, f"{vn}_response.csv"), index=False)

    means[vn] = {
        "global_mean": float(gm_orig[i]),
        "presence_mean": float(pm_orig[i]),
        "min": float(X_orig[:, i].min()),
        "max": float(X_orig[:, i].max()),
    }

    print(f"  {vn}: PDP [{pdp_vals.min():.4f}, {pdp_vals.max():.4f}]")

# ── Save metadata (rich provenance for reproducibility) ──
with open(os.path.join(OUTDIR, "means.json"), "w") as f:
    json.dump(means, f, indent=2)

provenance = {
    "model": used_ckpt,
    "config": {"width": [10, 20, 10, 1], "grid": 10, "k": 3,
               "steps": 300, "opt": "LBFGS", "lamb": 0.005, "lamb_entropy": 0.5},
    "n_variables": len(env_cols),
    "n_sweep_points": N_SWEEP,
    "n_presence": N_pres,
    "n_background": N_total - N_pres,
    "exported_at": datetime.now().isoformat(),
    "script": "10_export_response_data.py",
    "note": "KAN_SPE_Ginkgo_Reboot is the clean 10-screened-variable project. "
            "This CSV set (PDP + global mean + presence mean) is deterministically derived from the weights in the referenced checkpoint. "
            "Re-training the KAN (08_train_kan_full_interpret.py) will produce similar but not identical curves.",
}

with open(os.path.join(OUTDIR, "summary.json"), "w") as f:
    json.dump(provenance, f, indent=2)

print(f"\n✓ Response data exported → {OUTDIR}/")
print(f"  Files: means.json, summary.json, 10 × _response.csv (rich, 3 methods), 10 × _response_pdp.csv (PDP only)")

# ── Optional verification (10 screened variables only — KAN_SPE_Ginkgo_Reboot is strictly 10-var) ──
if args.verify:
    print("\n--- Verification of PDP signals on the current 10 screened variables ---")
    pdp_ranges = {}
    for vn in env_cols:
        p = pd.read_csv(os.path.join(OUTDIR, f"{vn}_response_pdp.csv"))["prob_pdp"].values
        pdp_ranges[vn] = p.max() - p.min()

    # Sort by range descending
    sorted_vars = sorted(pdp_ranges.items(), key=lambda x: x[1], reverse=True)
    print("  PDP ranges (sorted):")
    for vn, r in sorted_vars:
        print(f"    {vn}: {r:.6f}")

    top3 = [v[0] for v in sorted_vars[:3]]
    print(f"\n  Dominant variables (largest PDP range): {top3}")

    checks = []
    # Top variables should show non-trivial marginal effect
    for vn in top3:
        r = pdp_ranges[vn]
        checks.append((f"{vn} has clear PDP signal (range > 0.01)", r > 0.01))
    # Remaining variables are expected to be weak/flat (multivariate effects dominate)
    weak_ranges = [pdp_ranges[v[0]] for v in sorted_vars[3:]]
    if weak_ranges:
        avg_weak = float(np.mean(weak_ranges))
        checks.append(("remaining variables mostly flat/small range (avg < 0.005)", avg_weak < 0.005))

    all_pass = True
    for name, passed in checks:
        status = "PASS" if passed else "WEAK/FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_pass = False

    print(f"\nOverall interpretability data quality check (PDP-focused): {'GOOD' if all_pass else 'NEEDS REVIEW'}")
    if not all_pass:
        print("  (Fresh training runs can vary slightly due to KAN optimization. Compare with previous canonical export.)")
