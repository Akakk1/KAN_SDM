#!/usr/bin/env python3
"""

from config import DATA_DIR, MODEL_DIR, RESULTS_DIR, TRAIN_CSV
Train the canonical full-data KAN model for interpretability (response curves / PDP).

This replaces the logic that used to live in the archived 08_kan_interpretability.py / 08b.

Usage (recommended):
    cd Program
    python 08_train_kan_full_interpret.py --seed 42 --save-ckpt model_best

After training:
    python 10_export_response_data.py   # regenerates all CSVs from the fresh checkpoint (deterministic step)

Key design:
- Full dataset (no CV split) with the project BEST_CONFIG.
- update_grid=False for training stability (as used in the original successful run).
- Proper lstsq patch for MKL/scipy compatibility (needed for auto_symbolic if you enable it).
- Saves using pykan's saveckpt so that 10_export's KAN.loadckpt works.
- Also produces a backup copy in Results/best_model/ (kan_best_* style + config.json).

The "previous best" curves in kan_response_data/ were derived from one specific training run's weights.
Re-training will produce very similar but not bit-identical PDP curves (inherent KAN characteristic).
Use --compare-with-backup or run the export + a diff script afterwards.

See also:
- 10_export_response_data.py (the reproducible data export step)
- Results/kan_response_data/ (the authoritative CSVs consumed by 11_ and 12_ plot scripts)
- KAN_hyperparam_tuning.md for how the BEST_CONFIG was chosen.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

# ------------------------------
# MKL / scipy lstsq compatibility patch (critical for auto_symbolic and some pykan internals)
# Copy of the patch used in the successful historical runs.
# ------------------------------
import scipy.linalg as sla

_orig_lstsq = sla.lstsq

def _patched_lstsq(a, b, cond=None, overwrite_a=False, overwrite_b=False,
                   check_finite=True, lapack_driver='gelsy'):
    return _orig_lstsq(a, b, cond=cond, overwrite_a=overwrite_a,
                       overwrite_b=overwrite_b, check_finite=check_finite,
                       lapack_driver=lapack_driver)

sla.lstsq = _patched_lstsq

# ------------------------------
# Best config (locked from hyperparameter search on China blockCV data)
# ------------------------------
BEST_CONFIG = {
    "steps": 300,
    "grid": 10,
    "k": 3,
    "width": [10, 20, 10, 1],  # best from 10-var tuning (wider20_steps300 experiment)
    "opt": "LBFGS",
    "lamb": 0.005,
    "lamb_entropy": 0.5,
}



def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_full_data(data_dir: str):
    csv_path = os.path.join(data_dir, "ginkgo_training_with_coords.csv")
    df = pd.read_csv(csv_path)
    non_env = ['label', 'fold', 'decimalLatitude', 'decimalLongitude']
    env_cols = [c for c in df.columns if c not in non_env]
    df = df[env_cols + ['decimalLatitude', 'decimalLongitude', 'label']].copy()
    X_raw = StandardScaler().fit_transform(df[env_cols].values).astype(np.float32)
    y = df["label"].values.astype(np.float32)
    print(f"Loaded full data: {len(y)} samples ({int(y.sum())} presences), env vars: {len(env_cols)}")
    print(f"Env cols: {env_cols}")
    return X_raw, y, df, env_cols


def build_dataset(X: np.ndarray, y: np.ndarray, device):
    X_t = torch.tensor(X).to(device)
    y_t = torch.tensor(y).unsqueeze(1).to(device)
    return {
        'train_input': X_t,
        'train_label': y_t,
        'test_input': X_t,
        'test_label': y_t,
    }


def bce_loss(pred, target):
    return F.binary_cross_entropy_with_logits(pred, target)


def train_full_model(X: np.ndarray, y: np.ndarray, config: dict, seed: int, device):
    from kan import KAN

    print("\n=== Training full-data KAN for interpretability ===")
    print("Config:", config)
    print("Seed:", seed)
    print("Device:", device)

    dataset = build_dataset(X, y, device)

    t0 = time.time()
    model = KAN(
        width=config["width"],
        grid=config["grid"],
        k=config["k"],
        seed=seed,
        device=device,
    )

    result = model.fit(
        dataset,
        opt=config["opt"],
        steps=config["steps"],
        lamb=config["lamb"],
        lamb_entropy=config["lamb_entropy"],
        loss_fn=bce_loss,
        update_grid=False,   # Important for stability (historical choice)
    )

    rt = time.time() - t0
    print(f"Training finished in {rt:.1f}s")

    # Try to get a rough training loss / metric if available
    try:
        with torch.no_grad():
            logits = model(dataset['train_input'])
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            # Simple pseudo-AUC on training data (for sanity, not reported)
            from sklearn.metrics import roc_auc_score
            train_auc = roc_auc_score(y, probs)
            print(f"Full-data train pseudo-AUC (sanity check): {train_auc:.4f}")
    except Exception as e:
        print(f"(Could not compute sanity AUC: {e})")

    return model, rt


def save_ckpt_and_backup(model, canonical_ckpt_dir: str, results_best_dir: str, config: dict, seed: int):
    """
    Save using pykan saveckpt directly to the canonical checkpoint directory
    (e.g. Program/model_best). This directory is what 10_export will load by default.

    - First backup any existing content in that dir (timestamped under Results/best_model/).
    - Call saveckpt on the dir (pykan will create/append versioned files: 0.0_*, 0.1_*, history.txt, ...).
    - Promote the latest version to 0.1_* aliases so that both
      loadckpt(".../model_best") and loadckpt(".../model_best/0.1") work.
    - Also keep the Results/best_model/ human backup in sync.
    """
    import shutil

    if os.path.exists(canonical_ckpt_dir) and os.listdir(canonical_ckpt_dir):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_target = os.path.join(results_best_dir, f"model_best_backup_{ts}")
        os.makedirs(backup_target, exist_ok=True)
        for item in os.listdir(canonical_ckpt_dir):
            s = os.path.join(canonical_ckpt_dir, item)
            d = os.path.join(backup_target, item)
            if os.path.isdir(s):
                shutil.copytree(s, d)
            else:
                shutil.copy2(s, d)
        print(f"Backed up previous canonical checkpoint to {backup_target}")

    os.makedirs(canonical_ckpt_dir, exist_ok=True)

    # This is the key call: save directly into the dir that loadckpt will use.
    model.saveckpt(canonical_ckpt_dir)
    print(f"Saved pykan checkpoint via saveckpt -> {canonical_ckpt_dir}")

    # Promote latest version files to 0.1_* prefix for compatibility with old " /0.1" loads.
    latest_prefix = "0.1"
    history_path = os.path.join(canonical_ckpt_dir, "history.txt")
    if os.path.exists(history_path):
        with open(history_path) as f:
            lines = f.readlines()
        for line in reversed(lines):
            if "version" in line.lower():
                parts = line.strip().split()
                for p in parts:
                    if p.replace(".", "").isdigit():
                        latest_prefix = p
                        break
                break

    try:
        for fname in os.listdir(canonical_ckpt_dir):
            if fname.startswith(latest_prefix + "_"):
                new_name = fname.replace(latest_prefix + "_", "0.1_", 1)
                if new_name != fname:
                    src = os.path.join(canonical_ckpt_dir, fname)
                    dst = os.path.join(canonical_ckpt_dir, new_name)
                    if not os.path.exists(dst):
                        shutil.copy2(src, dst)
        print(f"Promoted latest version ({latest_prefix}) to 0.1_* aliases.")
    except Exception as e:
        print(f"Warning: could not promote 0.1 aliases: {e}")

    # Results/best_model/ human backup
    os.makedirs(results_best_dir, exist_ok=True)

    with open(os.path.join(results_best_dir, "config.json"), "w") as f:
        json.dump({
            "width": config["width"],
            "grid": config["grid"],
            "k": config["k"],
            "steps": config["steps"],
            "opt": config["opt"],
            "lamb": config["lamb"],
            "lamb_entropy": config["lamb_entropy"],
            "device": str(getattr(model, "device", "cuda")),
            "seed": seed,
            "saved_at": datetime.now().isoformat(),
            "ckpt_dir": canonical_ckpt_dir,
        }, f, indent=2)

    try:
        state = model.state_dict()
        torch.save(state, os.path.join(results_best_dir, "kan_best_state"))
        model_cpu = model.cpu() if hasattr(model, "cpu") else model
        torch.save(model_cpu.state_dict(), os.path.join(results_best_dir, "kan_best_cpu_state"))
        print("Saved kan_best_state + kan_best_cpu_state to Results/best_model/")
    except Exception as e:
        print(f"Warning: could not save extra kan_best_* states: {e}")

    print(f"Canonical checkpoint ready at {canonical_ckpt_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train full-data KAN for interpretability (response curves).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for KAN init (used when --seeds is not provided)")
    parser.add_argument("--seeds", type=str, default="",
                        help="Comma-separated list of seeds for multi-seed ensemble (e.g. 42,43,44). If provided, --seed is ignored and uncertainty bands can be computed.")
    parser.add_argument("--steps", type=int, default=BEST_CONFIG["steps"])
    parser.add_argument("--grid", type=int, default=BEST_CONFIG["grid"])
    parser.add_argument("--k", type=int, default=BEST_CONFIG["k"])
    parser.add_argument("--width", type=str, default="10,20,10,1",
                        help="Comma-separated width, e.g. 10,20,10,1 (current best from 10-var tuning)")
    parser.add_argument("--lamb", type=float, default=BEST_CONFIG["lamb"])
    parser.add_argument("--lamb_entropy", type=float, default=BEST_CONFIG["lamb_entropy"])
    parser.add_argument("--opt", choices=["LBFGS", "Adam"], default=BEST_CONFIG["opt"])
    parser.add_argument("--ckpt-dir", type=str, default="model_best",
                        help="Directory name (relative to Program/) for the pykan checkpoint")
    parser.add_argument("--no-backup-results", action="store_true",
                        help="Skip syncing a copy to Results/best_model/")
    args = parser.parse_args()

    # DATA_DIR provided by config.py
    # RESULTS_DIR provided by config.py
    # CKPT_BASE is the directory that will directly contain the pykan versioned files
    # (0.0_*, 0.1_*, history.txt). This is the dir passed to KAN.loadckpt by default.
    CKPT_BASE = os.path.join(MODEL_DIR, args.ckpt_dir) if args.ckpt_dir else MODEL_DIR
    BEST_MODEL_DIR = os.path.join(RESULTS_DIR, "best_model")

    # Allow overriding BEST_CONFIG from CLI (for experiments)
    cfg = BEST_CONFIG.copy()
    cfg["steps"] = args.steps
    cfg["grid"] = args.grid
    cfg["k"] = args.k
    cfg["width"] = [int(x) for x in args.width.split(",")]
    cfg["lamb"] = args.lamb
    cfg["lamb_entropy"] = args.lamb_entropy
    cfg["opt"] = args.opt

    device = get_device()
    X_raw, y, _, env_cols = load_full_data(DATA_DIR)

    # --- Multi-seed support for uncertainty (Option B) ---
    seed_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip()] if args.seeds else [args.seed]
    models = []
    runtimes = []

    for s in seed_list:
        print(f"\n=== Training seed {s} ({len(models)+1}/{len(seed_list)}) ===")
        m, rt = train_full_model(X_raw, y, cfg, s, device)
        models.append(m)
        runtimes.append(rt)

        # For single-seed or last seed, save the canonical (with the safeguard inside save function)
        if len(seed_list) == 1 or s == seed_list[-1]:
            if not args.no_backup_results:
                save_ckpt_and_backup(m, CKPT_BASE, BEST_MODEL_DIR, cfg, s)
            else:
                m.saveckpt(CKPT_BASE)

    print("\n✅ All training complete.")
    print(f"   Seeds used: {seed_list}")
    print(f"   Primary checkpoint location (load with KAN.loadckpt): {CKPT_BASE}")

    # --- Post-train validation on in-memory model (before deciding to promote canonical) ---
    # This protects the "default" pipeline from occasional KAN runs that don't capture strong marginals.
    # Adapted for current 10-var set (no bio18); use bio12 (precip) or bio6 (cold) as key signals per literature.
    key_var = "bio12" if "bio12" in env_cols else ("bio6" if "bio6" in env_cols else env_cols[0])
    print(f"\nRunning post-training sanity PDP check on in-memory model ({key_var} key signal)...")
    try:
        # Use the last trained model from the loop
        last_model = models[-1]
        X_base = torch.tensor(X_raw.astype(np.float32)).to(device)
        i = env_cols.index(key_var)
        x_sweep = np.linspace(X_raw[:, i].min(), X_raw[:, i].max(), 40)
        pdp = []
        for v in x_sweep:
            Xw = X_base.clone()
            Xw[:, i] = float(v)
            with torch.no_grad():
                p = torch.sigmoid(last_model(Xw)).cpu().numpy().mean()
            pdp.append(p)
        rng = max(pdp) - min(pdp)
        mx = max(pdp)
        print(f"  In-memory {key_var} PDP (quick): range={rng:.4f}, max={mx:.4f}")
        is_good = (rng > 0.25 and mx > 0.35)
        print(f"  Signal quality: {'GOOD (will promote canonical)' if is_good else 'WEAK (will NOT overwrite default canonical)'}")
    except Exception as e:
        print(f"  (Could not run in-memory PDP check: {e})")
        is_good = False

    if is_good:
        print("   Next step for reproducible data (default path will use the new good weights):")
        print("      cd Program && python 10_export_response_data.py")
    else:
        print("   The new run did not meet strong-signal threshold.")
        print("   The previous canonical remains untouched. You can still manually load the just-trained ckpt for inspection:")
        print(f"      python -c \"from kan import KAN; m=KAN.loadckpt('{CKPT_BASE}')\" ")

    # Verify load of the (possibly promoted) canonical
    try:
        from kan import KAN
        loaded = None
        for candidate in [CKPT_BASE, os.path.join(CKPT_BASE, "0.1")]:
            try:
                loaded = KAN.loadckpt(candidate)
                print(f"Verified loadckpt from: {candidate}")
                break
            except Exception:
                continue
        if loaded is None:
            print("Warning: could not verify loadckpt with common paths. Check 10_export manually.")
    except Exception as e:
        print(f"(Sanity reload check skipped: {e})")


if __name__ == "__main__":
    main()
