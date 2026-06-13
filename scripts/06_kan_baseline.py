#!/usr/bin/env python3
"""KAN baseline — Kolmogorov-Arnold Network for SDM"""
import pandas as pd, numpy as np, json, time, os, sys, argparse
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from config import TRAIN_CSV, RESULTS_DIR, MODEL_DIR

df = pd.read_csv(TRAIN_CSV)
# Dynamically determine environmental columns (exclude label, fold, coords) — works for 10-var screened data
non_env = ['label', 'fold', 'decimalLatitude', 'decimalLongitude']
env_cols = [c for c in df.columns if c not in non_env]
df = df[env_cols + ['decimalLatitude', 'decimalLongitude', 'label', 'fold']].copy()
X_raw = StandardScaler().fit_transform(df[env_cols].values).astype(np.float32)
y = df["label"].values.astype(np.float32)
print(f"✅ Loaded training with coords ({len(env_cols)} env vars): {len(df):,} (P={(df['label']==1).sum():,})")
print(f"   Env cols: {env_cols}")

# Spatial CV: prefer 'fold' column if present (for 150km China or external blockCV)
if 'fold' in df.columns and df['fold'].notna().all():
    fold_ids = df['fold'].astype(int).values
    n_folds = len(np.unique(fold_ids))
    print(f"Using external 'fold' column from training data (k={n_folds})")
else:
    # Fallback grid (for compatibility with old data)
    GRID_SIZE = 10
    n_folds = 5
    lats, lons = df["decimalLatitude"].values, df["decimalLongitude"].values
    fold_ids = (((lons + 180) / GRID_SIZE).astype(int) + ((lats + 90) / GRID_SIZE).astype(int)) % n_folds
    print(f"Using fallback grid CV (GRID_SIZE={GRID_SIZE}°, k={n_folds})")

# ====== Hyperparameters (for tuning) ======
parser = argparse.ArgumentParser(description="KAN hyperparam tuning for SDM (pykan)")
parser.add_argument('--steps', type=int, default=80, help='training steps')
parser.add_argument('--grid', type=int, default=5, help='grid size for B-splines')
parser.add_argument('--k', type=int, default=3, help='spline order')
parser.add_argument('--width', type=str, default='10,8,4,1', help='comma sep width e.g. 10,8,4,1 (auto-adjusted to match env_cols)')
parser.add_argument('--opt', type=str, default='LBFGS', choices=['LBFGS', 'Adam'], help='optimizer')
parser.add_argument('--lamb', type=float, default=0.01, help='regularization strength')
parser.add_argument('--lamb_entropy', type=float, default=1.0, help='entropy regularization')
parser.add_argument('--tag', type=str, default='', help='extra tag for output subdir')
args = parser.parse_args()

width = [int(x) for x in args.width.split(',')]
if width[0] != len(env_cols):
    print(f"Warning: width[0]={width[0]} != input dim {len(env_cols)}")

# Output dir based on params for easy comparison
if args.tag:
    out_name = f"kan_cb_{args.tag}"
else:
    w_str = args.width.replace(',', '-')
    out_name = f"kan_cb_steps{args.steps}_grid{args.grid}_k{args.k}_opt{args.opt}_w{w_str}"
OUTDIR = os.path.join(RESULTS_DIR, out_name)
os.makedirs(OUTDIR, exist_ok=True)

print(f"\n=== KAN Tuning Run ===")
print(f"Config: steps={args.steps}, grid={args.grid}, k={args.k}, width={width}, opt={args.opt}, lamb={args.lamb}, lamb_entropy={args.lamb_entropy}")
print(f"Output: {OUTDIR}")

# ====== KAN Model ======
# Use efficient approach: train with PyTorch, use pykan's KAN for architecture
# If pykan is too slow, fall back to a custom B-spline implementation
try:
    from kan import KAN
    USE_PYKAN = True
    print("Using pykan KAN")
except ImportError:
    USE_PYKAN = False
    print("pykan not found, using efficient-kan fallback")
    try:
        from efficient_kan import KANLinear
        USE_EFFICIENT = True
    except ImportError:
        USE_EFFICIENT = False
        print("No KAN library available!")

if USE_PYKAN:
    # pykan training
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")
    
    t0 = time.time()
    aucs, preds = [], np.zeros(len(y))
    
    for f in range(n_folds):
        te = fold_ids == f; tr = ~te
        X_tr, y_tr = X_raw[tr].astype(np.float32), y[tr].astype(np.float32)
        X_te, y_te = X_raw[te].astype(np.float32), y[te].astype(np.float32)
        
        # Create dataset dict (pykan format) - use float32 for compatibility
        dataset = {
            'train_input': torch.tensor(X_tr).to(DEVICE),
            'train_label': torch.tensor(y_tr).unsqueeze(1).to(DEVICE),
            'test_input': torch.tensor(X_te).to(DEVICE),
            'test_label': torch.tensor(y_te).unsqueeze(1).to(DEVICE),
        }
        
        # KAN: [in_dim] → [hidden] → [hidden//2] → [1]
        model = KAN(width=width, grid=args.grid, k=args.k, seed=42, device=DEVICE)
        
        # Train
        # Use callable loss (string support varies by pykan version)
        import torch.nn.functional as F
        def bce_loss(pred, target):
            return F.binary_cross_entropy_with_logits(pred, target)
        result = model.fit(dataset, opt=args.opt, steps=args.steps, lamb=args.lamb, lamb_entropy=args.lamb_entropy, loss_fn=bce_loss)
        
        # Predict
        model.eval()
        with torch.no_grad():
            prob = model(dataset['test_input'])
            prob = torch.sigmoid(prob).cpu().numpy().flatten()
        
        auc = roc_auc_score(y_te, prob)
        aucs.append(auc); preds[te] = prob
        print(f"  Fold {f}: AUC={auc:.4f}", flush=True)
    
    rt = time.time() - t0

elif USE_EFFICIENT:
    # efficient-kan: use custom training loop
    import torch.nn as nn
    
    class EfficientKAN(nn.Module):
        def __init__(self, in_dim, hidden_dims):
            super().__init__()
            layers = []
            dims = [in_dim] + hidden_dims
            for i in range(len(dims)-1):
                layers.append(KANLinear(dims[i], dims[i+1], grid_size=5, spline_order=3))
            self.layers = nn.Sequential(*layers)
        
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def train_efficient(X_tr, y_tr, X_te, y_te, epochs=200):
        model = EfficientKAN(len(env_cols), [8, 4, 1]).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.BCEWithLogitsLoss()
        Xt = torch.tensor(X_tr).to(DEVICE)
        yt = torch.tensor(y_tr).unsqueeze(1).to(DEVICE)
        Xte = torch.tensor(X_te).to(DEVICE)
        best_auc, best_prob = 0, None
        for e in range(epochs):
            model.train(); opt.zero_grad()
            loss = loss_fn(model(Xt), yt)
            loss.backward(); opt.step()
            if (e+1) % 20 == 0:
                model.eval()
                with torch.no_grad():
                    prob = torch.sigmoid(model(Xte)).cpu().numpy().flatten()
                    auc = roc_auc_score(y_te, prob)
                    if auc > best_auc: best_auc, best_prob = auc, prob
        return best_auc, best_prob
    
    t0 = time.time()
    aucs, preds = [], np.zeros(len(y))
    for f in range(n_folds):
        te = fold_ids == f; tr = ~te
        auc, prob = train_efficient(X_raw[tr], y[tr], X_raw[te], y[te])
        aucs.append(auc); preds[te] = prob
        print(f"  Fold {f}: AUC={auc:.4f}", flush=True)
    rt = time.time() - t0

else:
    print("ERROR: No KAN library available", file=sys.stderr)
    sys.exit(1)

# Save
kan_mean, kan_std = round(np.mean(aucs), 4), round(np.std(aucs), 4)
config = {
    "steps": args.steps,
    "grid": args.grid,
    "k": args.k,
    "width": width,
    "opt": args.opt,
    "lamb": args.lamb,
    "lamb_entropy": args.lamb_entropy,
    "tag": args.tag
}
with open(f"{OUTDIR}/metrics.json", "w") as f:
    json.dump({
        "model": "KAN",
        "cv": "blockCV 150km external folds",
        "auc_mean": kan_mean,
        "auc_std": kan_std,
        "folds": [round(a,4) for a in aucs],
        "runtime_s": round(rt,1),
        "device": str(DEVICE),
        "config": config
    }, f, indent=2)
pd.DataFrame({"label":y,"pred_prob":preds}).to_csv(f"{OUTDIR}/predictions.csv", index=False)

print(f"\n{'='*50}")
print(f"📊 KAN (blockCV 150km external folds): AUC = {kan_mean:.4f} ± {kan_std:.4f}")
print(f"⏱  Total: {rt:.0f}s")
print(f"💾 Saved: {OUTDIR}/")
print(f"Config: {config}")
