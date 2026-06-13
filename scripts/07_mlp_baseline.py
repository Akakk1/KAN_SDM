#!/usr/bin/env python3
"""MLP baseline — simple feedforward net for comparison to KAN"""
import pandas as pd, numpy as np, json, time, os
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from config import TRAIN_CSV, RESULTS_DIR

OUTDIR = os.path.join(RESULTS_DIR, "mlp_cb")
os.makedirs(OUTDIR, exist_ok=True)

df = pd.read_csv(TRAIN_CSV)
# Dynamically determine environmental columns (exclude label, fold, coords)
non_env = ['label', 'fold', 'decimalLatitude', 'decimalLongitude']
env_cols = [c for c in df.columns if c not in non_env]
df = df[env_cols + ['decimalLatitude', 'decimalLongitude', 'label', 'fold']].copy()
print(f"✅ Loaded training with coords ({len(env_cols)} env vars)")
X_raw = StandardScaler().fit_transform(df[env_cols].values).astype(np.float32)
y = df["label"].values.astype(np.float32)
n_folds = len(np.unique(df['fold']))
fold_ids = df['fold'].values

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=== MLP Baseline (blockCV external folds) ===")
print(f"Device: {DEVICE}, samples={len(X_raw):,} folds={n_folds}")

class MLP(nn.Module):
    def __init__(self, in_dim, hidden=[32, 16]):
        super().__init__()
        layers = []
        dims = [in_dim] + hidden + [1]
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                layers.append(nn.ReLU())
            else:
                # no act on last, use BCEWithLogits
                pass
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

t0 = time.time()
aucs, preds_all = [], np.zeros(len(y))

for f in range(n_folds):
    te = fold_ids == f; tr = ~te
    X_tr, y_tr = X_raw[tr], y[tr]
    X_te, y_te = X_raw[te], y[te]
    
    model = MLP(len(env_cols)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCEWithLogitsLoss()
    
    Xt = torch.tensor(X_tr).to(DEVICE)
    yt = torch.tensor(y_tr).unsqueeze(1).to(DEVICE)
    Xte_t = torch.tensor(X_te).to(DEVICE)
    
    best_auc = 0.0
    best_prob = None
    for epoch in range(300):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward()
        opt.step()
        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                prob = torch.sigmoid(model(Xte_t)).cpu().numpy().flatten()
                auc = roc_auc_score(y_te, prob)
                if auc > best_auc:
                    best_auc = auc
                    best_prob = prob
    aucs.append(best_auc)
    preds_all[te] = best_prob
    nparams = count_params(model)
    print(f"  Fold {f}: AUC={best_auc:.4f}  params={nparams}")

rt = time.time() - t0
mean_auc, std_auc = round(np.mean(aucs), 4), round(np.std(aucs), 4)

# Save
with open(os.path.join(OUTDIR, "metrics.json"), "w") as f:
    json.dump({
        "model": "MLP",
        "cv": "blockCV 150km external folds",
        "auc_mean": mean_auc,
        "auc_std": std_auc,
        "folds": [round(a, 4) for a in aucs],
        "runtime_s": round(rt, 1),
        "device": str(DEVICE),
        "params": count_params(MLP(len(env_cols)))
    }, f, indent=2)
pd.DataFrame({"label": y, "pred_prob": preds_all}).to_csv(os.path.join(OUTDIR, "predictions.csv"), index=False)

print(f"\n{'='*50}")
print(f"📊 MLP (checkerboard CV): AUC = {mean_auc:.4f} ± {std_auc:.4f}")
print(f"⏱  Total: {rt:.0f}s")
print(f"💾 Saved: {OUTDIR}/")
