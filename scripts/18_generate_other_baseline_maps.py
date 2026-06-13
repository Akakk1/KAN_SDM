#!/usr/bin/env python3
"""

from config import BOUNDARY_GEOJSON, DATA_DIR, RESULTS_DIR, TRAIN_CSV
Generate current suitability maps (GeoTIFF) for the other baselines:
Random Forest, XGBoost, MLP.

This mirrors the approach used for KAN (full-data model + same 10-var scaler + China clip if desired).

Outputs in Results/maps/:
  - rf_current_suitability.tif
  - xgb_current_suitability.tif
  - mlp_current_suitability.tif

Also saves the full-data fitted models (for reproducibility):
  Results/maps/rf_full_model.joblib
  Results/maps/xgb_full_model.joblib
  Results/maps/mlp_full_model.pt   (state_dict + config)

Uses the exact hyper-parameters from the reported CV runs (05_rf_xgb.py, 07_mlp_baseline.py).

Run under kan_spe env (has sklearn, xgboost, torch, rasterio, etc.).
"""

import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask as rio_mask
import joblib
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

# DATA_DIR provided by config.py
# RESULTS_DIR provided by config.py
MAPS_DIR = os.path.join(RESULTS_DIR, "maps")
os.makedirs(MAPS_DIR, exist_ok=True)

# Exact 10 screened variables (order must match training data columns)
ENV_COLS = ['bio6','bio11','bio12','bio13','bio2','bio3','bio4','bio5','bio14','bio15']

BIO_FOLDER = os.path.join(DATA_DIR, "History", "wc2.1_10m_bio")
TRAIN_CSV = TRAIN_CSV

# Boundary for optional clipping (same as Figure 3)
BOUNDARY = BOUNDARY_GEOJSON

# Hyperparams from the reported runs
RF_PARAMS = dict(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
XGB_PARAMS = dict(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, verbosity=0)
MLP_HIDDEN = [32, 16]

class MLP(nn.Module):
    def __init__(self, in_dim, hidden=MLP_HIDDEN):
        super().__init__()
        dims = [in_dim] + hidden + [1]
        layers = []
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def load_training_data():
    df = pd.read_csv(TRAIN_CSV)
    X = df[ENV_COLS].values.astype(np.float64)
    y = df["label"].values.astype(np.float32)
    scaler = StandardScaler().fit(X)
    X_std = scaler.transform(X).astype(np.float32)
    print(f"Training data: {len(y)} samples, env={ENV_COLS}")
    return X_std, y, scaler

def fit_and_save_rf_xgb(X, y, scaler):
    print("\n=== Fitting full-data Random Forest ===")
    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X, y)
    joblib.dump(rf, os.path.join(MAPS_DIR, "rf_full_model.joblib"))
    print("Saved rf_full_model.joblib")

    print("\n=== Fitting full-data XGBoost ===")
    xgb = XGBClassifier(**XGB_PARAMS)
    xgb.fit(X, y)
    joblib.dump(xgb, os.path.join(MAPS_DIR, "xgb_full_model.joblib"))
    print("Saved xgb_full_model.joblib")

    return rf, xgb

def fit_and_save_mlp(X, y, scaler):
    print("\n=== Fitting full-data MLP ===")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(len(ENV_COLS)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCEWithLogitsLoss()

    Xt = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    yt = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(DEVICE)

    # Simple full-data training (more epochs than CV folds for stability)
    model.train()
    for epoch in range(100):
        opt.zero_grad()
        out = model(Xt)
        loss = loss_fn(out, yt)
        loss.backward()
        opt.step()
        if (epoch+1) % 20 == 0:
            print(f"  epoch {epoch+1}/100 loss={loss.item():.4f}")

    torch.save({
        "state_dict": model.state_dict(),
        "hidden": MLP_HIDDEN,
        "in_dim": len(ENV_COLS)
    }, os.path.join(MAPS_DIR, "mlp_full_model.pt"))
    print("Saved mlp_full_model.pt")
    return model, DEVICE

def predict_sklearn_on_raster(model, scaler, out_name, clip_to_china=True):
    """Predict on current climate rasters and write tif."""
    ref_path = os.path.join(BIO_FOLDER, f"{ENV_COLS[0]}.tif")
    with rasterio.open(ref_path) as src:
        profile = src.profile.copy()
        height, width = src.shape
        transform = src.transform

    # Stack the 10 bands
    stack = np.zeros((len(ENV_COLS), height, width), dtype=np.float32)
    for i, var in enumerate(ENV_COLS):
        path = os.path.join(BIO_FOLDER, f"{var}.tif")
        with rasterio.open(path) as src:
            stack[i] = src.read(1).astype(np.float32)

    flat = stack.reshape(len(ENV_COLS), -1).T
    valid = ~np.isnan(flat).any(axis=1)
    Xv = flat[valid]

    # Scale + predict
    Xs = scaler.transform(Xv).astype(np.float32)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(Xs)[:, 1]
    else:
        with torch.no_grad():
            t = torch.tensor(Xs, dtype=torch.float32).to(next(model.parameters()).device)
            probs = torch.sigmoid(model(t)).cpu().numpy().ravel()

    out = np.full(flat.shape[0], np.nan, dtype=np.float32)
    out[valid] = probs
    out2d = out.reshape(height, width)

    # Optional clip to China (recommended for consistency with Fig3)
    if clip_to_china and os.path.exists(BOUNDARY):
        import geopandas as gpd
        gdf = gpd.read_file(BOUNDARY)
        geom = [gdf.geometry.iloc[0]]
        with rasterio.open(ref_path) as src:
            out_image, out_transform = rio_mask(src, geom, crop=True, nodata=np.nan, filled=False)
            # We need to re-rasterize the predictions onto the cropped grid.
            # Simpler: since we already have full out2d, re-open and mask the array.
            # For simplicity here we clip the 2d array using the same logic.
            # (The previous KAN generator did full predict then we clip in plot; here we can do full)
            # To keep identical to previous workflow, we write the full and let plot clip.
            pass   # write full for now; clipping done at plot time for consistency

    profile.update(dtype=rasterio.float32, count=1, nodata=np.nan, compress='deflate')
    out_tif = os.path.join(MAPS_DIR, f"{out_name}_current_suitability.tif")
    with rasterio.open(out_tif, 'w', **profile) as dst:
        dst.write(out2d.astype(np.float32), 1)
    print(f"Saved {out_tif}")

    # Also write a quick preview png (raw, unclipped view)
    import matplotlib.pyplot as plt
    plt.imsave(os.path.join(MAPS_DIR, f"{out_name}_current_suitability.png"), out2d, cmap='YlGnBu', vmin=0, vmax=1)
    print(f"  preview png saved")

def main():
    print("=== Generating maps for RF / XGB / MLP (full-data versions) ===")
    X, y, scaler = load_training_data()

    rf, xgb = fit_and_save_rf_xgb(X, y, scaler)
    mlp, device = fit_and_save_mlp(X, y, scaler)

    print("\n=== Predicting on current rasters ===")
    # Note: for exact consistency with KAN/MaxEnt tifs, we predict full then clip at plot time.
    # Here we predict full (like the KAN 16 script originally did).
    predict_sklearn_on_raster(rf, scaler, "rf", clip_to_china=False)
    predict_sklearn_on_raster(xgb, scaler, "xgb", clip_to_china=False)
    predict_sklearn_on_raster(mlp, scaler, "mlp", clip_to_china=False)

    print("\nDone. Now you can run a plotting script (e.g. extend 17 or new 19) to make a multi-panel figure for RF/XGB/MLP (+ optionally KAN/Maxnet for reference).")

if __name__ == "__main__":
    main()