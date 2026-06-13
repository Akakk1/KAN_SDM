#!/usr/bin/env python3
"""

from config import DATA_DIR, MODEL_DIR, RESULTS_DIR, TRAIN_CSV
Generate current-climate habitat suitability maps for Figure 3 (and as baseline for future projections).

Focuses on the KAN model (loaded from the canonical model_best checkpoint used for interpretability).
The 10 screened variables are used, with the exact StandardScaler fitted on the training data.

Outputs:
  - Results/maps/kan_current_suitability.tif   (GeoTIFF, same crs/res as input rasters)
  - Results/maps/kan_current_suitability.png   (quick preview)
  - Optionally a simple side-by-side placeholder if a MaxEnt raster is provided later.

This script is the first step toward proper spatial output. Once it works reliably for current rasters,
extending it to the Future/ SSP layers (ssp126/370/585) is straightforward (same variables, same scaler, same model).

Usage (from Program/):
    python 16_generate_current_suitability_maps.py

Requirements: rasterio, the current 10-var rasters in Data/History/, and model_best/ checkpoint.
"""

import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol
from rasterio.warp import Resampling
import torch
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from kan import KAN

# DATA_DIR provided by config.py
# RESULTS_DIR provided by config.py
MAPS_DIR = os.path.join(RESULTS_DIR, "maps")
os.makedirs(MAPS_DIR, exist_ok=True)

# The exact 10 screened variables in the order they appear in ginkgo_training_with_coords.csv
ENV_COLS = ['bio6','bio11','bio12','bio13','bio2','bio3','bio4','bio5','bio14','bio15']

# Current climate rasters (WorldClim 2.1 10 arcmin bioclimatics)
# All screened variables (bio6, bio11, bio12 ... bio15) are available as bioN.tif
BIO_FOLDER = os.path.join(DATA_DIR, "History", "wc2.1_10m_bio")

def get_raster_path(var):
    if var.startswith('bio'):
        return os.path.join(BIO_FOLDER, f"{var}.tif")
    else:
        raise ValueError(f"Unexpected var {var} - all current screened vars are bio*")

def load_full_data_for_scaler():
    """Load the training data to fit the exact same StandardScaler used for the KAN model."""
    csv_path = TRAIN_CSV
    df = pd.read_csv(csv_path)
    X = df[ENV_COLS].values.astype(np.float64)
    scaler = StandardScaler().fit(X)
    print(f"Scaler fitted on {len(X)} samples with vars: {ENV_COLS}")
    return scaler

def load_kan_model():
    """Load the canonical full-data interpretability KAN."""
    ckpt = os.path.join(SCRIPT_DIR, "model_best")
    model = KAN.loadckpt(ckpt)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"KAN model loaded from {ckpt} on {device}")
    return model, device

def predict_kan_on_raster(model, device, scaler):
    """
    Predict suitability on the current climate rasters.
    Returns a 2D array of predicted values (same shape as the rasters) + the raster profile.
    """
    # Read the first raster to get shape, transform, crs
    ref_path = get_raster_path(ENV_COLS[0])
    with rasterio.open(ref_path) as src:
        profile = src.profile.copy()
        height, width = src.shape
        transform = src.transform
        crs = src.crs

    print(f"Raster grid: {height} rows x {width} cols")

    # Prepare output array
    suitability = np.full((height, width), np.nan, dtype=np.float32)

    # Read all 10 bands into a (10, H, W) stack (this is small at 10')
    stack = np.zeros((len(ENV_COLS), height, width), dtype=np.float32)
    for i, var in enumerate(ENV_COLS):
        path = get_raster_path(var)
        with rasterio.open(path) as src:
            stack[i] = src.read(1).astype(np.float32)

    # Flatten for prediction (only valid pixels)
    flat = stack.reshape(len(ENV_COLS), -1).T   # (N_pixels, 10)
    valid_mask = ~np.isnan(flat).any(axis=1)
    X_valid = flat[valid_mask]

    if len(X_valid) == 0:
        print("WARNING: No valid pixels found.")
        return suitability, profile

    # Scale exactly as during training
    X_scaled = scaler.transform(X_valid).astype(np.float32)

    # Predict in batches (KAN forward is cheap)
    batch_size = 4096
    preds = np.zeros(len(X_valid), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(X_valid), batch_size):
            end = min(start + batch_size, len(X_valid))
            xb = torch.tensor(X_scaled[start:end]).to(device)
            out = torch.sigmoid(model(xb)).cpu().numpy().ravel()
            preds[start:end] = out

    # Put back into 2D grid
    suitability_flat = np.full(flat.shape[0], np.nan, dtype=np.float32)
    suitability_flat[valid_mask] = preds
    suitability = suitability_flat.reshape(height, width)

    print(f"Prediction done. Valid pixels: {valid_mask.sum()}/{len(valid_mask)}")
    print(f"Predicted range: {np.nanmin(suitability):.4f} - {np.nanmax(suitability):.4f}")

    return suitability, profile

def save_geotiff(arr, profile, out_path):
    profile.update(
        dtype=rasterio.float32,
        count=1,
        nodata=np.nan,
        compress='deflate'
    )
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(arr.astype(np.float32), 1)
    print(f"Saved GeoTIFF → {out_path}")

def quick_plot(arr, title, out_png):
    plt.figure(figsize=(8, 7))
    plt.imshow(arr, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(label='Predicted suitability')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Quick preview saved → {out_png}")

def main():
    print("=== Generating current suitability map (KAN) ===")
    scaler = load_full_data_for_scaler()
    model, device = load_kan_model()

    suitability, profile = predict_kan_on_raster(model, device, scaler)

    out_tif = os.path.join(MAPS_DIR, "kan_current_suitability.tif")
    save_geotiff(suitability, profile, out_tif)

    out_png = os.path.join(MAPS_DIR, "kan_current_suitability.png")
    quick_plot(suitability, "KAN current suitability (10-var model)", out_png)

    print("\nNext: We can now add the R maxnet map using a similar raster prediction step in R (terra + maxnet::predict).")
    print("Once both rasters exist, a proper Figure 3 side-by-side + difference plot can be generated.")
    print("After current maps work, extending the exact same pipeline to Data/Future/ SSP layers is trivial.")

if __name__ == "__main__":
    main()