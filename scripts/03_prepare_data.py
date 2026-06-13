#!/usr/bin/env python3
"""Variable screening (Pearson |r|>0.8) + background point generation + final training data — v2"""
import pandas as pd
import numpy as np
import rasterio, os, csv
from rasterio.transform import rowcol

from config import DATA_DIR, TRAIN_CSV

# DATA_DIR provided by config.py
ENV_CSV = os.path.join(DATA_DIR, "ginkgo_envdata.csv")
BIO_DIR = os.path.join(DATA_DIR, "History/wc2.1_10m_bio")
ELEV_FILE = os.path.join(DATA_DIR, "History/wc2.1_10m_elev/wc2.1_10m_elev.tif")

# ====== 1. Load data ======
df = pd.read_csv(ENV_CSV).dropna()
env_cols = [c for c in df.columns if c.startswith("bio") or c == "elev"]
print(f"Presence: {len(df):,}, vars: {len(env_cols)}")

# ====== 2. Pearson pre-screening (|r|>0.8) ======
corr = df[env_cols].corr().abs()
dropped = set()
pairs = []

# Process from highest to lowest correlation
for i in range(len(env_cols)):
    for j in range(i+1, len(env_cols)):
        if env_cols[i] not in dropped and env_cols[j] not in dropped:
            if corr.iloc[i, j] > 0.8:
                pairs.append((env_cols[i], env_cols[j], corr.iloc[i, j]))

pairs.sort(key=lambda x: -x[2])

print(f"\n=== Highly correlated pairs (|r|>0.8) ===")
# From each pair, drop one: prefer keeping more general BIO variables
# Rule: prefer lower bio numbers (bio1-11 are temperature, bio12-19 are precipitation)
for a, b, r in pairs:
    if a in dropped or b in dropped:
        continue
    # Drop the one with higher bio number, or the one more dependent on others
    a_has_others = sum(1 for p in pairs if a in p[:2]) - 1
    b_has_others = sum(1 for p in pairs if b in p[:2]) - 1
    if b_has_others > a_has_others:
        dropped.add(b)
        print(f"  {a} <-> {b}: r={r:.3f}  -> drop {b}")
    else:
        dropped.add(a)
        print(f"  {a} <-> {b}: r={r:.3f}  -> drop {a}")

KEEP = [c for c in env_cols if c not in dropped]
print(f"\nKept: {KEEP} ({len(KEEP)} vars)")
print(f"\nPost-screening correlation (max |r|): {df[KEEP].corr().abs().where(lambda x: x<1).max().max():.3f}")

# ====== 3. Background points ======
ref_tif = os.path.join(BIO_DIR, "bio1.tif")
N_BG = 10000
np.random.seed(42)

candidates_lon = np.random.uniform(-180, 180, N_BG * 5)
candidates_lat = np.random.uniform(-60, 75, N_BG * 5)

bg_lats, bg_lons = [], []
with rasterio.open(ELEV_FILE) as elev_src:
    nd = elev_src.nodata
    for i in range(len(candidates_lat)):
        r, c = rowcol(elev_src.transform, candidates_lon[i], candidates_lat[i])
        if 0 <= r < elev_src.height and 0 <= c < elev_src.width:
            v = elev_src.read(1)[r, c]
            if v != nd and v > -1e20 and v > -500:
                bg_lats.append(candidates_lat[i])
                bg_lons.append(candidates_lon[i])
                if len(bg_lats) >= N_BG:
                    break

print(f"\nBackground points: {len(bg_lats):,}")

# ====== 4. Extract background environment values ======
tif_files = {}
for v in KEEP:
    if v == "elev":
        tif_files[v] = ELEV_FILE
    else:
        tif_files[v] = os.path.join(BIO_DIR, f"{v}.tif")

bg_env = {}
for v in KEEP:
    vals = []
    with rasterio.open(tif_files[v]) as src:
        nd = src.nodata
        for lon, lat in zip(bg_lons, bg_lats):
            r, c = rowcol(src.transform, lon, lat)
            val = src.read(1)[r, c]
            if np.isnan(val) or val == nd or abs(val) > 1e20:
                vals.append(np.nan)
            else:
                vals.append(float(val))
        bg_env[v] = vals

# ====== 5. Merge and save ======
presence = df[KEEP].copy()
presence["label"] = 1

bg_df = pd.DataFrame(bg_env).dropna()
bg_df["label"] = 0

final = pd.concat([presence, bg_df], ignore_index=True).dropna()

outfile = f"{DATA_DIR}/ginkgo_training_data.csv"
final.to_csv(outfile, index=False)

print(f"\n✅ {outfile}")
print(f"   Presence: {len(presence):,}")
print(f"   Background: {len(bg_df):,}")
print(f"   Total samples: {len(final):,}")
print(f"   Variables ({len(KEEP)}): {', '.join(KEEP)}")
print(f"\n=== Presence vs Background means ===")
for v in KEEP:
    pm, bm = presence[v].mean(), bg_df[v].mean()
    print(f"  {v:6s}: P={pm:8.1f}  BG={bm:8.1f}  Δ={pm-bm:+8.1f}")
