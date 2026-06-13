"""
Central path configuration for KAN_Ginkgo.
Import this in all scripts to avoid hardcoded absolute paths.

Usage:
    from config import DATA_DIR, MODEL_DIR, RESULTS_DIR, EXTERNAL_DATA, OUTPUT_DIR

The repo structure is:
    KAN_Ginkgo/
    ├── data/          ← committed small files (CSV, geojson, ...)
    ├── model/         ← canonical KAN checkpoint
    ├── results/       ← key metrics
    └── scripts/       ← all .py / .R scripts (this file lives here)

External (large) data is NOT in the repo. Users should set the environment
variable KAN_GINKGO_DATA to point to the directory holding:
    KAN_GINKGO_DATA/
    ├── History/           ← WorldClim 2.1 BIO tifs (e.g. wc2.1_10m_bio_1.tif)
    └── Future/            ← future climate layers (SSP dirs with bio/ subdirs)

If KAN_GINKGO_DATA is not set, scripts will look for a sibling directory
named `data_external/` alongside the repo root.
"""

import os
import sys

# ── Repo root (parent of scripts/) ──────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

# ── Committed data directories (inside the repo) ────────────────────────
DATA_DIR = os.path.join(REPO_ROOT, "data")
MODEL_DIR = os.path.join(REPO_ROOT, "model")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

# Canonical files
TRAIN_CSV = os.path.join(DATA_DIR, "ginkgo_training_with_coords.csv")
BOUNDARY_GEOJSON = os.path.join(DATA_DIR, "boundaries", "china_admin_union.geojson")
MODEL_CHECKPOINT = os.path.join(MODEL_DIR, "model_best_config.yml")  # loadckpt uses dir

# ── External (large) data ───────────────────────────────────────────────
# WorldClim tifs, Future climate layers — too large for GitHub.
# Set env var KAN_GINKGO_DATA or create data_external/ next to the repo.
_explicit = os.environ.get("KAN_GINKGO_DATA", "")
if _explicit and os.path.isdir(_explicit):
    EXTERNAL_DATA = _explicit
else:
    _candidate = os.path.join(os.path.dirname(REPO_ROOT), "data_external")
    if os.path.isdir(_candidate):
        EXTERNAL_DATA = _candidate
    else:
        EXTERNAL_DATA = ""  # user must configure

HISTORY_DIR = os.path.join(EXTERNAL_DATA, "History") if EXTERNAL_DATA else ""
FUTURE_DIR = os.path.join(EXTERNAL_DATA, "Future") if EXTERNAL_DATA else ""

# ── Output directories ──────────────────────────────────────────────────
# Figures and intermediate results that aren't committed.
OUTPUT_DIR = os.path.join(REPO_ROOT, "output")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
MAPS_DIR = os.path.join(OUTPUT_DIR, "maps")
KAN_RESPONSE_DIR = os.path.join(OUTPUT_DIR, "kan_response_data")

# Ensure output directories exist
for _d in [OUTPUT_DIR, FIGURES_DIR, MAPS_DIR, KAN_RESPONSE_DIR]:
    os.makedirs(_d, exist_ok=True)


# ── Validation ──────────────────────────────────────────────────────────
def validate():
    """Print config and check that critical paths exist."""
    print(f"REPO_ROOT      = {REPO_ROOT}")
    print(f"DATA_DIR       = {DATA_DIR}       {'✓' if os.path.isdir(DATA_DIR) else '✗ MISSING'}")
    print(f"MODEL_DIR      = {MODEL_DIR}      {'✓' if os.path.isdir(MODEL_DIR) else '✗ MISSING'}")
    print(f"RESULTS_DIR    = {RESULTS_DIR}    {'✓' if os.path.isdir(RESULTS_DIR) else '✗ MISSING'}")
    print(f"TRAIN_CSV      = {TRAIN_CSV}  {'✓' if os.path.isfile(TRAIN_CSV) else '✗ MISSING'}")
    print(f"BOUNDARY       = {BOUNDARY_GEOJSON}  {'✓' if os.path.isfile(BOUNDARY_GEOJSON) else '✗ MISSING'}")
    print(f"MODEL_CKPT     = {MODEL_CHECKPOINT}  {'✓' if os.path.isfile(MODEL_CHECKPOINT) else '✗ MISSING'}")
    print(f"EXTERNAL_DATA  = {EXTERNAL_DATA or '(not set — set KAN_GINKGO_DATA env var or create data_external/ next to repo)'}")
    print(f"HISTORY_DIR    = {HISTORY_DIR or '✗ — WorldClim tifs go here'}")
    print(f"FUTURE_DIR     = {FUTURE_DIR or '✗ — future climate layers go here'}")
    print(f"OUTPUT_DIR     = {OUTPUT_DIR}")
    print(f"FIGURES_DIR    = {FIGURES_DIR}")
    print(f"MAPS_DIR       = {MAPS_DIR}")
    print()

if __name__ == "__main__":
    validate()
