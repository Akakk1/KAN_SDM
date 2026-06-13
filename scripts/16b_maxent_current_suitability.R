#!/usr/bin/env Rscript
# 16b_maxent_current_suitability.R
#
# Generate current-climate suitability map for the canonical R maxnet model
# (the "real" MaxEnt baseline) on the screened 10-variable set.
#
# This mirrors the logic of 16_generate_current_suitability_maps.py for KAN,
# so that Figure 3 can be a fair side-by-side.
#
# Outputs:
#   Results/maps/maxent_current_suitability.tif
#
# Requirements:
#   - R packages: maxnet, terra
#   - Run from the Program/ directory (or adjust paths)
#
# The model is fitted on the FULL training data (no CV split) using the same
# 10 variables and the same regmult=1.0 as the baseline in 04b_maxent_R.R.
#
# This is the reproducible way to get the exact map corresponding to the
# reported R maxnet AUC 0.9168.

library(maxnet)
library(terra)

SCRIPT_DIR <- dirname(normalizePath(sub("--file=", "", commandArgs(trailingOnly = FALSE)[grep("--file=", commandArgs(trailingOnly = FALSE))])))
if (length(SCRIPT_DIR) == 0) SCRIPT_DIR <- getwd()

DATA_DIR <- file.path(SCRIPT_DIR, "..", "data")  # data/ is at repo root
RESULTS_DIR <- file.path(SCRIPT_DIR, "Results")
MAPS_DIR <- file.path(RESULTS_DIR, "maps")
dir.create(MAPS_DIR, showWarnings = FALSE, recursive = TRUE)

# Exact 10 screened variables (must match the Python side and the training CSV)
ENV_COLS <- c("bio6", "bio11", "bio12", "bio13", "bio2", "bio3", "bio4", "bio5", "bio14", "bio15")

# Training data (the canonical file with folds, but we use full data for the map)
train_csv <- file.path(DATA_DIR, "ginkgo_training_with_coords.csv")
if (!file.exists(train_csv)) stop("Cannot find ", train_csv)

df <- read.csv(train_csv)
cat("Loaded training data:", nrow(df), "rows\n")

# Presence / background for maxnet (p = 1 for presence, 0 for background)
p <- df$label
data_env <- df[, ENV_COLS, drop = FALSE]

# Fit the canonical full-data maxnet model (regmult = 1.0, same as the reported baseline)
cat("Fitting full-data maxnet model (regmult=1.0) on 10 variables...\n")
mx_model <- maxnet(p, data_env, maxnet::maxnet.formula(p, data_env), regmult = 1.0)
cat("maxnet model fitted.\n")

# Current climate rasters (same source as used for training data extraction)
# All bio variables are in the History bio folder as bioN.tif
bio_folder <- file.path(DATA_DIR, "History", "wc2.1_10m_bio")

# Build a SpatRaster stack with exactly the 10 variables in the correct order
rasters <- list()
for (v in ENV_COLS) {
  tif <- file.path(bio_folder, paste0(v, ".tif"))
  if (!file.exists(tif)) stop("Missing raster: ", tif)
  rasters[[v]] <- rast(tif)
}
env_stack <- rast(rasters)
names(env_stack) <- ENV_COLS
cat("Raster stack created:", names(env_stack), "\n")
cat("  Resolution:", res(env_stack)[1], "degrees\n")

# Predict with cloglog (standard for the reported MaxEnt baseline)
cat("Predicting on raster stack (type = 'cloglog')...\n")
suitability <- predict(env_stack, mx_model, type = "cloglog", na.rm = TRUE)

# Write GeoTIFF
out_tif <- file.path(MAPS_DIR, "maxent_current_suitability.tif")
writeRaster(suitability, out_tif, overwrite = TRUE, gdal = c("COMPRESS=DEFLATE"))
cat("Saved GeoTIFF ->", out_tif, "\n")

# Quick stats
vals <- values(suitability, na.rm = TRUE)
cat("MaxEnt suitability range:", min(vals), "-", max(vals), "\n")
cat("Mean:", mean(vals), "\n")

cat("\nDone. Now run the Python plotting script to produce the publication Figure 3.\n")