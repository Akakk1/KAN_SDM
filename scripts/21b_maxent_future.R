#!/usr/bin/env Rscript
# 21b_maxent_future.R
# Predict R maxnet on a future climate stack for given ssp and period.
# Re-fits on full training data (same as current maps), then predicts on the future bioclim tif (19-band).
# Outputs a GeoTIFF in Results/maps/future/maxent_<ssp>_<period>.tif
#
# Usage: Rscript 21b_maxent_future.R --ssp 585 --period 2081-2100

library(maxnet)
library(terra)
library(optparse)

option_list <- list(
  make_option(c("--ssp"), type="character", default="585", help="e.g. 126, 370, 585"),
  make_option(c("--period"), type="character", default="2081-2100", help="e.g. 2021-2040, 2081-2100")
)
opt <- parse_args(OptionParser(option_list=option_list))

ssp <- paste0("ssp", opt$ssp)
period <- opt$period

SCRIPT_DIR <- dirname(normalizePath(sub("--file=", "", commandArgs(trailingOnly = FALSE)[grep("--file=", commandArgs(trailingOnly = FALSE))])))
if (length(SCRIPT_DIR) == 0) SCRIPT_DIR <- getwd()

DATA_DIR <- file.path(SCRIPT_DIR, "..", "data")  # data/ is at repo root
RESULTS_DIR <- file.path(SCRIPT_DIR, "Results")
MAPS_DIR <- file.path(RESULTS_DIR, "maps", "future")
dir.create(MAPS_DIR, showWarnings = FALSE, recursive = TRUE)

ENV_COLS <- c("bio6", "bio11", "bio12", "bio13", "bio2", "bio3", "bio4", "bio5", "bio14", "bio15")
BIO_BANDS <- setNames(as.integer(sub("bio", "", ENV_COLS)), ENV_COLS)  # bio6 -> 6 etc.

train_csv <- file.path(DATA_DIR, "ginkgo_training_with_coords.csv")
if (!file.exists(train_csv)) {
  train_csv <- file.path(SCRIPT_DIR, "..", "data", "ginkgo_training_with_coords.csv")
}
df <- read.csv(train_csv)
cat("Loaded training data:", nrow(df), "rows\n")

p <- df$label
data_env <- df[, ENV_COLS, drop = FALSE]

cat("Fitting full-data maxnet (regmult=1.0) ...\n")
mx_model <- maxnet(p, data_env, maxnet::maxnet.formula(p, data_env), regmult = 1.0)

# Future bioclim file (19-band)
bioc_name <- sprintf("wc2.1_10m_bioc_ACCESS-CM2_%s_%s.tif", ssp, period)
bioc_path <- file.path(DATA_DIR, "Future", ssp, "bio", bioc_name)
if (!file.exists(bioc_path)) {
  bioc_path <- file.path(file.path(SCRIPT_DIR, "..", "..", "data_external", "Future"), ssp, "bio", bioc_name)
}
if (!file.exists(bioc_path)) {
  stop("Future file not found: ", bioc_name)
}

cat("Reading future stack from", bioc_path, "\n")
future_bioc <- rast(bioc_path)  # 19 bands

# Build 10-var stack in correct order
stack_list <- list()
for (v in ENV_COLS) {
  b <- BIO_BANDS[[v]]
  stack_list[[v]] <- future_bioc[[b]]
}
future_stack <- rast(stack_list)
names(future_stack) <- ENV_COLS

cat("Predicting with maxnet (cloglog) on future stack ...\n")
suit <- predict(future_stack, mx_model, type = "cloglog", na.rm = TRUE)

out_tif <- file.path(MAPS_DIR, sprintf("maxent_%s_%s.tif", ssp, period))
writeRaster(suit, out_tif, overwrite = TRUE, gdal = c("COMPRESS=DEFLATE"))
cat("Saved", out_tif, "\n")

# Optional: clip to China here or at plot time (we do at plot time for consistency with current)
cat("Done for", ssp, period, "\n")