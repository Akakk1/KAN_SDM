# generate_spatial_folds.R
# Use blockCV::cv_spatial() to generate spatially blocked folds for China (incl. HK/MO/TW)
# Run this in R after having a points CSV (lat, lon, label, id optional).
#
# Requirements:
#   install.packages(c("blockCV", "sf", "dplyr", "ggplot2"))
#
# Usage example:
#   Rscript Program/R/generate_spatial_folds.R \
#     --input Program/Data/ginkgo_occurrence.csv \
#     --output Program/Data/folds_prelim_blockcv.csv \
#     --block_size 200000 \   # meters (200 km)
#     --k 5 \
#     --selection "checkerboard"
#
# Outputs:
#   - folds_prelim_blockcv.csv   (input columns + fold)
#   - folds_prelim_blockcv_fold_summary.txt
#   - folds_prelim_blockcv_map.png  (diagnostic plot + points)

library(blockCV)
library(sf)
library(dplyr)
library(optparse)
library(ggplot2)  # for saving fold map PNG (available in this env)

option_list <- list(
  make_option(c("-i", "--input"), type="character", default=NULL,
              help="Input CSV with columns: decimalLatitude, decimalLongitude, label (and optional id)"),
  make_option(c("-o", "--output"), type="character", default="folds.csv",
              help="Output CSV with fold assignments"),
  make_option(c("-b", "--block_size"), type="integer", default=200000,
              help="Block size in meters (e.g. 100000-300000 for 100-300 km)"),
  make_option(c("-k", "--k"), type="integer", default=5,
              help="Number of folds"),
  make_option(c("-s", "--selection"), type="character", default="checkerboard",
              help="Selection method: 'checkerboard' or 'systematic'"),
  make_option(c("--seed"), type="integer", default=42,
              help="Random seed for reproducibility")
)

opt <- parse_args(OptionParser(option_list=option_list))

if (is.null(opt$input)) {
  stop("Input file is required (--input)")
}

set.seed(opt$seed)

# Read points
points_df <- read.csv(opt$input)

# Auto-add label=1 if missing (common for presence-only occurrence data)
if (!"label" %in% names(points_df)) {
  points_df$label <- 1
  cat("No 'label' column found -- assuming all points are presences (label=1)\n")
}

# Create sf object (WGS84)
points_sf <- st_as_sf(points_df,
                      coords = c("decimalLongitude", "decimalLatitude"),
                      crs = 4326)

cat("Loaded", nrow(points_sf), "points.\n")

# For strict China admin boundary (incl. HK/MO/TW), user should provide a polygon.
# Example: load a China level-0 shapefile (GADM or equivalent) that includes Taiwan claim.
# If no polygon provided, we proceed with the points as-is (bbox was already applied in download).
# Later we can add: points_sf <- st_intersection(points_sf, china_polygon)

# Generate spatial folds
# blockCV cv_spatial works with sf points + response column
folds <- cv_spatial(
  x = points_sf,
  column = "label",           # presence/background column
  size = opt$block_size,      # in meters
  k = opt$k,
  selection = opt$selection,  # "checkerboard" or "systematic"
  seed = opt$seed,
  progress = TRUE,
  plot = FALSE                # avoid interactive device issues in Rscript; we save PNG below
)

# Attach fold IDs
points_df$fold <- folds$folds_ids

# Save CSV
write.csv(points_df, opt$output, row.names = FALSE)
cat("Folds saved to", opt$output, "\n")
cat("Fold distribution:\n")
print(table(points_df$fold, points_df$label))

# Also save a simple summary
summary_file <- sub("\\.csv$", "_fold_summary.txt", opt$output)
sink(summary_file)
print(table(points_df$fold, points_df$label))
cat("\nBlock size (m):", opt$block_size, "\n")
cat("k:", opt$k, "\n")
cat("selection:", opt$selection, "\n")
sink()
cat("Summary saved to", summary_file, "\n")

# Save visualization PNG (uses ggplot2 which is pre-installed)
map_file <- sub("\\.csv$", "_map.png", opt$output)
p <- plot(folds)
p <- p + geom_sf(data = points_sf, aes(color = factor(label)), size = 0.7, alpha = 0.75, inherit.aes = FALSE) +
  labs(color = "label (1=presence)") +
  theme_minimal(base_size = 10) +
  ggtitle(paste0("Spatial folds (blockCV ", opt$selection, ", size=", round(opt$block_size/1000), "km, k=", opt$k, ")"))
ggsave(map_file, p, width = 9, height = 7, dpi = 150)
cat("Fold map saved to", map_file, "\n")
