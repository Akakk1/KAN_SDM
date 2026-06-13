#!/usr/bin/env Rscript
# Real R maxnet baseline — blockCV 150km external folds
# Uses R maxnet package (the real MaxEnt implementation) with the exact
# same data and blockCV fold assignments as the Python baselines.

library(glmnet)
library(maxnet)
library(pROC)
library(jsonlite)

# When run as Rscript from Program/, use current dir
SCRIPT_DIR <- getwd()
DATA_DIR <- file.path(SCRIPT_DIR, "..", "data")  # data/ is at repo root
OUTDIR <- file.path(SCRIPT_DIR, "Results/maxent_cb")
dir.create(OUTDIR, recursive=TRUE, showWarnings=FALSE)

train <- read.csv(file.path(DATA_DIR, "ginkgo_training_with_coords.csv"))
# Dynamically select environmental columns (exclude label, fold, coords)
non_env <- c('label', 'fold', 'decimalLatitude', 'decimalLongitude')
env_cols <- setdiff(names(train), non_env)
env <- train[, env_cols]
label <- train$label
fold_ids <- train$fold
n_folds <- length(unique(fold_ids))

cat(sprintf("=== R maxnet Baseline (blockCV 150km folds) ===\n"))
cat(sprintf("Samples: %d, Presence: %d, Folds: %d, Env vars: %d\n", nrow(train), sum(label), n_folds, length(env_cols)))
cat(sprintf("Env vars: %s\n", paste(env_cols, collapse=", ")))

aucs <- c()
preds <- rep(0, nrow(train))

for (f in 0:(n_folds-1)) {
    te <- fold_ids == f
    tr <- !te
    cat(sprintf("Fold %d: train=%d (P=%d), test=%d (P=%d)\n",
                f, sum(tr), sum(label[tr]==1), sum(te), sum(label[te]==1)))
    
    # Set high maxit before each fold (glmnet.control is global)
    glmnet.control(maxit=10000)
    model <- maxnet(p = label[tr], data = env[tr,], regmult = 1.0)
    
    prob <- predict(model, env[te,], type="cloglog")
    preds[te] <- prob
    auc <- roc(label[te], prob, quiet=TRUE)$auc
    aucs <- c(aucs, auc)
    cat(sprintf("  AUC=%.4f, nonzero=%d\n", auc, length(model$betas)))
}

mean_auc <- mean(aucs)
std_auc <- sd(aucs)

metrics <- list(
    model = "MaxEnt (R maxnet)",
    cv = "blockCV 150km external folds",
    auc_mean = round(mean_auc, 4),
    auc_std = round(std_auc, 4),
    folds = round(aucs, 4)
)
write_json(metrics, file.path(OUTDIR, "metrics_R.json"), auto_unbox=TRUE)
write.csv(data.frame(label=label, pred_prob=preds),
          file.path(OUTDIR, "predictions_R.csv"), row.names=FALSE)

cat(sprintf("\nR maxnet AUC: %.4f ± %.4f\n", mean_auc, std_auc))
cat(sprintf("Saved to %s/\n", OUTDIR))
