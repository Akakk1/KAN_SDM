#!/usr/bin/env Rscript
# Pairwise significance test (Delong) for all model AUCs
# Reads predictions.csv from each model and runs roc.test(..., method="delong")

library(pROC)
library(jsonlite)

SCRIPT_DIR <- getwd()
RESULTS_DIR <- file.path(SCRIPT_DIR, "Results")

models <- list(
    list(name = "MaxEnt (R maxnet)", file = "maxent_cb/predictions_R.csv"),
    list(name = "Random Forest",     file = "rf_cb/predictions.csv"),
    list(name = "XGBoost",           file = "xgb_cb/predictions.csv"),
    list(name = "MLP",               file = "mlp_cb/predictions.csv"),
    list(name = "KAN (best)",        file = "kan_cb_wider20_steps300_10var/predictions.csv")
)

# Load predictions
preds <- list()
cat("Loading predictions...\n")
for (m in models) {
    path <- file.path(RESULTS_DIR, m$file)
    if (!file.exists(path)) {
        cat(sprintf("  WARNING: %s not found, skipping\n", path))
        next
    }
    df <- read.csv(path)
    if (!("label" %in% names(df)) || !("pred_prob" %in% names(df))) {
        cat(sprintf("  WARNING: %s missing label/pred_prob columns\n", path))
        next
    }
    preds[[length(preds) + 1]] <- list(
        name = m$name,
        label = df$label,
        prob = df$pred_prob
    )
    cat(sprintf("  %s: %d samples\n", m$name, length(df$label)))
}

n <- length(preds)
if (n < 2) {
    cat("ERROR: need at least 2 models\n")
    quit(status=1)
}

# Verify all models use the same label order
for (i in 2:n) {
    if (any(preds[[i]]$label != preds[[1]]$label)) {
        cat("ERROR: label mismatch between models\n")
        quit(status=1)
    }
}

# Build ROC objects
cat("\nComputing AUCs and Delong tests...\n")
rocs <- list()
aucs <- c()
for (i in 1:n) {
    rocs[[i]] <- roc(preds[[i]]$label, preds[[i]]$prob, quiet=TRUE)
    aucs <- c(aucs, round(auc(rocs[[i]]), 4))
    cat(sprintf("  %s: AUC=%.4f\n", preds[[i]]$name, aucs[i]))
}

# Pairwise Delong test
cat("\nPairwise Delong test (z-statistic, p-value):\n\n")
results <- data.frame(matrix(ncol=n, nrow=n), row.names=sapply(preds, `[[`, "name"))
colnames(results) <- sapply(preds, `[[`, "name")

pval_matrix <- data.frame(matrix(ncol=n, nrow=n), row.names=sapply(preds, `[[`, "name"))
colnames(pval_matrix) <- sapply(preds, `[[`, "name")

for (i in 1:n) {
    for (j in 1:n) {
        if (i >= j) {
            results[i,j] <- NA
            pval_matrix[i,j] <- NA
            next
        }
        test <- roc.test(rocs[[i]], rocs[[j]], method="delong")
        z <- round(test$statistic, 3)
        p <- round(test$p.value, 4)
        results[i,j] <- z
        pval_matrix[i,j] <- p
        sig <- ifelse(p < 0.05, "*", "")
        cat(sprintf("  %s vs %s: z=%.3f, p=%.4f%s\n",
                    preds[[i]]$name, preds[[j]]$name, z, p, sig))
    }
}

# Save
out <- list(
    aucs = setNames(aucs, sapply(preds, `[[`, "name")),
    delong_z = results,
    delong_p = pval_matrix,
    significance_level = 0.05
)
write_json(out, file.path(RESULTS_DIR, "significance_tests.json"), auto_unbox=TRUE, pretty=TRUE, digits=4)
cat(sprintf("\nSaved to %s/significance_tests.json\n", RESULTS_DIR))
