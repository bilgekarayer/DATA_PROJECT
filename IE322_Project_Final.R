# ===================================================================
# IE 322 - Exploratory Data Analytics - Final Project
# Group Members: [Group Members - Fill in]
# Dataset: Student Performance Data Set (Cortez & Silva, 2008)
# Course: Portuguese Language (n = 649)
# Date: May 2026
# ===================================================================
#
# PROBLEM DEFINITION:
# We predict the BINARY pass/fail status (G3 >= 10) of students in
# Portuguese language using ONLY social/demographic/family variables,
# explicitly EXCLUDING the first-period (G1) and second-period (G2)
# grades to avoid trivial autocorrelation / data leakage. This
# corresponds to the 'Setup C' input scenario of Cortez & Silva (2008),
# but with Logistic Regression and k-NN (course methods) instead of
# the original DT/RF/NN/SVM.
#
# EVALUATION PROTOCOL (read this before Section 4):
#   1. ONE stratified 80/20 train/test split, done at the start.
#   2. ALL hyperparameter tuning (threshold for logistic, k for kNN)
#      is performed via 10-fold stratified CV INSIDE the training set.
#   3. For logistic regression, stepwise variable selection is re-run
#      INSIDE each CV fold so the CV error estimate is honest. If we
#      selected variables once on all training data and then CV'd that
#      fixed formula, the CV would be optimistically biased because
#      the selection has already peeked at every fold's data.
#   4. The kNN min-max scaler is FIT on the TRAINING set only and the
#      same training-derived min/max are APPLIED to the test set, so
#      no test information leaks into feature scaling either.
#   5. The held-out 20% test set is touched EXACTLY ONCE, at the end,
#      to compute the final reported metrics.
#
# Class imbalance is ~85/15 (Pass/Fail), so we emphasize AUC,
# sensitivity, specificity, and balanced accuracy alongside raw
# accuracy, and we compare every model to a naive baseline of
# "always predict pass".
#
# REQUIRED PACKAGES: dplyr, ggplot2, class, caret, pROC
# To install missing packages, run:
#   install.packages(c("dplyr","ggplot2","class","caret","pROC"))
# ===================================================================

# --- 0. Setup ---------------------------------------------------
library(dplyr)
library(ggplot2)
library(class)
library(caret)
library(pROC)

set.seed(123)  # reproducibility

# IMPORTANT: Set this to your folder containing student-por.csv
# In RStudio: Session > Set Working Directory > To Source File Location
# setwd("YOUR/PATH/HERE")

# --- 1. Load and Inspect Data -----------------------------------
data_por <- read.table("student-por.csv", sep = ";", header = TRUE,
                       stringsAsFactors = TRUE)

cat("====================================================\n")
cat("SECTION 2 - DATASET SUMMARY\n")
cat("====================================================\n")
cat("Dataset: Student Performance (Portuguese)\n")
cat("Source : UCI ML Repository - Cortez & Silva (2008)\n")
cat("Rows   :", nrow(data_por), "\n")
cat("Cols   :", ncol(data_por), "\n")
cat("Missing:", sum(is.na(data_por)), "\n\n")

# --- 2. Target Variable & Drop G1, G2 ---------------------------
# Binary target: pass = 1 if G3 >= 10, else 0
# Drop G1 and G2 to predict from social/demographic features ONLY.
df <- data_por %>%
  mutate(target_pass = ifelse(G3 >= 10, 1, 0)) %>%
  select(-G1, -G2, -G3)

cat("Target distribution (Pass = G3 >= 10):\n")
print(table(df$target_pass, dnn = "target_pass"))
pass_rate <- mean(df$target_pass)
cat(sprintf("Pass rate     : %.2f%%\n", pass_rate * 100))
cat(sprintf("Fail rate     : %.2f%%\n", (1 - pass_rate) * 100))
cat(sprintf("BASELINE: 'always predict pass' accuracy = %.2f%%\n\n",
            pass_rate * 100))

# ===================================================================
# SECTION 3 - DESCRIPTIVE STATISTICS
# ===================================================================
cat("====================================================\n")
cat("SECTION 3 - DESCRIPTIVE STATISTICS\n")
cat("====================================================\n")

# Separate numerical and categorical columns (before dummy encoding)
num_vars <- names(df)[sapply(df, is.numeric)]
num_vars <- setdiff(num_vars, "target_pass")
cat_vars <- names(df)[sapply(df, is.factor)]

# --- 3a. Summary statistics for numerical variables -------------
desc_stats <- data.frame(
  Variable = num_vars,
  Mean   = sapply(df[num_vars], function(x) round(mean(x), 2)),
  Median = sapply(df[num_vars], function(x) round(median(x), 2)),
  SD     = sapply(df[num_vars], function(x) round(sd(x), 2)),
  Min    = sapply(df[num_vars], min),
  Max    = sapply(df[num_vars], max)
)
cat("\n--- Numerical variable summary ---\n")
print(desc_stats, row.names = FALSE)
write.csv(desc_stats, "table_descriptive_stats.csv", row.names = FALSE)

# --- 3b. Frequency tables (categorical) -------------------------
cat("\n--- Frequency tables (categorical variables) ---\n")
sink("tables_frequency.txt")
for (v in cat_vars) {
  cat("\n--", v, "--\n")
  print(table(df[[v]]))
}
sink()
cat("Frequency tables written to tables_frequency.txt\n")

# --- 3c. Contingency tables (key cross-tabs) --------------------
cat("\n--- Contingency: Mother's Education x Pass/Fail ---\n")
ct_medu <- table(Mother_Education = df$Medu,
                 Status = ifelse(df$target_pass == 1, "Pass", "Fail"))
print(ct_medu)
cat("\nRow proportions (pass rate within each Medu level):\n")
print(round(prop.table(ct_medu, margin = 1), 3))

cat("\n--- Contingency: School x Pass/Fail ---\n")
ct_school <- table(School = df$school,
                   Status = ifelse(df$target_pass == 1, "Pass", "Fail"))
print(ct_school)

cat("\n--- Contingency: Higher Ed Goal x Pass/Fail ---\n")
ct_higher <- table(WantsHigherEd = df$higher,
                   Status = ifelse(df$target_pass == 1, "Pass", "Fail"))
print(ct_higher)

# --- 3d. Correlation matrix (numeric predictors + target) -------
cor_mat <- round(cor(df[, c(num_vars, "target_pass")]), 3)
cat("\n--- Correlation matrix (numeric variables vs target_pass) ---\n")
print(cor_mat)
write.csv(cor_mat, "table_correlation_matrix.csv")

# --- 3e. Visualizations: PDFs for the report --------------------
# Plot 1: Histogram of Absences
pdf("plot1_absences_histogram.pdf", width = 7, height = 5)
print(
  ggplot(df, aes(x = absences)) +
    geom_histogram(binwidth = 2, fill = "steelblue", color = "black") +
    labs(title = "Distribution of Student Absences",
         x = "Number of Absences", y = "Frequency") +
    theme_minimal()
)
dev.off()

# Plot 2: Boxplot - Weekend Alcohol (Walc) vs Pass/Fail
pdf("plot2_alcohol_boxplot.pdf", width = 8, height = 5)
print(
  ggplot(df, aes(x = factor(Walc), y = as.numeric(target_pass),
                 fill = factor(Walc))) +
    geom_boxplot(alpha = 0.7) +
    labs(title = "Pass/Fail vs Weekend Alcohol Consumption",
         x = "Weekend Alcohol (1=very low to 5=very high)",
         y = "Pass Rate (proportion)") +
    theme_minimal() + theme(legend.position = "none")
)
# A more interpretable plot: pass rate by Walc level
walc_summary <- df %>%
  group_by(Walc) %>%
  summarise(pass_rate = mean(target_pass), n = n())
print(
  ggplot(walc_summary, aes(x = factor(Walc), y = pass_rate)) +
    geom_col(fill = "steelblue") +
    geom_text(aes(label = paste0(round(pass_rate*100), "%\n(n=", n, ")")),
              vjust = -0.3) +
    ylim(0, 1) +
    labs(title = "Pass Rate by Weekend Alcohol Consumption (Walc)",
         x = "Weekend Alcohol Level", y = "Pass Rate") +
    theme_minimal()
)
dev.off()

# Plot 3: Boxplot - Past Failures vs Pass/Fail
pdf("plot3_failures_boxplot.pdf", width = 7, height = 5)
print(
  ggplot(df, aes(x = factor(target_pass, labels = c("Fail", "Pass")),
                 y = failures, fill = factor(target_pass))) +
    geom_boxplot(alpha = 0.8) +
    labs(title = "Past Class Failures vs Current Status",
         x = "Current Status (Final Grade G3)",
         y = "Number of Past Class Failures") +
    scale_fill_manual(values = c("tomato", "seagreen")) +
    theme_minimal() + theme(legend.position = "none")
)
dev.off()

# Plot 4: Stacked bar - Mother's Education vs Pass/Fail proportion
pdf("plot4_medu_stacked.pdf", width = 7, height = 5)
print(
  ggplot(df, aes(x = factor(Medu),
                 fill = factor(target_pass, labels = c("Fail", "Pass")))) +
    geom_bar(position = "fill", alpha = 0.85) +
    labs(title = "Pass/Fail Proportion by Mother's Education Level",
         x = "Mother's Education (0 = none, 4 = higher ed)",
         y = "Proportion of Students") +
    scale_fill_manual(values = c("tomato", "seagreen"), name = "Status") +
    theme_minimal()
)
dev.off()

# Plot 5: Scatter - studytime vs failures (jittered, sized by pass)
pdf("plot5_studytime_failures.pdf", width = 7, height = 5)
print(
  ggplot(df, aes(x = jitter(studytime, 0.5),
                 y = jitter(as.numeric(failures), 0.3),
                 color = factor(target_pass, labels = c("Fail", "Pass")))) +
    geom_point(alpha = 0.6) +
    labs(title = "Study Time vs Past Failures",
         x = "Weekly Study Time (1-4)", y = "Past Failures",
         color = "Status") +
    scale_color_manual(values = c("tomato", "seagreen")) +
    theme_minimal()
)
dev.off()

# Plot 6: Histogram of absences split by status
pdf("plot6_absences_by_status.pdf", width = 7, height = 5)
print(
  ggplot(df, aes(x = absences,
                 fill = factor(target_pass, labels = c("Fail", "Pass")))) +
    geom_histogram(binwidth = 2, position = "identity", alpha = 0.6) +
    labs(title = "Absences Distribution by Pass/Fail Status",
         x = "Number of Absences", y = "Frequency",
         fill = "Status") +
    scale_fill_manual(values = c("tomato", "seagreen")) +
    theme_minimal()
)
dev.off()

cat("\nPlots saved: plot1..plot6 PDFs in working directory.\n")

# ===================================================================
# SECTION 4 - MODELING
# ===================================================================
# Evaluation protocol (recap):
#   - Single stratified 80/20 train/test split (the split is the
#     single source of truth; both models use the same rows).
#   - All tuning (threshold, k) via 10-fold CV INSIDE the training set.
#   - Stepwise variable selection re-run INSIDE each CV fold to avoid
#     selection bias in the CV error estimate.
#   - kNN scaler fit on train only; same scaler applied to test.
#   - Test set is touched ONCE, at the end of this section.
# ===================================================================
cat("\n====================================================\n")
cat("SECTION 4 - MODELING\n")
cat("====================================================\n")

# --- 4a. Prepare data for both models ---------------------------
# Logistic: glm() handles factors natively, no dummy encoding needed.
df_glm <- df
df_glm$target_pass <- as.integer(df_glm$target_pass)

# kNN: distance metrics require numeric input, so we dummy-encode factors.
factor_cols <- sapply(df, is.factor)
formula_dummies <- as.formula(paste("~", paste(names(df)[factor_cols],
                                                collapse = "+")))
dummies <- model.matrix(formula_dummies, data = df)[, -1]
df_knn  <- cbind(df[, !factor_cols], as.data.frame(dummies))
df_knn$target_pass <- as.integer(df_knn$target_pass)

cat("Logistic-ready data dims :", dim(df_glm), "\n")
cat("kNN-ready data dims      :", dim(df_knn), "\n\n")

# --- 4b. ONE stratified 80/20 train/test split ------------------
# Both models use the SAME train_index so their evaluations are
# directly comparable. The test partition is FROZEN until Section 4f.
set.seed(123)
train_index <- createDataPartition(df$target_pass, p = 0.8, list = FALSE)

train_glm <- df_glm[ train_index, ]
test_glm  <- df_glm[-train_index, ]
train_knn <- df_knn[ train_index, ]
test_knn  <- df_knn[-train_index, ]

baseline_test_acc <- mean(test_glm$target_pass)
cat(sprintf("Train rows: %d   Test rows: %d\n",
            nrow(train_glm), nrow(test_glm)))
cat(sprintf("Test-set baseline (always-pass) accuracy: %.2f%%\n\n",
            baseline_test_acc * 100))

# ============================================================
# 4c. Method 1: Logistic Regression with Stepwise Selection
# ============================================================
cat("---- Method 1: Logistic Regression + Stepwise ----\n")

# --- 4c.i. 10-fold CV with stepwise selection INSIDE each fold ---
# WHY inside-fold: stepwise variable selection IS part of the model-
# building procedure. Selecting variables once on all training data
# and then CV'ing that fixed formula gives an optimistically biased
# error estimate, because the selection has already seen every fold's
# data. Re-running step() per fold is the honest way to estimate how
# the WHOLE procedure (selection + fit) generalizes.
set.seed(123)
folds <- createFolds(train_glm$target_pass, k = 10,
                     list = TRUE, returnTrain = FALSE)
oof_prob   <- numeric(nrow(train_glm))   # out-of-fold predicted probability
oof_actual <- integer(nrow(train_glm))   # corresponding true labels

for (i in seq_along(folds)) {
  val_idx   <- folds[[i]]
  fold_tr   <- train_glm[-val_idx, ]
  fold_va   <- train_glm[ val_idx, ]
  fold_full <- glm(target_pass ~ ., data = fold_tr, family = "binomial")
  fold_step <- step(fold_full, direction = "both", trace = 0)
  oof_prob[val_idx]   <- predict(fold_step, newdata = fold_va,
                                 type = "response")
  oof_actual[val_idx] <- fold_va$target_pass
}
cv_auc_log <- as.numeric(auc(roc(oof_actual, oof_prob,
                                  quiet = TRUE, direction = "<")))
cat(sprintf("Logistic 10-fold CV AUC (stepwise inside each fold): %.4f\n",
            cv_auc_log))

# --- 4c.ii. Threshold tuning on the SAME out-of-fold predictions ---
# Optimize balanced accuracy = (sensitivity + specificity) / 2.
# Balanced accuracy is appropriate for the ~85/15 class imbalance:
# raw accuracy is dominated by the Pass class, but we also care about
# correctly identifying failing students (the operationally useful case).
thresholds <- seq(0.10, 0.95, by = 0.05)
bal_acc <- sapply(thresholds, function(t) {
  p <- as.integer(oof_prob > t)
  TP <- sum(p == 1 & oof_actual == 1); FN <- sum(p == 0 & oof_actual == 1)
  TN <- sum(p == 0 & oof_actual == 0); FP <- sum(p == 1 & oof_actual == 0)
  sens <- TP / (TP + FN); spec <- TN / (TN + FP)
  (sens + spec) / 2
})
best_t <- thresholds[which.max(bal_acc)]
cat(sprintf("CV-selected threshold (max balanced accuracy): %.2f\n", best_t))
cat("\nBalanced-accuracy curve across thresholds (CV-derived):\n")
print(data.frame(threshold = thresholds,
                 balanced_accuracy = round(bal_acc, 4)),
      row.names = FALSE)

# --- 4c.iii. Final stepwise model fit on the FULL training set ----
# Standard practice: CV estimates the procedure's expected error; the
# model we ultimately deploy / report coefficients for is fit on all
# available training data, with no further peeking at the test set.
full_log <- glm(target_pass ~ ., data = train_glm, family = "binomial")
step_log <- step(full_log, direction = "both", trace = 0)

cat("\nFinal stepwise-selected formula (fit on full training set):\n")
print(formula(step_log))
cat("\nFinal model coefficients:\n")
print(round(summary(step_log)$coefficients, 4))

# --- Performance metrics function (used for logistic, kNN, baseline) -
get_metrics <- function(actual, predicted, prob = NULL, label = "") {
  cm <- table(Predicted = factor(predicted, levels = c(0, 1)),
              Actual    = factor(actual,    levels = c(0, 1)))
  TP <- cm["1","1"]; TN <- cm["0","0"]
  FP <- cm["1","0"]; FN <- cm["0","1"]
  accuracy     <- (TP + TN) / sum(cm)
  sensitivity  <- TP / (TP + FN)        # recall for Pass
  specificity  <- TN / (TN + FP)        # recall for Fail
  bal_accuracy <- (sensitivity + specificity) / 2
  precision    <- if ((TP + FP) > 0) TP / (TP + FP) else NA
  f1           <- if (!is.na(precision) &&
                      (precision + sensitivity) > 0)
                    2 * precision * sensitivity /
                    (precision + sensitivity) else NA
  auc_val      <- if (!is.null(prob))
                    as.numeric(auc(roc(actual, prob, quiet = TRUE,
                                       direction = "<"))) else NA
  cat("\n--", label, "--\n")
  cat("Confusion matrix (rows = predicted, cols = actual):\n")
  print(cm)
  cat(sprintf("Accuracy          : %.4f\n", accuracy))
  cat(sprintf("Sensitivity       : %.4f  (recall for Pass)\n", sensitivity))
  cat(sprintf("Specificity       : %.4f  (recall for Fail)\n", specificity))
  cat(sprintf("Balanced accuracy : %.4f\n", bal_accuracy))
  cat(sprintf("Precision         : %.4f\n", precision))
  cat(sprintf("F1 score          : %.4f\n", f1))
  if (!is.na(auc_val)) cat(sprintf("AUC               : %.4f\n", auc_val))
  data.frame(model = label, accuracy = accuracy,
             sensitivity = sensitivity, specificity = specificity,
             balanced_accuracy = bal_accuracy,
             precision = precision, f1 = f1, auc = auc_val)
}

# --- 4c.iv. Predict on test set ONCE using the CV-tuned threshold --
log_prob_test <- predict(step_log, newdata = test_glm, type = "response")
log_pred_test <- as.integer(log_prob_test > best_t)
log_metrics <- get_metrics(
  test_glm$target_pass, log_pred_test, log_prob_test,
  sprintf("Logistic Regression (CV-tuned threshold = %.2f)", best_t))

# ============================================================
# 4d. Method 2: k-Nearest Neighbors (kNN)
# ============================================================
cat("\n---- Method 2: k-Nearest Neighbors ----\n")

# --- 4d.i. Continuous/ordinal vs binary dummy columns -----------
# We normalize the 13 continuous + ordinal-Likert features (their
# raw scales differ wildly, e.g. absences 0-32 vs studytime 1-4,
# which would otherwise dominate Euclidean distance).
# We leave binary 0/1 dummies untouched. Note: min-max of a 0/1
# variable is a mathematical no-op (it stays 0/1), so this exclusion
# is for clarity, not numerical change.
cont_ord_vars <- c("age", "absences",
                   "Medu", "Fedu",
                   "traveltime", "studytime", "failures",
                   "famrel", "freetime", "goout",
                   "Dalc", "Walc", "health")
feature_cols  <- setdiff(names(df_knn), "target_pass")
binary_vars   <- setdiff(feature_cols, cont_ord_vars)
cat(sprintf("kNN normalization: %d continuous/ordinal vars scaled, %d binary dummies left as 0/1.\n",
            length(cont_ord_vars), length(binary_vars)))

# --- 4d.ii. Min-max scaler: FIT ON TRAINING ONLY, APPLY TO TEST -
# NO LEAKAGE: we compute (min, max) from train_knn only, then
# transform the test set with those EXACT training-derived values.
# We never recompute min/max on the test set, matching the rigor
# of the CV evaluation.
train_min <- sapply(train_knn[, cont_ord_vars], min)
train_max <- sapply(train_knn[, cont_ord_vars], max)

scale_with_train_range <- function(x, vars, mn, mx) {
  out <- x
  for (v in vars) {
    if (mx[v] != mn[v]) {
      out[[v]] <- (x[[v]] - mn[v]) / (mx[v] - mn[v])
    } else {
      out[[v]] <- 0
    }
  }
  out
}

train_knn_scaled <- scale_with_train_range(train_knn, cont_ord_vars,
                                           train_min, train_max)
test_knn_scaled  <- scale_with_train_range(test_knn,  cont_ord_vars,
                                           train_min, train_max)

train_x <- train_knn_scaled[, feature_cols]
test_x  <- test_knn_scaled[ , feature_cols]
train_y <- train_knn_scaled$target_pass
test_y  <- test_knn_scaled$target_pass

# --- 4d.iii. Tune k via 10-fold CV on the TRAINING set ----------
# Use caret::train, which refits on each fold's training portion and
# evaluates on its held-out portion. preProcess = "range" applies
# min-max per fold. Since min-max on the binary dummies is a no-op,
# this is mathematically equivalent to scaling only the 13 numeric
# columns identified above.
k_vals <- c(3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 25)
train_knn_caret <- train_knn
train_knn_caret$target_pass <- factor(
  ifelse(train_knn$target_pass == 1, "Pass", "Fail"),
  levels = c("Fail", "Pass"))

set.seed(123)
ctrl_knn <- trainControl(method = "cv", number = 10,
                         classProbs = TRUE,
                         summaryFunction = twoClassSummary,
                         savePredictions = "final")
cv_knn <- train(target_pass ~ ., data = train_knn_caret,
                method = "knn",
                trControl = ctrl_knn, metric = "ROC",
                preProcess = "range",
                tuneGrid = expand.grid(k = k_vals))

cat("\nkNN 10-fold CV results (training set only):\n")
print(cv_knn$results[, c("k", "ROC", "Sens", "Spec")])
best_k     <- cv_knn$bestTune$k
cv_auc_knn <- cv_knn$results$ROC[cv_knn$results$k == best_k]
cat(sprintf("CV-selected k = %d   (CV AUC = %.4f)\n", best_k, cv_auc_knn))

# --- 4d.iv. Fit final kNN on full training set, predict on test ONCE -
set.seed(123)
knn_pred_raw <- knn(train = train_x, test = test_x,
                    cl = train_y, k = best_k, prob = TRUE)
# class::knn's "prob" attribute is the proportion of votes for the
# winning class. Convert to probability of class 1 (Pass).
knn_prob_win <- attr(knn_pred_raw, "prob")
knn_prob_1   <- ifelse(knn_pred_raw == 1, knn_prob_win, 1 - knn_prob_win)
knn_pred     <- as.integer(as.character(knn_pred_raw))

knn_metrics  <- get_metrics(test_y, knn_pred, knn_prob_1,
                            sprintf("kNN (CV-tuned k = %d)", best_k))

# ============================================================
# 4e. Naive baseline on the SAME test set
# ============================================================
baseline_metrics <- data.frame(
  model = "BASELINE (always predict Pass)",
  accuracy = baseline_test_acc,
  sensitivity = 1, specificity = 0,
  balanced_accuracy = 0.5,
  precision = baseline_test_acc, f1 = NA, auc = 0.5
)

# ===================================================================
# SECTION 5 - RESULTS & COMPARISON  (test-set numbers ONLY)
# ===================================================================
cat("\n====================================================\n")
cat("SECTION 5 - SUMMARY OF RESULTS\n")
cat("====================================================\n")

results_summary <- rbind(log_metrics, knn_metrics, baseline_metrics)
cat("\n--- FINAL RESULTS COMPARISON (test set, n =",
    nrow(test_glm), ") ---\n")
print(round(results_summary[, -1], 4), row.names = FALSE)
write.csv(results_summary, "table_final_results.csv", row.names = FALSE)

# --- ROC curves -------------------------------------------------
pdf("plot7_roc_curves.pdf", width = 7, height = 6)
roc_log_curve <- roc(test_glm$target_pass, log_prob_test,
                     quiet = TRUE, direction = "<")
roc_knn_curve <- roc(test_y, knn_prob_1,
                     quiet = TRUE, direction = "<")
plot(roc_log_curve, col = "blue", lwd = 2,
     main = "ROC Curves: Logistic Regression vs kNN")
lines(roc_knn_curve, col = "red", lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "gray")
legend("bottomright",
       legend = c(sprintf("Logistic Reg (AUC = %.3f)", auc(roc_log_curve)),
                  sprintf("kNN k=%d (AUC = %.3f)",
                          best_k, auc(roc_knn_curve)),
                  "Random (AUC = 0.500)"),
       col = c("blue", "red", "gray"),
       lwd = c(2, 2, 1), lty = c(1, 1, 2))
dev.off()

# --- Coefficients table for the final stepwise model ------------
coef_table <- summary(step_log)$coefficients
coef_table <- coef_table[order(coef_table[, "Pr(>|z|)"]), ]
cat("\n--- Most significant predictors (final stepwise logistic) ---\n")
print(round(coef_table, 4))
write.csv(coef_table, "table_logistic_coefficients.csv")

# --- Side-by-side honest-gap summary (CV vs test vs baseline) ----
# THIS is the table to use in the report's conclusion: it makes the
# CV-vs-test gap explicit and compares both models to the baseline.
honest_gap <- data.frame(
  Model         = c("Logistic Regression (stepwise)",
                    sprintf("kNN (k = %d)", best_k),
                    "Baseline (always Pass)"),
  CV_AUC        = c(round(cv_auc_log, 4),
                    round(cv_auc_knn, 4),
                    0.5),
  Test_AUC      = c(round(log_metrics$auc, 4),
                    round(knn_metrics$auc, 4),
                    0.5),
  Test_Accuracy = c(round(log_metrics$accuracy, 4),
                    round(knn_metrics$accuracy, 4),
                    round(baseline_test_acc, 4)),
  Baseline_Acc  = round(baseline_test_acc, 4),
  stringsAsFactors = FALSE
)
cat("\n--- HONEST GAP SUMMARY (CV vs test vs baseline) ---\n")
print(honest_gap, row.names = FALSE)
write.csv(honest_gap, "table_honest_gap.csv", row.names = FALSE)

# --- Concluding numbers for the report --------------------------
cat("\n====================================================\n")
cat("CONCLUSION POINTS FOR REPORT\n")
cat("====================================================\n")
cat(sprintf("- Pass-rate baseline       : %.2f%%\n",
            baseline_test_acc * 100))
cat(sprintf("- Logistic test accuracy   : %.2f%%   (CV AUC %.3f, test AUC %.3f)\n",
            log_metrics$accuracy * 100, cv_auc_log, log_metrics$auc))
cat(sprintf("- kNN test accuracy        : %.2f%%   (CV AUC %.3f, test AUC %.3f)\n",
            knn_metrics$accuracy * 100, cv_auc_knn, knn_metrics$auc))
cat("\nKEY MESSAGE: Without G1/G2, social/demographic variables\n")
cat("provide LIMITED predictive power above the naive baseline.\n")
cat("This is CONSISTENT with Cortez & Silva (2008) Setup C results.\n")
cat("Most informative predictors come from the stepwise model above\n")
cat("(typically: failures, absences, school, higher, and study time).\n")

cat("\n=== SCRIPT COMPLETE ===\n")
cat("Output files in working directory:\n")
cat("  - table_descriptive_stats.csv\n")
cat("  - table_correlation_matrix.csv\n")
cat("  - tables_frequency.txt\n")
cat("  - table_final_results.csv\n")
cat("  - table_honest_gap.csv\n")
cat("  - table_logistic_coefficients.csv\n")
cat("  - plot1_absences_histogram.pdf\n")
cat("  - plot2_alcohol_boxplot.pdf\n")
cat("  - plot3_failures_boxplot.pdf\n")
cat("  - plot4_medu_stacked.pdf\n")
cat("  - plot5_studytime_failures.pdf\n")
cat("  - plot6_absences_by_status.pdf\n")
cat("  - plot7_roc_curves.pdf\n")
