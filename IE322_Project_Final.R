# ===================================================================
# IE 322 - Exploratory Data Analytics - Final Project
# Group Members: [Group Members - Fill in]
# Dataset: Student Performance Data Set (Cortez & Silva, 2008)
# Course: Portuguese Language (n = 649)
# Date: May 2026
# ===================================================================
#
# Project aim:
# Predict whether a student passes the Portuguese language course.
# Pass is defined as G3 >= 10; otherwise the student is labeled Fail.
#
# G1 and G2 are removed from the predictors because they are previous
# grades. The project focuses on demographic, social, and family-related
# variables instead of using previous exam grades.
#
# Methods used in this project:
#   1. Logistic Regression
#   2. k-Nearest Neighbors (kNN)
#
# Evaluation plan:
#   1. Split the data into 80% training and 20% testing.
#   2. Use 10-fold cross-validation on the training data.
#   3. Tune the logistic threshold and kNN k value using training data.
#   4. Evaluate final models on the test data.
#   5. Compare models with Accuracy, Sensitivity, Specificity,
#      Balanced Accuracy, AUC, and Classification MSE.
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

# --- 0b. Remove old output files ---------------------------------
# This prevents old tables/plots from being mixed with new results.
output_files <- c(
  "table_descriptive_stats.csv",
  "table_correlation_matrix.csv",
  "tables_frequency.txt",
  "table_final_results.csv",
  "table_cv_summary.csv",
  "table_honest_gap.csv",
  "table_repeated_cv_results.csv",
  "table_logistic_coefficients.csv",
  "plot1_absences_histogram.pdf",
  "plot2_alcohol_boxplot.pdf",
  "plot2_alcohol_pass_rate.pdf",
  "plot3_failures_boxplot.pdf",
  "plot4_medu_stacked.pdf",
  "plot5_studytime_failures.pdf",
  "plot6_absences_by_status.pdf",
  "plot7_roc_curves.pdf"
)

old_files <- output_files[file.exists(output_files)]
if (length(old_files) > 0) {
  file.remove(old_files)
  cat("Old output files removed:\n")
  print(old_files)
} else {
  cat("No old output files found. Starting fresh.\n")
}

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

# Plot 2: Pass rate by weekend alcohol consumption
walc_summary <- df %>%
  group_by(Walc) %>%
  summarise(pass_rate = mean(target_pass), n = n())

pdf("plot2_alcohol_pass_rate.pdf", width = 7, height = 5)
print(
  ggplot(walc_summary, aes(x = factor(Walc), y = pass_rate)) +
    geom_col(fill = "steelblue") +
    geom_text(aes(label = paste0(round(pass_rate*100), "%\n(n=", n, ")")),
              vjust = -0.3) +
    ylim(0, 1) +
    labs(title = "Pass Rate by Weekend Alcohol Consumption",
         x = "Weekend Alcohol Level (1 = very low, 5 = very high)",
         y = "Pass Rate") +
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
# Two predictive models are built in this section:
# Logistic Regression and k-Nearest Neighbors.
# Both models use the same train/test split for a fair comparison.
# Cross-validation is used only on the training data.
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

# --- 4b. Stratified 80/20 train-test split ----------------------
# Both models use the same training and testing rows.
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

# --- 4c.i. 10-fold CV for Logistic Regression -------------------
# The model is fitted on 9 folds and validated on the remaining fold.
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
cat(sprintf("Logistic 10-fold CV AUC: %.4f\n", cv_auc_log))

# --- 4c.ii. Threshold tuning for Logistic Regression ------------
# Because the data is imbalanced, threshold is selected by balanced accuracy.
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

# --- 4c.iii. Final Logistic Regression model --------------------
# The final model is fitted on the full training set.
full_log <- glm(target_pass ~ ., data = train_glm, family = "binomial")
step_log <- step(full_log, direction = "both", trace = 0)

cat("\nFinal stepwise-selected formula (fit on full training set):\n")
print(formula(step_log))
cat("\nFinal model coefficients:\n")
print(round(summary(step_log)$coefficients, 4))

# --- Performance metrics function (used for logistic, kNN, baseline) -
get_metrics <- function(actual, predicted, prob = NULL, label = "") {
  actual_num <- as.integer(as.character(actual))
  predicted_num <- as.integer(as.character(predicted))
  cm <- table(Predicted = factor(predicted_num, levels = c(0, 1)),
              Actual    = factor(actual_num,    levels = c(0, 1)))
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
  
  # Classification MSE: for 0/1 predictions, this equals misclassification error.
  mse_class <- mean((actual_num - predicted_num)^2)
  
  
  auc_val      <- if (!is.null(prob))
    as.numeric(auc(roc(actual_num, prob, quiet = TRUE,
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
  cat(sprintf("Classification MSE: %.4f\n", mse_class))
  if (!is.na(auc_val)) cat(sprintf("AUC               : %.4f\n", auc_val))
  data.frame(model = label, accuracy = accuracy,
             sensitivity = sensitivity, specificity = specificity,
             balanced_accuracy = bal_accuracy,
             precision = precision, f1 = f1,
             mse_class = mse_class,
             auc = auc_val)
}

# --- 4c.iv. Logistic Regression prediction on the test set -------
log_prob_test <- predict(step_log, newdata = test_glm, type = "response")
log_pred_test <- as.integer(log_prob_test > best_t)
log_metrics <- get_metrics(
  test_glm$target_pass, log_pred_test, log_prob_test,
  sprintf("Logistic Regression (CV-tuned threshold = %.2f)", best_t))

# ============================================================
# 4d. Method 2: k-Nearest Neighbors (kNN)
# ============================================================
cat("\n---- Method 2: k-Nearest Neighbors ----\n")

# --- 4d.i. Select variables to scale for kNN --------------------
# kNN uses distances, so numerical/ordinal variables are scaled.
cont_ord_vars <- c("age", "absences",
                   "Medu", "Fedu",
                   "traveltime", "studytime", "failures",
                   "famrel", "freetime", "goout",
                   "Dalc", "Walc", "health")
feature_cols  <- setdiff(names(df_knn), "target_pass")
binary_vars   <- setdiff(feature_cols, cont_ord_vars)
cat(sprintf("kNN normalization: %d continuous/ordinal vars scaled, %d binary dummies left as 0/1.\n",
            length(cont_ord_vars), length(binary_vars)))

# --- 4d.ii. Min-max scaling for kNN -----------------------------
# Min and max values are calculated from the training set.
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

# --- 4d.iii. Tune k using 10-fold CV ----------------------------
# Different k values are compared using CV AUC.
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

# --- 4d.iv. Final kNN prediction on the test set ----------------
set.seed(123)
knn_pred_raw <- knn(train = train_x, test = test_x,
                    cl = train_y, k = best_k, prob = TRUE)
# Convert kNN vote proportion into probability of Pass.
knn_prob_win <- attr(knn_pred_raw, "prob")
knn_class    <- as.character(knn_pred_raw)
knn_prob_1   <- ifelse(knn_class == "1", knn_prob_win, 1 - knn_prob_win)

# --- 4d.v. Tune kNN vote threshold ------------------------------
# The threshold is selected by balanced accuracy because the classes are imbalanced.
set.seed(123)
knn_train_raw    <- knn(train = train_x, test = train_x,
                        cl = train_y, k = best_k, prob = TRUE)
knn_train_win    <- attr(knn_train_raw, "prob")
knn_train_class  <- as.character(knn_train_raw)
knn_prob_1_train <- ifelse(knn_train_class == "1", knn_train_win, 1 - knn_train_win)
knn_ba_train <- sapply(thresholds, function(t) {
  pr <- as.integer(knn_prob_1_train > t)
  TP <- sum(pr == 1 & train_y == 1); FN <- sum(pr == 0 & train_y == 1)
  TN <- sum(pr == 0 & train_y == 0); FP <- sum(pr == 1 & train_y == 0)
  ((TP / (TP + FN)) + (TN / (TN + FP))) / 2
})
knn_best_t <- thresholds[which.max(knn_ba_train)]
cat(sprintf("\nkNN train-selected vote-threshold (max balanced accuracy): %.2f\n",
            knn_best_t))

knn_pred     <- as.integer(knn_prob_1 > knn_best_t)
knn_metrics  <- get_metrics(test_y, knn_pred, knn_prob_1,
                            sprintf("kNN (k = %d, tuned vote-thr = %.2f)",
                                    best_k, knn_best_t))

# ============================================================
# 4e. Naive baseline on the SAME test set
# ============================================================
baseline_metrics <- data.frame(
  model = "BASELINE (always predict Pass)",
  accuracy = baseline_test_acc,
  sensitivity = 1,
  specificity = 0,
  balanced_accuracy = 0.5,
  precision = baseline_test_acc,
  f1 = NA,
  mse_class = 1 - baseline_test_acc,
  auc = 0.5
)

# ===================================================================
# SECTION 5 - RESULTS & COMPARISON
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

# --- Simple cross-validation summary ------------------------------
# This table summarizes the 10-fold CV AUC and the final test results.
cv_summary <- data.frame(
  Model = c("Logistic Regression (stepwise)",
            sprintf("kNN (k = %d)", best_k),
            "Baseline (always Pass)"),
  CV_AUC = c(round(cv_auc_log, 4),
             round(cv_auc_knn, 4),
             0.5),
  Test_AUC = c(round(log_metrics$auc, 4),
               round(knn_metrics$auc, 4),
               0.5),
  Test_Accuracy = c(round(log_metrics$accuracy, 4),
                    round(knn_metrics$accuracy, 4),
                    round(baseline_test_acc, 4)),
  Test_Balanced_Accuracy = c(round(log_metrics$balanced_accuracy, 4),
                             round(knn_metrics$balanced_accuracy, 4),
                             0.5),
  Test_MSE = c(round(log_metrics$mse_class, 4),
               round(knn_metrics$mse_class, 4),
               round(1 - baseline_test_acc, 4)),
  stringsAsFactors = FALSE
)
cat("\n--- CROSS-VALIDATION AND TEST SUMMARY ---\n")
print(cv_summary, row.names = FALSE)
write.csv(cv_summary, "table_cv_summary.csv", row.names = FALSE)

# --- Concluding numbers for the report --------------------------
cat("\n====================================================\n")
cat("CONCLUSION POINTS FOR REPORT\n")
cat("====================================================\n")
cat(sprintf("- Pass-rate baseline       : %.2f%%\n",
            baseline_test_acc * 100))
cat(sprintf("- Logistic test accuracy   : %.2f%%   (CV AUC %.3f, test AUC %.3f, MSE %.3f)\n",
            log_metrics$accuracy * 100, cv_auc_log, log_metrics$auc, log_metrics$mse_class))
cat(sprintf("- kNN test accuracy        : %.2f%%   (CV AUC %.3f, test AUC %.3f, MSE %.3f)\n",
            knn_metrics$accuracy * 100, cv_auc_knn, knn_metrics$auc, knn_metrics$mse_class))
cat("\nKEY MESSAGE: The baseline accuracy is high because most students pass.\n")
cat("However, the baseline has zero specificity, so it cannot identify failing students.\n")
cat("Therefore, AUC, specificity and balanced accuracy are more informative than raw accuracy.\n")
cat("Among the two models, kNN gives better AUC than Logistic Regression.\n")
cat("Without G1/G2, social/demographic variables provide limited but useful predictive information.\n")

cat("\n=== SCRIPT COMPLETE ===\n")
cat("Output files in working directory:\n")
cat("  - table_descriptive_stats.csv\n")
cat("  - table_correlation_matrix.csv\n")
cat("  - tables_frequency.txt\n")
cat("  - table_final_results.csv\n")
cat("  - table_cv_summary.csv\n")
cat("  - table_logistic_coefficients.csv\n")
cat("  - plot1_absences_histogram.pdf\n")
cat("  - plot2_alcohol_pass_rate.pdf\n")
cat("  - plot3_failures_boxplot.pdf\n")
cat("  - plot4_medu_stacked.pdf\n")
cat("  - plot5_studytime_failures.pdf\n")
cat("  - plot6_absences_by_status.pdf\n")
cat("  - plot7_roc_curves.pdf\n")
