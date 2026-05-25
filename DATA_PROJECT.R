# Load required libraries
library(dplyr)
library(fastDummies)

# Windows i??in ??rnek dosya yollar?? (E??ik ??izgilere / dikkat et!)
path_mat <- "C:/Users/Casper/Desktop/student-mat.csv"
path_por <- "C:/Users/Casper/Desktop/student-por.csv"

# E??er Mac kullan??yorsan yol ??una benzer: "/Users/KullaniciAdin/Desktop/DataMiningProject/student-mat.csv"

# 1. Veri Setlerini Direkt Yoldan (Path) Y??kleme
data_mat <- read.table(path_mat, sep=";", header=TRUE)
data_por <- read.table(path_por, sep=";", header=TRUE)

# 2. Merge datasets to find common students (382 students)
# Using the 13 demographic variables mentioned in Cortez (2008)
common_cols <- c("school", "sex", "age", "address", "famsize", "Pstatus",
                 "Medu", "Fedu", "Mjob", "Fjob", "reason", "nursery", "internet")

df_merged <- merge(data_mat, data_por, by=common_cols)

# 3. Feature Engineering: Create binary target and drop mid-term grades
df_cleaned <- df_merged %>%
  mutate(
    # Calculate the average of Math and Portuguese final grades
    G3_avg = (G3.x + G3.y) / 2, 
    
    # Binary Classification Target: 1 (Pass) if average >= 10, else 0 (Fail)
    target_pass = ifelse(G3_avg >= 10, 1, 0) 
  ) %>%
  # Drop G1 and G2 to prevent data leakage, and drop old G3 columns
  select(-starts_with("G1"), -starts_with("G2"), -G3.x, -G3.y, -G3_avg)

# 4. Create Indicator (Dummy) Variables
# Required for kNN (distance metric) and Logistic Regression assumptions
df_final <- dummy_cols(df_cleaned, 
                       remove_first_dummy = TRUE, # Prevents multicollinearity
                       remove_selected_columns = TRUE) # Removes original categorical columns

# Check the dimensions and structure of the final dataset
dim(df_final)
str(df_final)
# Optional: Save the final processed dataset to your computer as a CSV file
##  write.csv(df_final, "final_student_data.csv", row.names = FALSE)

# Grafik k??t??phanesini aktif edelim
library(ggplot2)

# 1. Summary Statistics (??zet ??statistikler)
# Ya??, devams??zl??k ve ??al????ma s??resi gibi say??sal verilerin ??zetini al??yoruz.
# Matematik dosyas?? baz al??nd?????? i??in .x uzant??l?? s??tunlar?? se??iyoruz.
summary_stats <- summary(df_cleaned[, c("age", "absences.x", "studytime.x")])
print("--- SUMMARY STATISTICS ---")
print(summary_stats)

# 2. Contingency Table (??apraz Tablo)
# Anne e??itim seviyesi ile ba??ar?? durumu aras??ndaki ili??kiyi inceliyoruz.
table_medu <- table(Mother_Education = df_cleaned$Medu, Status = df_cleaned$target_pass)
print("--- CONTINGENCY TABLE ---")
print(table_medu)

# 1. H??STOGRAM: Devams??zl??k Da????l??m?? (Absences)
hist_plot <- ggplot(df_cleaned, aes(x = absences.x)) +
  geom_histogram(binwidth = 4, fill = "steelblue", color = "black", alpha = 0.8) +
  labs(title = "Distribution of Student Absences",
       x = "Number of Absences",
       y = "Frequency") +
  theme_minimal()
print(hist_plot)

# 2. BOX PLOT: Ge??mi?? Ba??ar??s??zl??klar ve G??ncel Ba??ar?? (Failures)
fail_plot <- ggplot(df_cleaned, aes(x = factor(target_pass, labels = c("Fail", "Pass")), 
                                    y = failures.x, 
                                    fill = factor(target_pass))) +
  geom_boxplot(alpha = 0.8) +
  labs(title = "Impact of Past Class Failures on Current Success",
       x = "Current Academic Status",
       y = "Number of Past Class Failures") +
  scale_fill_manual(values = c("tomato", "palegreen3")) +
  theme_minimal() +
  theme(legend.position = "none")
print(fail_plot)

# 3. STACKED BAR PLOT: Anne E??itimi ve Ba??ar?? Oran?? (Medu)
medu_plot <- ggplot(df_cleaned, aes(x = factor(Medu), fill = factor(target_pass, labels = c("Fail", "Pass")))) +
  geom_bar(position = "fill", alpha = 0.8) +
  labs(title = "Student Pass/Fail Ratio by Mother's Education Level",
       x = "Mother's Education Level (0: None to 4: Higher Ed)",
       y = "Proportion of Students") +
  scale_fill_manual(values = c("tomato", "palegreen3"), name = "Status") +
  theme_minimal()
print(medu_plot)

# Gerekli K??t??phaneler (kNN ve Confusion Matrix i??in)
library(class)
library(caret)

# ---------------------------------------------------------
# 1. TRAIN / TEST SPLIT (Veriyi %80 E??itim, %20 Test olarak b??lme)
# ---------------------------------------------------------
set.seed(123) # Sonu??lar??n her ??al????t??rd??????nda ayn?? ????kmas?? i??in
train_index <- createDataPartition(df_final$target_pass, p = 0.8, list = FALSE)

train_data <- df_final[train_index, ]
test_data  <- df_final[-train_index, ]

# ---------------------------------------------------------
# 2. STEPWISE LOGISTIC REGRESSION
# ---------------------------------------------------------
# ??nce t??m de??i??kenlerle ( ~ . ) bir lojistik regresyon kuruyoruz
full_log_model <- glm(target_pass ~ ., data = train_data, family = "binomial")

# Stepwise algoritmas?? ile gereksiz de??i??kenleri eliyoruz (AIC de??erine g??re)
# trace = 0 diyerek konsolu y??zlerce sat??r i??lemle bo??mas??n?? engelliyoruz
step_log_model <- step(full_log_model, direction = "both", trace = 0)
# Stepwise modelinin i??inde hangi de??i??kenlerin kald??????n?? ve p-de??erlerini g??rmek i??in:
summary(step_log_model)
# Modelin Test Verisi ??zerinde Tahmin Yapmas??
log_prob <- predict(step_log_model, newdata = test_data, type = "response")
log_pred <- ifelse(log_prob > 0.5, 1, 0) # %50'den b??y??kse Pass(1), k??????kse Fail(0)

# Sonu??lar?? G??relim
cat("\n--- LOGISTIC REGRESSION CONFUSION MATRIX ---\n")
log_cm <- table(Predicted = log_pred, Actual = test_data$target_pass)
print(log_cm)
log_accuracy <- sum(diag(log_cm)) / sum(log_cm)
cat("Logistic Regression Accuracy:", round(log_accuracy * 100, 2), "%\n")

# ---------------------------------------------------------
# 3. k-NEAREST NEIGHBORS (kNN) & NORMALIZATION
# ---------------------------------------------------------
# Min-Max Normalizasyon Form??l?? (Ders notlar??ndaki mesafe hesaplamas?? bozulmas??n diye)
normalize <- function(x) { return ((x - min(x)) / (max(x) - min(x))) }

# Hedef de??i??ken (target_pass) hari?? t??m veri setini normalize ediyoruz
df_features <- as.data.frame(lapply(df_final[, -which(names(df_final) == "target_pass")], normalize))
df_target <- df_final$target_pass

# Normalize edilmi?? veriyi de ayn?? index ile b??l??yoruz
train_knn_x <- df_features[train_index, ]
test_knn_x  <- df_features[-train_index, ]
train_knn_y <- df_target[train_index]
test_knn_y  <- df_target[-train_index]

# kNN modelini kurma (k=5 kom??u se??tik, genelde en iyi sonu??lardan birini verir)
knn_pred <- knn(train = train_knn_x, test = test_knn_x, cl = train_knn_y, k = 5)

# kNN Sonu??lar??n?? G??relim
cat("\n--- kNN CONFUSION MATRIX ---\n")
knn_cm <- table(Predicted = knn_pred, Actual = test_knn_y)
print(knn_cm)
knn_accuracy <- sum(diag(knn_cm)) / sum(knn_cm)
cat("kNN Accuracy:", round(knn_accuracy * 100, 2), "%\n")

# Gerekli K??t??phaneler (ROC e??risi i??in)
# E??er pROC y??kl?? de??ilse konsola install.packages("pROC") yaz??p kurabilirsin.
library(pROC)
library(caret)

# ---------------------------------------------------------
# 1. K-FOLD CROSS VALIDATION (K=10) - G??NCELLENM???? VERS??YON
# ---------------------------------------------------------
df_cv <- df_final
df_cv$target_pass <- as.factor(ifelse(df_cv$target_pass == 1, "Pass", "Fail"))

# 10 Katl?? ??apraz Do??rulama Ayar??
train_control <- trainControl(method = "cv", number = 10)

# D??KKAT: Stepwise modelimizin o 17 de??i??kenlik harika form??l??n?? otomatik olarak ??ekiyoruz
step_formula <- formula(step_log_model)

# 10 par??aya b??lerek o form??l?? (ayn?? modeli) test ediyoruz
cv_model <- train(step_formula, data = df_cv, method = "glm", family = "binomial", trControl = train_control)

cat("\n--- 10-FOLD CROSS VALIDATION RESULTS ---\n")
print(cv_model$results)

# ---------------------------------------------------------
# 2. COOK'S DISTANCE (Ayk??r?? De??er Tespiti)
# ---------------------------------------------------------
# Lojistik Regresyon form??l??n?? sapt??ran o istisnai ????rencileri buluyoruz
cooks_d <- cooks.distance(step_log_model)

# E??ik de??eri belirliyoruz (4 / n)
threshold <- 4 / nrow(train_data)

# Grafi??i ??izelim
plot(cooks_d, pch = 20, main = "Cook's Distance for Influential Students", 
     ylab = "Cook's Distance", xlab = "Student Index")
abline(h = threshold, col = "red", lwd = 2)

# ---------------------------------------------------------
# 3. ROC CURVE ve AUC HESAPLAMASI (Modelleri Kap????t??rma)
# ---------------------------------------------------------
# Lojistik Regresyon i??in ROC
roc_log <- roc(test_data$target_pass, log_prob)

# kNN modelinin olas??l??klar??n?? ROC i??in uygun formata ??eviriyoruz
knn_raw <- knn(train = train_knn_x, test = test_knn_x, cl = train_knn_y, k = 5, prob = TRUE)
knn_prob_win <- attr(knn_raw, "prob")
knn_prob_1 <- ifelse(knn_raw == 1, knn_prob_win, 1 - knn_prob_win)

# kNN i??in ROC
roc_knn <- roc(test_knn_y, knn_prob_1)

# ROC E??rilerini ??izelim
plot(roc_log, col = "blue", main = "ROC Curves: Logistic Regression vs kNN", lwd = 2)
lines(roc_knn, col = "red", lwd = 2)
legend("bottomright", legend = c(paste("Logistic Reg AUC:", round(auc(roc_log), 3)),
                                 paste("kNN AUC:", round(auc(roc_knn), 3))),
       col = c("blue", "red"), lwd = 2)