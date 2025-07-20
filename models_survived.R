library(dplyr)
library(caret)
library(glmnet)
library(randomForest)
library(rpart)
library(gbm)
library(xgboost)
library(e1071)
library(naivebayes)
library(class)
library(pROC)
library(knitr)


df <- read.csv("~/Downloads/rediscover_ecmo_vv_v3.csv")

df <- df[!is.na(df$bmi_avg), ]
df <- df[, !(names(df) %in% c("wbc_avg", "lactate_avg", 
                              "fibrin_avg", "sofa_neurological_score",
                              "patient_site", "site_description", "gender",
                              "race", "ethnicity", "visit_start_date",
                              "visit_ecmo_start_date", "visit_ecmo_end_date",
                              "death_date"))]


################################################
# random forest missing imputation
################################################
library(missForest)
set.seed(123)
df <- missForest(df)$ximp

################################################
# Data Preprocessing
################################################
set.seed(123)

# Assume df is already loaded and complete (imputed).

# Obesity indicator (BMI ≥ 30)
df$obese <- ifelse(df$bmi_avg >= 30, 1, 0)

# Convert numeric categorical variables to factors
df <- df %>% mutate(across(c(visit_occurrence_id, 
                             gender_concept_id, race_concept_id, 
                             ethnicity_concept_id), as.factor))

# Remove redundant variables if necessary (e.g., IDs)
df <- df %>% select(-visit_occurrence_id, -died_within_30_days_of_ecmo)

# Define predictor matrix and outcome
X <- model.matrix(died ~ . -1, data = df)
y <- df$died

################################################
# feature selection with LASSO
################################################
cv_lasso <- cv.glmnet(X, y, alpha=1, family="binomial")
lasso_model <- glmnet(X, y, alpha=1, family="binomial", lambda=cv_lasso$lambda.min)

# Selected features
coef(lasso_model)
selected_features <- rownames(coef(lasso_model))[which(coef(lasso_model)!=0)][-1] # excluding intercept
print(selected_features)

################################################
# Models
################################################
# Create 10 folds
folds <- cut(seq(1, nrow(df_model)), breaks = 10, labels = FALSE)

# Initialize AUC storage
auc_logit <- c()
auc_rf    <- c()
auc_gbm   <- c()
auc_xgb   <- c()

# Add before CV loop
roc_list_logit <- list()
roc_list_rf <- list()
roc_list_gbm <- list()
roc_list_xgb <- list()


for (i in 1:10) {
  
  cat("Fold", i, "\n")
  
  # Split data
  test_idx <- which(folds == i)
  train_data <- df_model[-test_idx, ]
  test_data  <- df_model[test_idx, ]
  
  ### Logistic Regression ###
  logit_model <- glm(died ~ ., data = train_data, family = binomial())
  logit_prob <- predict(logit_model, newdata = test_data, type = "response")
  logit_roc <- roc(test_data$died, logit_prob)
  auc_logit[i] <- auc(logit_roc)
  # Inside the loop (after computing `logit_roc`)
  roc_list_logit[[i]] <- logit_roc
  
  ### Random Forest ###
  rf_model <- randomForest(died ~ ., data = train_data, ntree = 500)
  rf_prob <- predict(rf_model, newdata = test_data, type = "prob")[, "Died"]
  rf_roc <- roc(test_data$died, rf_prob)
  auc_rf[i] <- auc(rf_roc)
  # Inside the loop (after computing `logit_roc`)
  roc_list_rf[[i]] <- rf_roc
  
  ### GBM ###
  train_gbm <- train_data %>% mutate(y = ifelse(died == "Died", 1, 0)) %>% select(-died)
  test_gbm  <- test_data %>% mutate(y = ifelse(died == "Died", 1, 0)) %>% select(-died)
  gbm_model <- gbm(y ~ ., data = train_gbm,
                   distribution = "bernoulli",
                   n.trees = 100, interaction.depth = 3,
                   shrinkage = 0.05, verbose = FALSE)
  gbm_prob <- predict(gbm_model, newdata = test_gbm, n.trees = 100, type = "response")
  gbm_roc <- roc(test_gbm$y, gbm_prob)
  auc_gbm[i] <- auc(gbm_roc)
  # Inside the loop (after computing `logit_roc`)
  roc_list_gbm[[i]] <- gbm_roc
  
  ### XGBoost ###
  X_train <- model.matrix(died ~ . -1, data = train_data)
  y_train <- ifelse(train_data$died == "Died", 1, 0)
  X_test  <- model.matrix(died ~ . -1, data = test_data)
  y_test  <- ifelse(test_data$died == "Died", 1, 0)
  
  xgb_model <- xgboost(data = X_train, label = y_train,
                       objective = "binary:logistic",
                       eval_metric = "auc",
                       nrounds = 100, verbose = 0)
  xgb_prob <- predict(xgb_model, newdata = X_test)
  xgb_roc <- roc(y_test, xgb_prob)
  auc_xgb[i] <- auc(xgb_roc)
  roc_list_xgb[[i]] <- xgb_roc
}


#####
# model comparison
#####

auc_table_cv <- data.frame(
  Model = c("Logistic Regression", "Random Forest", "GBM", "XGBoost"),
  Mean_AUC = round(c(mean(auc_logit), mean(auc_rf), mean(auc_gbm), mean(auc_xgb)), 3),
  Median_AUC = round(c(median(auc_logit), median(auc_rf), median(auc_gbm), median(auc_xgb)), 3),
  SD_AUC = round(c(sd(auc_logit), sd(auc_rf), sd(auc_gbm), sd(auc_xgb)), 3)
)

print(auc_table_cv)



#####
# visualization
#####
boxplot(auc_logit, auc_rf, auc_gbm, auc_xgb,
        names = c("Logistic", "RF", "GBM", "XGB"),
        ylab = "AUC", main = "10-Fold CV AUC Comparison")



# After the loop


# Optionally add smoothed average ROC (interpolate)
avg_roc_logit <- roc(response = unlist(lapply(roc_list_logit, `[[`, "response")),
               predictor = unlist(lapply(roc_list_logit, `[[`, "predictor")))
avg_roc_gbm <- roc(response = unlist(lapply(roc_list_gbm, `[[`, "response")),
                     predictor = unlist(lapply(roc_list_gbm, `[[`, "predictor")))
avg_roc_rf <- roc(response = unlist(lapply(roc_list_rf, `[[`, "response")),
                   predictor = unlist(lapply(roc_list_rf, `[[`, "predictor")))
avg_roc_xgb <- roc(response = unlist(lapply(roc_list_xgb, `[[`, "response")),
                   predictor = unlist(lapply(roc_list_xgb, `[[`, "predictor")))

plot(avg_roc_logit, col = "blue", main="Logistic ROC Curves Across Folds")
lines(avg_roc_gbm, col = "green", lwd=3)
lines(avg_roc_rf, col = "orange", lwd=3)
lines(avg_roc_xgb, col = "red", lwd=3)
legend("bottomright",legend=c("Logistic","GBM", "Random Forest",  "XGBoost"),
       col=c("blue", "green", "orange", "red"),lwd=3)
