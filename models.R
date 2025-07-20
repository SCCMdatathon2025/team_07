library(dplyr)
library(caret)
library(randomForest)
library(xgboost)
library(brms)
library(mgcv)
library(e1071)
library(pROC)


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
# Data preprocessing
################################################
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

set.seed(123)

# Assume df is already loaded and complete (imputed).

# Obesity indicator (BMI ≥ 30)
df$obese <- ifelse(df$bmi_avg >= 30, 1, 0)

# Convert numeric categorical variables to factors
df <- df %>% mutate(across(c(visit_occurrence_id, gender_concept_id, race_concept_id, ethnicity_concept_id), as.factor))

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
selected_features

################################################
# 5 fold CV
################################################
# Prepare final dataset with selected features
df_model <- df[,c("race_concept_id", "ethnicity_concept_id",
                  selected_features[-c(1,2,3)], "died")]
train_control <- trainControl(method="cv", number=5, classProbs=TRUE, summaryFunction=twoClassSummary)

# Convert outcome to factor for caret
df_model$died <- factor(ifelse(df_model$died==1, "Died", "Survived"), levels=c("Survived", "Died"))

# Store models and results
models_list <- list()
results <- data.frame()


################################################
# Models
################################################

set.seed(123)
models_list$Logistic <- train(died ~ ., data=df_model, method="glm", family="binomial",
                              trControl=train_control, metric="ROC")


set.seed(123)
models_list$DecisionTree <- train(died ~ ., data=df_model, method="rpart",
                                  trControl=train_control, metric="ROC")


set.seed(123)
models_list$RandomForest <- train(died ~ ., data=df_model, method="rf",
                                  trControl=train_control, metric="ROC")


set.seed(123)
models_list$GBM <- train(died ~ ., data=df_model, method="gbm",
                         trControl=train_control, metric="ROC", verbose=FALSE)


set.seed(123)
models_list$XGBoost <- train(died ~ ., data=df_model, method="xgbTree",
                             trControl=train_control, metric="ROC")


set.seed(123)
models_list$SVM <- train(died ~ ., data=df_model, method="svmRadial",
                         trControl=train_control, metric="ROC", preProcess=c("center","scale"))


set.seed(123)
models_list$NaiveBayes <- train(died ~ ., data=df_model, method="naive_bayes",
                                trControl=train_control, metric="ROC")


set.seed(123)
models_list$KNN <- train(died ~ ., data=df_model, method="knn",
                         trControl=train_control, metric="ROC", preProcess=c("center","scale"))


#####
# model comparison
#####

# Summarize model performances
model_resamples <- resamples(models_list)
summary(model_resamples)

# Boxplot comparison
bwplot(model_resamples, metric="ROC")
dotplot(model_resamples, metric="ROC")


#####
# visualization
#####
roc_list <- list()

for(model_name in names(models_list)) {
  pred <- predict(models_list[[model_name]], df_model, type="prob")
  roc_obj <- roc(df_model$died, pred$Died)
  roc_list[[model_name]] <- roc_obj
}

# Plot ROC curves
plot(roc_list[[1]], col=1, lwd=2, main="ROC Curves Comparison")
colors <- rainbow(length(roc_list))
i <- 1
for(name in names(roc_list)) {
  plot(roc_list[[name]], col=colors[i], add=TRUE, lwd=2)
  i <- i + 1
}
legend("bottomright", legend=names(roc_list), col=colors, lwd=2)

