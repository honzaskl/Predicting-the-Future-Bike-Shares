# Install and load required libraries
install.packages(c("tidyverse", "recipes", "caret", "lubridate", "xgboost", "writexl"))
library(tidyverse)
library(recipes)
library(caret)
library(lubridate)
library(xgboost)
library(writexl)

# Load dataset
data <- read.csv("C:\\Users\\Andy\\Downloads\\london_merged.csv")

#Distribution plots
options(repr.plot.width=18, repr.plot.height=6)
par(mfrow = c(1, 3))
hist(data$cnt, xlab = "Number of bike shares", col = "steelblue")
hist(data$t1, xlab = "Temperature (◦C)", col = "steelblue")
hist(data$t2, xlab = "Feels like temperature (◦C)", col = "steelblue")

options(repr.plot.width=18, repr.plot.height=6)
par(mfrow = c(1, 3))
hist(data$hum, xlab = "Humidity (%)", col = "steelblue")
hist(data$wind_speed, xlab = "Wind speed (km/h)", col = "steelblue")
barplot(table(data$is_holiday),
        xlab ="Is holiday?",
        ylab ="Count", col = c("steelblue", "lightcoral"),
        names.arg = c("non-Holiday", "Holiday"))

options(repr.plot.width=18, repr.plot.height=6)
par(mfrow = c(1, 3))
barplot(table(data$is_weekend),
        xlab = "Is weekend day?",
        ylab = "Count", col = c("steelblue", "lightcoral"),
        names.arg = c("Weekday", "Weekend"))
barplot(table(data$season),
        xlab ="Season",
        ylab ="Count", col = c("steelblue", "lightcoral"),
        names.arg = c("Spring", "Summer", "Autumn", "Winter"))
barplot(table(data$weather_code),
        xlab = "Weather",
        ylab = "Count", col = c("steelblue", "lightcoral"))


# Preprocessing
data$humidity <- data$hum / 100  # Convert humidity to 0-1 scale
data$hour <- hour(data$timestamp)  # Extract hour from timestamp
data <- subset(data, select = -timestamp)  # Remove timestamp

# Define preprocessing pipeline
preprocessing_pipeline <- function(data) {
  numerical_features <- c("t1", "t2", "humidity", "wind_speed")
  categorical_features <- c("season", "weather_code")

  rental_recipe <- recipe(cnt ~ ., data = data) %>%
    step_mutate(across(all_of(categorical_features), as.factor)) %>%
    step_dummy(all_of(categorical_features), one_hot = TRUE) %>%
    step_ns(hour, deg_free = 5) %>%
    step_normalize(all_numeric_predictors(), -all_of("humidity")) %>%
    step_center(all_numeric_predictors()) %>%
    step_scale(all_numeric_predictors())

  prepped_data <- prep(rental_recipe, training = data)
  processed_data <- bake(prepped_data, new_data = NULL)

  return(list(recipe = rental_recipe, processed_data = processed_data))
}

# Run preprocessing
processed <- preprocessing_pipeline(data)
data1 <- processed$processed_data

# Add polynomial features manually
data1 <- data1 %>%
  mutate(
    t1_sq = t1^2,
    t2_sq = t2^2,
    wind_speed_sq = wind_speed^2
  )

View(data1)


# ===============================
# 🚀 Train-Test Split
# ===============================
set.seed(123)
train_index <- createDataPartition(data1$cnt, p = 0.8, list = FALSE)
train_data <- data1[train_index, ]
test_data <- data1[-train_index, ]

# Separate features (X) and target variable (y)
X_train <- train_data %>% select(-cnt)
y_train <- train_data$cnt
X_test <- test_data %>% select(-cnt)
y_test <- test_data$cnt

# Convert data to matrix format for XGBoost
X_train_matrix <- as.matrix(X_train)
X_test_matrix <- as.matrix(X_test)

# ===============================
# 🚀 XGBoost Model Training
# ===============================

#The whole grid search comes with hashtags for technical reasons (running takes 
#too long), but serves to find best hyperparameters... similar later for RF

# Define parameter grid
#xgb_grid <- expand.grid(
#  nrounds = seq(100, 500, 100),  # Number of boosting rounds
# eta = c(0.01, 0.1, 0.3),  # Learning rate
# max_depth = c(3, 6, 9),  # Tree depth
# gamma = c(0, 1, 5),  # Minimum loss reduction
# colsample_bytree = c(0.5, 0.8, 1),  # Subsample ratio of columns
# min_child_weight = c(1, 5, 10),
# subsample = c(0.5, 0.8, 1)# Minimum sum of instance weight
#)

# Train XGBoost using cross-validation
#xgb_control <- trainControl(method = "cv", number = 5, verboseIter = TRUE)

#xgb_model <- train(
# x = X_train_matrix, y = y_train,
# method = "xgbTree",
# trControl = xgb_control,
# tuneGrid = xgb_grid,
# metric = "RMSE"
#)

# Print best parameters
# print("Best XGBoost Parameters:", xgb_model$bestTune, "\n")

# ===============================
# 🚀 Model Evaluation
# ===============================

# Predict on test data
# y_pred_xgb <- predict(xgb_model, newdata = X_test_matrix)

# Compute RMSE
# rmse_xgb <- sqrt(mean((y_pred_xgb - y_test)^2))
# cat("XGBoost Test RMSE:", rmse_xgb, "\n")
###########################################################################################################x
# XGBOOST with best parameters
##########################################################################################################xx
library(xgboost)
library(caret)

# 🚀 Define best parameters from grid search
best_params <- list(
  nrounds = 1000,
  max_depth = 9,
  eta = 0.01,
  gamma = 1,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 0.5
)

# 🚀 Train XGBoost model with best parameters
xgb_model_best <- xgboost(
  data = X_train_matrix,
  label = y_train,
  nrounds = best_params$nrounds,
  max_depth = best_params$max_depth,
  eta = best_params$eta,
  gamma = best_params$gamma,
  colsample_bytree = best_params$colsample_bytree,
  min_child_weight = best_params$min_child_weight,
  subsample = best_params$subsample,
  objective = "reg:squarederror",
  verbose = 1
)

# ===============================
# 🚀 Model Evaluation
# ===============================

# Predict on test data
y_pred_xgb_best <- predict(xgb_model_best, newdata = X_test_matrix)

# Compute RMSE
rmse_xgb_best <- sqrt(mean((y_pred_xgb_best - y_test)^2))
cat("XGBoost Test RMSE:", rmse_xgb_best, "\n")
#############################################################################################################
# Define the parameter grid
library(glmnet)
library(caret)

# Define alpha values
alpha_values <- seq(0, 1, by = 0.05)

# Train Elastic Net using cross-validation (ONLY tuning alpha)
enet_control <- trainControl(method = "cv", number = 5, verboseIter = TRUE)

enet_alpha_model <- train(
  x = X_train_matrix, y = y_train,
  method = "glmnet",
  trControl = enet_control,
  tuneGrid = expand.grid(alpha = alpha_values, lambda = 0),  # Placeholder lambda
  metric = "RMSE"
)

# Get best alpha
best_alpha <- enet_alpha_model$bestTune$alpha
cat("Best Alpha:", best_alpha, "\n")

# Define full lambda sequence (log-spaced)
lambda_values <- exp(seq(log(0.0001), log(1000), length.out = 500))

# Train glmnet manually using best alpha and all lambdas
enet_final_model <- glmnet(
  x = X_train_matrix, y = y_train,
  alpha = best_alpha,
  lambda = lambda_values
)

# Perform cross-validation to select best lambda
cv_enet <- cv.glmnet(
  x = X_train_matrix, y = y_train,
  alpha = best_alpha,
  lambda = lambda_values,
  nfolds = 5
)

# Get best lambda from cross-validation
best_lambda <- cv_enet$lambda.min
cat("Best Lambda:", best_lambda, "\n")

# ===============================
# 🚀 Model Evaluation
# ===============================

# Predict using best model
y_pred_enet <- predict(enet_final_model, newx = X_test_matrix, s = best_lambda)

# Compute RMSE
rmse_enet <- sqrt(mean((y_pred_enet - y_test)^2))
cat("Elastic Net Test RMSE:", rmse_enet, "\n")
###################################################################################

library(glmnet)
library(caret)

# Define alpha values
alpha_values <- seq(0, 1, by = 0.05)

# Train Poisson Elastic Net using cross-validation (ONLY tuning alpha)
poisson_control <- trainControl(method = "cv", number = 5, verboseIter = TRUE)

poisson_alpha_model <- train(
  x = X_train_matrix, y = y_train,
  method = "glmnet",
  trControl = poisson_control,
  tuneGrid = expand.grid(alpha = alpha_values, lambda = 0),  # Placeholder lambda
  metric = "RMSE",
  family = "poisson"
)

# Get best alpha
best_alpha_poisson <- poisson_alpha_model$bestTune$alpha
cat("Best Alpha for Poisson Model:", best_alpha_poisson, "\n")

# Define full lambda sequence (log-spaced)
lambda_values <- exp(seq(log(0.0001), log(1000), length.out = 500))

# Train glmnet manually using best alpha and all lambdas
poisson_final_model <- glmnet(
  x = X_train_matrix, y = y_train,
  alpha = best_alpha_poisson,
  lambda = lambda_values,
  family = "poisson"
)

# Perform cross-validation to select best lambda
cv_poisson <- cv.glmnet(
  x = X_train_matrix, y = y_train,
  alpha = best_alpha_poisson,
  lambda = lambda_values,
  nfolds = 5,
  family = "poisson"
)

# Get best lambda from cross-validation
best_lambda_poisson <- cv_poisson$lambda.min
cat("Best Lambda for Poisson Model:", best_lambda_poisson, "\n")

# ===============================
# 🚀 Model Evaluation
# ===============================

# Predict using best model
y_pred_poisson <- predict(poisson_final_model, newx = X_test_matrix, s = best_lambda_poisson, type = "response")

# Compute RMSE
rmse_poisson <- sqrt(mean((y_pred_poisson - y_test)^2))
cat("Poisson Elastic Net Test RMSE:", rmse_poisson, "\n")
#######################################################################################x
# Load required libraries
install.packages(c("randomForest", "caret", "e1071"))
library(randomForest)
library(caret)
library(e1071)
# Remove hour_ns columns from X_train_matrix
X_train_matrix_rf <- X_train_matrix[, !colnames(X_train_matrix) %in% c("hour_ns_1", "hour_ns_2", "hour_ns_3", "hour_ns_4", "hour_ns_5")]

# Similarly, remove from X_test_matrix if necessary
X_test_matrix_rf <- X_test_matrix[, !colnames(X_test_matrix) %in% c("hour_ns_1", "hour_ns_2", "hour_ns_3", "hour_ns_4", "hour_ns_5")]

# ===============================
# 🚀 Hyperparameter Grid Search
# ===============================
# set.seed(123)

# Define the parameter grid
#rf_grid <- expand.grid(
# mtry = seq(2, ncol(X_train_matrix), by = 2),  # Number of features tried at each split
# splitrule = c("variance"),  # Only "variance" for regression
# min.node.size = c( 10, 15,25,30)  # Minimum node size (smaller = more complex trees)
# )

# Define cross-validation method
# rf_control <- trainControl(method = "cv", number = 5, verboseIter = TRUE)

# Train Random Forest model
# rf_model <- train(
#  x = X_train_matrix, y = y_train,
#  method = "ranger",  # Fast Random Forest implementation
#  trControl = rf_control,
#  tuneGrid = rf_grid,
#  metric = "RMSE"
# )

# Print best parameters
# cat("Best Random Forest Parameters:", rf_model$bestTune, "\n")

# ===============================
# 🚀 Model Evaluation
# ===============================

# Predict on test data
# <- predict(rf_model, newdata = X_test_matrix)

# Compute RMSE
# rmse_rf <- sqrt(mean((y_pred_rf - y_test)^2))
# cat("Random Forest Test RMSE:", rmse_rf, "\n")
###############################################################################################
# RF with best parameters
###################################################################
# Load necessary library
library(ranger)

# Train Random Forest with best parameters
set.seed(123)
rf_model_best <- ranger(
  formula = cnt ~ .,
  data = train_data,
  num.trees = 500,  # Default or increase for more stability
  mtry = 25,  # Best found number of features tried at each split
  splitrule = "variance",  # Best split rule for regression
  min.node.size = 10,  # Best found minimum node size
  importance = "impurity"  # Variable importance calculation
)

# Predict on test set
y_pred_rf_best <- predict(rf_model_best, data = test_data)$predictions

# Compute RMSE
rmse_rf_best <- sqrt(mean((y_pred_rf_best - y_test)^2))
cat("Optimized Random Forest Test RMSE:", rmse_rf_best, "\n")







