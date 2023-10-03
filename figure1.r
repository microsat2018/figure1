# Install and load the e1071 package if you haven't already
# install.packages("e1071")
library(e1071)
# Install and load the randomForest package if you haven't already
# install.packages("randomForest")
library(randomForest)
# Install and load the xgboost package if you haven't already
# install.packages("xgboost")
library(xgboost)
# Install and load the glmnet package if you haven't already
# install.packages("glmnet")
library(glmnet)

MLMI <- function(data, missing_cols) {
    # Load your dataset with missing values (replace 'your_data.csv' with your dataset)
    # data <- read.csv("your_data.csv")

    pre_data <- sapply(data, function(x) {
        if (class(x) != "factor") {
            x[is.na(x)] <- mean(x, na.rm = TRUE)
        }
        return(x)
    }, simplify = "data.frame")

    # Loop through each column with missing values and impute using SVM
    for (col in missing_cols) {
        # Split the data into two sets: one with missing values and one without
        data_with_missing <- pre_data[is.na(data[, col]), ]
        data_without_missing <- pre_data[!is.na(data[, col]), ]

        # Train an SVM model to predict the missing column
        svm_model <- svm(data_without_missing[, col] ~ ., data = data_without_missing)

        # Predict the missing values
        imputed_values_svm <- predict(svm_model, newdata = data_with_missing)

        # Train a Random Forest model to predict the missing column
        rf_model <- randomForest(data_without_missing[, col] ~ ., data = data_without_missing)

        # Predict the missing values
        imputed_values_rf <- predict(rf_model, newdata = data_with_missing)

        # Train an xgboost model to predict the missing column
        xgb_model <- xgboost(
            data = as.matrix(data_without_missing[, setdiff(names(data), col)]),
            label = data_without_missing[, col],
            nrounds = 100, # Adjust as needed
            verbose = 0
        )

        # Predict the missing values
        imputed_values_xgb <- predict(xgb_model, as.matrix(data_with_missing[, setdiff(names(data), col)]))

        # Train a glmnet model to predict the missing column
        glmnet_model <- glmnet(
            x = as.matrix(data_without_missing[, setdiff(names(data), col)]),
            y = data_without_missing[, col], lambda = 0
        )

        # Predict the missing values
        imputed_values_glmnet <- predict(glmnet_model, newx = as.matrix(data_with_missing[, setdiff(names(data), col)]))

        imputed_values <- (imputed_values_svm + imputed_values_rf + imputed_values_xgb + imputed_values_glmnet) / 4

        # Replace missing values with imputed values in the original dataset
        data[is.na(data[, col]), col] <- imputed_values
    }
    return(data)
}

######################
###### testing begin
# simulated data with 18 features and 1 binary target variable.
# And the target variable (the one you want to predict) is named 'target_variable'
data <- read.table("./simulated_data.txt", head = TRUE)
data$target_variable <- as.factor(data$target_variable)

# Identify columns with missing values
missing_cols <- which(apply(data, 2, function(x) any(is.na(x))))

imputed_data <- MLMI(data, missing_cols)

print(imputed_data)
