# House Price Prediction Using Machine Learning

## Overview

This project uses machine learning regression algorithms to predict residential house prices based on property characteristics, size, condition, and location.

The goal is to train and compare multiple regression models, evaluate their performance, and identify which model provides the most reliable house price predictions.

## Objectives

- Build a machine learning model that predicts house prices.
- Compare five regression algorithms using the same dataset and train/test split.
- Apply data preprocessing and feature engineering.
- Optimize model hyperparameters using cross-validation.
- Evaluate models using R², Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).
- Generate predictions for new houses using specified property features.

## Machine Learning Algorithms

This project implements the following regression algorithms:

| Algorithm | Description |
|---|---|
| Linear Regression | Establishes a baseline by modeling linear relationships between features and house prices. |
| Decision Tree Regressor | Learns nonlinear relationships using decision rules. |
| Random Forest Regressor | Combines multiple decision trees to improve predictive performance. |
| Support Vector Regression (SVR) | Uses support vector methods to model relationships between features and continuous house prices. |
| XGBoost Regressor | Uses gradient-boosted decision trees to improve predictions iteratively. |

## Dataset

The project uses `dataset.csv`, which contains housing information.

Example features include:

- `bedrooms` — Number of bedrooms
- `bathrooms` — Number of bathrooms
- `sqft_living` — Living area in square feet
- `sqft_lot` — Lot size in square feet
- `floors` — Number of floors
- `waterfront` — Whether the property has waterfront access
- `view` — View rating
- `condition` — Property condition
- `sqft_above` — Above-ground living area
- `sqft_basement` — Basement area
- `yr_built` — Year the property was built
- `yr_renovated` — Year of renovation
- `city` — Property city
- `statezip` — State and ZIP code
- `price` — Target variable representing the house price

The exact feature names depend on the columns available in the dataset.

## Project Workflow

### 1. Data Cleaning

- Load the dataset using Pandas.
- Convert house prices to numeric values.
- Remove rows with missing or nonpositive target prices.
- Extract the sale year and month from the date column.
- Remove `price_per_sqft` to avoid potential target leakage.
- Exclude the high-cardinality `street` feature in the initial model comparison.

### 2. Data Preprocessing

- Separate input features (`X`) from the target (`y`).
- Split the dataset into training and testing sets.
- Impute missing numerical values using the median.
- Impute missing categorical values using the most frequent value.
- Apply one-hot encoding to categorical variables.
- Scale numerical features for algorithms that are sensitive to feature magnitude.

Preprocessing is implemented using Scikit-learn pipelines to keep transformations consistent during training, cross-validation, and prediction.

### 3. Model Training and Hyperparameter Tuning

Use `RandomizedSearchCV` to test different hyperparameter combinations for Decision Tree, Random Forest, SVR, and XGBoost.

Examples of tuned hyperparameters include:

- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `n_estimators`
- `max_features`
- `C`
- `epsilon`
- `gamma`
- `learning_rate`
- `subsample`

Linear Regression serves as a baseline model.

Three-fold cross-validation is used in the initial comparison to estimate model performance while limiting training time.

### 4. Model Evaluation

Each model is evaluated using the following metrics:

| Metric | Purpose | Preferred result |
|---|---|---|
| R² | Measures how much variation in house prices is explained by the model | Higher |
| MAE | Measures average absolute prediction error in dollars | Lower |
| RMSE | Penalizes larger prediction errors more heavily | Lower |

The models are compared using cross-validation R², training R², test R², MAE, and RMSE.

Cross-validation performance is used to identify promising models, while the held-out test set provides a final estimate of generalization performance.

### 5. House Price Prediction

After training, the saved in-memory model pipelines can predict the price of a new house using specified property features.

Example input:

```python
new_house = {
    "bedrooms": 3,
    "bathrooms": 2.5,
    "sqft_living": 1800,
    "sqft_lot": 5000,
    "floors": 2,
    "waterfront": 0,
    "view": 0,
    "condition": 3,
    "sqft_above": 1500,
    "sqft_basement": 300,
    "yr_built": 1995,
    "yr_renovated": 0,
    "city": "Seattle",
    "statezip": "WA 98103",
    "sale_year": 2014,
    "sale_month": 6
}
```

The input columns must match the feature names used during model training. Any missing input features must be handled consistently with the preprocessing pipeline.

The model returns an estimated house price based on patterns learned from the training data. Predictions are estimates, not guaranteed market values.

## Technologies Used

- **Python** — Main programming language
- **Pandas** — Data loading and manipulation
- **NumPy** — Numerical operations
- **Scikit-learn** — Preprocessing, regression models, cross-validation, and evaluation
- **XGBoost** — Gradient-boosted regression
- **Matplotlib** — Model comparison and prediction visualizations

## Link for the dataset:
- https://www.kaggle.com/datasets/debayank2024/house-price-prediction
