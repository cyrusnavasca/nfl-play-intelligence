# Feature Selection Plan

## Objective

The goal of feature selection is to identify the subset of engineered features that maximizes predictive performance while improving model interpretability, reducing overfitting, and simplifying the final model.

Feature selection will be performed **after feature engineering** and **before final model training**.

---

# Stage 0. Binary Feature Selection (Run vs. Pass Model)

The project will follow a **two-stage modeling pipeline**:

```text
Pre-Snap Features
        ↓
Run / Pass Probability Model
        ↓
Predicted P(Run), P(Pass)
        ↓
Expected Yards Model
```

The output probabilities from the first-stage classifier will become engineered features for the expected yards regression model.

Since this stage is a **binary classification** problem, additional feature selection techniques specific to classification can be evaluated here.

## Candidate Methods

* Weight of Evidence (WoE)
* Information Value (IV)
* XGBoost Feature Importance
* Permutation Importance

The selected features and predicted probabilities will then be passed into the expected yards model.

---

# Pipeline

```text
Clean Data
      ↓
Feature Engineering
      ↓
Feature Selection
      ↓
Run / Pass Model
      ↓
Predicted Run / Pass Probabilities
      ↓
Expected Yards Model
      ↓
Model Evaluation
```

---

# 1. Low Variance Filter

Remove features that contain little to no variation.

Examples include:

* Constant features
* Nearly constant indicators
* Features with extremely skewed value distributions

Purpose:

* Reduce dimensionality
* Remove non-informative variables
* Improve training efficiency

---

# 2. Missing Value Analysis

Evaluate each feature for missingness.

For each feature:

* Percentage of missing values
* Missingness pattern
* Appropriate handling strategy:

  * Drop
  * Impute
  * Retain

---

# 3. XGBoost Feature Importance

Train a baseline XGBoost model using the engineered feature set.

Extract feature importance using:

* Gain
* Weight
* Cover

Primary metric:

* Gain

Goals:

* Identify highly predictive features
* Remove consistently uninformative variables

---

# 4. Variance Inflation Factor (VIF)

Compute the Variance Inflation Factor for numerical features to identify multicollinearity.

Goals:

* Detect redundant predictors
* Reduce instability in feature estimates
* Improve overall model robustness

Typical guideline:

* VIF > 5: investigate
* VIF > 10: strong candidate for removal

Since tree-based models are generally robust to correlated predictors, VIF analysis will primarily be used to simplify the feature set and improve interpretability rather than as a strict filtering criterion.

---

# 5. Permutation Importance

Measure the decrease in validation performance when each feature is randomly shuffled.

Advantages:

* Model agnostic
* Measures true predictive contribution
* Less biased than built-in tree importance

Evaluation Metrics:

* Increase in MAE
* Increase in RMSE

---

# 6. Recursive Feature Elimination (Optional)

Iteratively remove the least useful features while monitoring validation performance.

Purpose:

* Determine the smallest feature subset that maintains predictive performance.

---

# 7. Final Feature Set

Select the smallest feature subset that achieves near-optimal validation performance.

Save:

* `selected_features.json`
* `feature_importance.csv`

The selected feature list will be used consistently across both modeling stages where applicable.

---

# Deliverables

## Tables

* XGBoost Feature Importance
* Permutation Importance
* VIF Summary

## Saved Artifacts

```text
artifacts/
├── feature_importance.csv
├── permutation_importance.csv
├── vif_scores.csv
└── selected_features.json
```

---

# Evaluation Criteria

Feature selection should improve or maintain:

* Validation MAE
* Validation RMSE
* Classification Accuracy (Run/Pass Model)
* ROC-AUC (Run/Pass Model)
* Model interpretability
* Training efficiency

while reducing:

* Redundant features
* Noise
* Overfitting

---

# Future Extensions

Additional feature selection methods may be explored for comparison:

* Mutual Information
* Boruta
* Recursive Feature Elimination with Cross Validation (RFECV)
* Sequential Feature Selection
* Stability Selection
* Bayesian Feature Selection
