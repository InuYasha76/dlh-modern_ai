# Linear Models

This module provides implementations and utilities for linear regression, evaluation metrics, regularized regression (Ridge, Lasso), SHAP model interpretability, logistic regression, and Support Vector Machines (SVM).

## Overview of Modules

| Script | Function | Description |
| :--- | :--- | :--- |
| `0-linear_regression.py` | `Linear_Regression` | Creates an untrained Linear Regression model (OLS). |
| `1-regression_evaluation_metrics.py` | `evaluation_metrics_for_regression` | Calculates MSE, RMSE, MAE, and R² evaluation metrics. |
| `2-ridge_regression.py` | `ridge_regression` | Creates an untrained Ridge regression model (L2 penalty). |
| `3-Lasso_regression.py` | `lasso_regression` | Creates an untrained Lasso regression model (L1 penalty). |
| `4-shap.py` | `get_shap_explainer_and_values` | Computes SHAP explainer and feature attribution values. |
| `5-logisitc_regression.py` | `Logistic_Regression_Model` | Creates an untrained Logistic Regression classifier. |
| `6-svm.py` | `get_SVM_model` | Creates an untrained Support Vector Classifier (SVC). |

## Setup

Create `.venv` from the repository root and install dependencies with:

```bash
/usr/bin/python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

