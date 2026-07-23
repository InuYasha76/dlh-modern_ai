#!/usr/bin/env python3
"""
Regression Evaluation Metrics Module.
This module provides a utility function to compute fundamental evaluation
metrics (MSE, RMSE, MAE, and R^2 Score) for regression models using
Scikit-Learn and NumPy.

Functions:
    evaluation_metrics_for_regression(y_true, y_pred): Calculates and
    returns regression performance metrics as a tuple.
"""

from sklearn import metrics
import numpy as np


def evaluation_metrics_for_regression(y_true, y_pred):
    """
    Computes common regression evaluation metrics.
    Arguments:
        y_true: 1D NumPy array containing true target values.
        y_pred: 1D NumPy array containing predicted target values.
    Returns:
        tuple: (mse, rmse, mae, r2)
            - mse: Mean Squared Error
            - rmse: Root Mean Squared Error
            - mae: Mean Absolute Error
            - r2: R² Score (Coefficient of Determination)
    """
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)

    return mse, rmse, mae, r2
