#!/usr/bin/env python3
"""Compute evaluation metrics for regression models."""
from sklearn import metrics
import numpy as np


def evaluation_metrics_for_regression(y_true, y_pred):
    """Compute common regression evaluation metrics.

    Args:
        y_true: 1D NumPy array containing true target values.
        y_pred: 1D NumPy array containing predicted target values.

    Returns:
        tuple[float, numpy.float64, float, float]: A tuple containing:
            - mse: Mean Squared Error.
            - rmse: Root Mean Squared Error.
            - mae: Mean Absolute Error.
            - r2: R² Score (Coefficient of Determination).
    """
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)

    return mse, rmse, mae, r2
