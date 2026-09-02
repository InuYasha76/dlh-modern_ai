#!/usr/bin/env python3
"""Build and return a linear regression model using Scikit-learn."""
from sklearn import linear_model


def Linear_Regression():
    """Create and return a LinearRegression instance using OLS.

    Returns:
        linear_model.LinearRegression: An untrained LinearRegression model.
    """
    return linear_model.LinearRegression()
