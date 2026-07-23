#!/usr/bin/env python3
"""
Lasso Regression Module.

This module provides a utility function to instantiate an untrained
Lasso Regression model with L1 regularization using Scikit-Learn.

Functions:
    lasso_regression(random_state): Returns an untrained Lasso model instance
    configured with a specific random seed.
"""

from sklearn import linear_model


def lasso_regression(random_state):
    """
    Creates and returns an untrained Lasso Regression model.
    Arguments:
        random_state: (int) used to set the random seed for reproducibility.
    Returns:
        model: An instance of sklearn.linear_model.Lasso
    """
    return linear_model.Lasso(random_state=random_state)
