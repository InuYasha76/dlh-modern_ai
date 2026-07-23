#!/usr/bin/env python3
"""
Ridge Regression Module.

This module provides a utility function to instantiate an untrained
Ridge Regression model with L2 regularization using Scikit-Learn.

Functions:
    ridge_regression(random_state): Returns an untrained Ridge model instance
    configured with a specific random seed.
"""

from sklearn import linear_model


def ridge_regression(random_state):
    """
    Creates and returns an untrained Ridge Regression model.
    Arguments:
        random_state (int) used to set the random seed for reproducibility.
    Returns:
        model: An instance of sklearn.linear_model.Ridge
    """
    return linear_model.Ridge(random_state=random_state)
