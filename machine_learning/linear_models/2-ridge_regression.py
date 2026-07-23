#!/usr/bin/env python3
"""Ridge Regression Module."""

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
