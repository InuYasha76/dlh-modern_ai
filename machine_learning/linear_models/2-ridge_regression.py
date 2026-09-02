#!/usr/bin/env python3
"""Build and return an untrained Ridge Regression model."""
from sklearn import linear_model


def ridge_regression(random_state):
    """Create and return an untrained Ridge Regression model.

    Args:
        random_state (int): Seed used by the random number generator.

    Returns:
        linear_model.Ridge: An untrained Ridge regression model.
    """
    return linear_model.Ridge(random_state=random_state)
