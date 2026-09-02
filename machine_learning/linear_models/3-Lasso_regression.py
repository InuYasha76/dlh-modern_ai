#!/usr/bin/env python3
"""Build and return an untrained Lasso Regression model."""
from sklearn import linear_model


def lasso_regression(random_state):
    """Create and return an untrained Lasso Regression model.

    Args:
        random_state (int): Seed used by the random number generator.

    Returns:
        linear_model.Lasso: An untrained Lasso regression model.
    """
    return linear_model.Lasso(random_state=random_state)
