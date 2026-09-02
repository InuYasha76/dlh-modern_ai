#!/usr/bin/env python3
"""Build and return an untrained Logistic Regression model."""
from sklearn import linear_model


def Logistic_Regression_Model(random_state):
    """Create and return an untrained Logistic Regression model.

    Args:
        random_state (int): Seed used by the random number generator.

    Returns:
        linear_model.LogisticRegression: An untrained LogisticRegression model.
    """
    return linear_model.LogisticRegression(random_state=random_state)
