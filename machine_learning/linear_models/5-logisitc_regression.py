#!/usr/bin/env python3
"""
Logistic Regression Classifier Module.

This module provides a utility function to instantiate an untrained
Logistic Regression model for binary and multiclass classification tasks
using Scikit-Learn.

Functions:
    Logistic_Regression_Model(random_state): Returns an untrained
    LogisticRegression model instance configured with a specific random seed.
"""

from sklearn import linear_model


def Logistic_Regression_Model(random_state):
    """
    Creates and returns an untrained Logistic Regression model.
    Arguments:
        random_state (int) used to set the random seed for reproducibility.
    Returns:
        model: An instance of sklearn.linear_model.LogisticRegression
    """
    return linear_model.LogisticRegression(random_state=random_state)
