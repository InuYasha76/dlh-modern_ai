#!/usr/bin/env python3
"""Logistic Regression Classifier Module."""

from sklearn import linear_model


def Logistic_Regression_Model(random_state):
    """
    Creates and returns an untrained Logistic Regression model.
    Arguments:
        random_stat (int) used to set the random seed for reproducibility.
    Returns:
        model: An instance of sklearn.linear_model.LogisticRegression
    """
    return linear_model.LogisticRegression(random_state=random_state)
