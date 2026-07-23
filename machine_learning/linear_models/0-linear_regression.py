#!/usr/bin/env python3
"""This module is about linear regression."""
from sklearn import linear_model


def Linear_Regression():
    """
    Creates and returns a linear regression model using Scikit-learn
    which uses ordinary least squares to fit a linear model to the data.
    Returns:
        model: An instance of sklearn.linear_model.LinearRegression
    """
    return linear_model.LinearRegression()
