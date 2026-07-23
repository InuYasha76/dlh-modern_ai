#!/usr/bin/env python3
"""
SHAP Model Explainability Module.

This module provides utility functions to compute SHAP (SHapley Additive
exPlanations) values to explain regression model predictions using training
and test datasets.

Functions:
    get_shap_explainer_and_values(model, X_train, X_test): Instantiates
    a SHAP explainer on training background data and computes SHAP values
    for test samples.
"""

import shap


def get_shap_explainer_and_values(model, X_train, X_test):
    """
    Creates a SHAP explainer using background training data and calculates
    SHAP values for the test set.
    Arguments:
        model: A trained regression model (standard, Ridge, Lasso, Tree-based).
        X_train: Array or DataFrame input data used as the background dataset.
        X_test: Array-like or DataFrame input data to compute explanations for.
    Returns:
        tuple: (explainer, shap_values)
            - explainer: The initialized shap.Explainer instance.
            - shap_values: The computed SHAP values for X_test predictions.
    """
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    return explainer, shap_values
