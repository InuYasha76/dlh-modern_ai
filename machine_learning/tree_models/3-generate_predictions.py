#!/usr/bin/env python3
"""
This module provides a function to generate predictions from a trained
tree-based classifier using Scikit-learn
"""


def generate_predictions(clf, X):
    """
    Generates class label predictions for the provided input feature matrix.
    Arguments:
        clf: A trained Scikit-learn tree-based classifier instance.
        X: Feature matrix (NumPy array or pandas DataFrame).
    Returns:
        A NumPy array containing the predicted class labels for the samples.
    """
    return clf.predict(X)
