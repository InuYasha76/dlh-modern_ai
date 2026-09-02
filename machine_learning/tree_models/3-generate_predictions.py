#!/usr/bin/env python3
"""Generate predictions from a trained tree-based classifier."""


def generate_predictions(clf, X):
    """Generate class label predictions for the provided input feature matrix.

    Args:
        clf: A trained Scikit-learn tree-based classifier instance.
        X: Feature matrix (NumPy array or pandas DataFrame).

    Returns:
        numpy.ndarray: Predicted class labels for the samples.
    """
    return clf.predict(X)
