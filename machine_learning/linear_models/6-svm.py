#!/usr/bin/env python3
"""Build and return an untrained Support Vector Classifier (SVC)."""
from sklearn import svm


def get_SVM_model(name, random_state):
    """Create and return an untrained Support Vector Classifier (SVC).

    Args:
        name (str): Kernel type (e.g., 'linear', 'poly', or 'rbf').
        random_state (int): Seed used by the random number generator.

    Returns:
        svm.SVC: An untrained Support Vector Classifier model.
    """
    if isinstance(name, str):
        name = name.strip().lower()
    return svm.SVC(kernel=name, random_state=random_state)
