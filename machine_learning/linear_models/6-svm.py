#!/usr/bin/env python3
"""Support Vector Classification Module."""

from sklearn import svm


def Logistic_Regression_Model(random_state):
    """
    Creates and returns an untrained Support Vector Classifier configured
    with a linear kernel to perform binary classification.
    Arguments:
        random_state (int) used to set the random seed for reproducibility.
    Returns:
        model: An instance of sklearn.svm.SVC
    """
    v = svm.SVC(kernel='linear', probability=True, random_state=random_state)
    return v
