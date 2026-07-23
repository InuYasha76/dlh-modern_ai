#!/usr/bin/env python3
"""SVM Classifier Module."""

from sklearn import svm


def get_SVM_model(name, random_state):
    """
    Creates and returns an untrained Support Vector Classifier (SVC).
    Arguments:
        name: A string indicating the kernel type ('linear', 'poly', or 'rbf').
        random_state: An integer used to set the random seed for reproducibility.
    Returns:
        model: An untrained instance of sklearn.svm.SVC
    """
    return svm.SVC(kernel=name, random_state=random_state)
