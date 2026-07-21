#!/usr/bin/env python3
"""
This module provides a function to train a Scikit-learn tree-based classifier
on a given dataset.
"""


def train_tree(clf, X, y):
    """
    Trains a tree-based classifier using the provided features and labels.
    Arguments:
        clf: A Scikit-learn tree classifier instance (e.g., DecisionTree).
        X: The input features.
        y: The target labels.
    Returns:
        None
    """
    clf.fit(X, y)
