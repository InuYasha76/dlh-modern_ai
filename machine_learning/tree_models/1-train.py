#!/usr/bin/env python3
"""Train a tree-based classifier using Scikit-learn."""


def train_tree(clf, X, y):
    """Train a tree-based classifier using the provided features and labels.

    Args:
        clf: A Scikit-learn tree classifier instance (e.g.,
            DecisionTreeClassifier).
        X: The input features.
        y: The target labels.

    Returns:
        None
    """
    clf.fit(X, y)
