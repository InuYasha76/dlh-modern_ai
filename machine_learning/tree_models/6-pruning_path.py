#!/usr/bin/env python3
"""Retrieve the cost-complexity pruning path of a decision tree."""


def get_pruning_path(clf, X, y):
    """Retrieve cost-complexity pruning path for a decision tree classifier.

    Args:
        clf: A DecisionTreeClassifier instance.
        X: Input features.
        y: Target labels.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: A tuple containing:
            - ccp_alphas: Effective alpha values used for pruning.
            - impurities: Total impurity of leaves at each alpha.
    """
    path = clf.cost_complexity_pruning_path(X, y)
    return path.ccp_alphas, path.impurities
