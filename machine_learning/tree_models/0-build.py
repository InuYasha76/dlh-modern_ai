#!/usr/bin/env python3
"""
This module provides a function to build a decision tree classifier
using Scikit-learn.
"""
from sklearn import tree


def build_decision_tree(min_samples_leaf, min_samples_split, random_state):
    """
    Creates and returns a Scikit-learn DecisionTreeClassifier instance.
    Arguments:
        min_samples_leaf (int): The minimum number of samples required to be
                                at a leaf node.
        min_samples_split (int): The minimum number of samples required to
                                 split an internal node.
        random_state (int): Seed used by the random number generator.
    Returns:
        tree.DecisionTreeClassifier: A configured decision tree model.
    """
    model = tree.DecisionTreeClassifier(
        criterion="gini",
        max_depth=None,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        random_state=random_state,
    )
    return model
