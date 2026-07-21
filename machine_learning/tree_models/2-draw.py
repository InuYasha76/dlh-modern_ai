#!/usr/bin/env python3
"""
This module provides a function to display the textual decision structure of a
decision tree classifier using Scikit-learn.
"""
from sklearn import tree


def draw(clf, feature_names, class_names):
    """
    Prints a clean, text-based flowchart representation of a decision tree.
    Arguments:
        clf: A trained DecisionTreeClassifier instance from Scikit-learn.
        feature_names (list): A list of strings matching the input features.
        class_names (list): A list of strings matching the target classes.
    Returns:
        None
    """
    tree_rules = tree.export_text(
        clf, feature_names=list(feature_names), class_names=list(class_names)
    )
    print(tree_rules, end="")
