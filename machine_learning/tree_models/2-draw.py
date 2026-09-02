#!/usr/bin/env python3
"""Display the textual decision structure of a decision tree classifier."""
from sklearn import tree


def draw(clf, feature_names, class_names):
    """Print a clean, text-based flowchart representation of a decision tree.

    Args:
        clf: A trained DecisionTreeClassifier instance from Scikit-learn.
        feature_names (list): A list of strings matching the input features.
        class_names (list): A list of strings matching the target classes.

    Returns:
        None
    """
    f_names = list(feature_names) if feature_names is not None else None
    c_names = list(class_names) if class_names is not None else None
    tree_rules = tree.export_text(
        clf, feature_names=f_names, class_names=c_names
    )
    print(tree_rules)
