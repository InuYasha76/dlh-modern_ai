#!/usr/bin/env python3
"""Create a Random Forest classifier."""
from sklearn import ensemble


def random_forest(n_estimators, random_state):
    """Create a configured RandomForestClassifier instance.

    Args:
        n_estimators (int): Number of trees in the forest.
        random_state (int): Seed used by the random number generator for
            reproducibility.

    Returns:
        ensemble.RandomForestClassifier: A configured Random Forest model.
    """
    return ensemble.RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state
    )
