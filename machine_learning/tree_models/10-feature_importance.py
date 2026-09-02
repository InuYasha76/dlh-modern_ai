#!/usr/bin/env python3
"""Compute feature importances from a trained Random Forest model."""
import numpy as np


def feature_importance(rf):
    """Compute and return feature importances from a Random Forest model.

    Args:
        rf: A trained Scikit-learn RandomForestClassifier instance.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: A tuple containing:
            - importances: Feature importance scores.
            - indices: Feature indices sorted from least to most important
              (ascending order).
    """
    importances = rf.feature_importances_
    indices = np.argsort(importances)
    return importances, indices
