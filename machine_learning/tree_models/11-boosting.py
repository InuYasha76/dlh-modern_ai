#!/usr/bin/env python3
"""Compare boosting classifiers on the wine dataset."""
from sklearn import ensemble
import xgboost as xgb
import lightgbm as lgb


def compare_boosting_classifiers(name, n_estimators, random_state):
    """
    Initializes and returns an untrained boosting classifier based on
    the specified algorithm name.
    Args:
        name: Name of the boosting algorithm. Must be one of
            'adaboost', 'gradientboosting', 'xgboost', 'lightgbm'.
        n_estimators: Number of boosting iterations (trees).
        random_state: Random seed for reproducibility.
    Returns:
        An untrained instance of the selected boosting classifier.
    Raises:
        ValueError: If the provided model name is invalid.
    """
    if name == "adaboost":
        model = ensemble.AdaBoostClassifier(
            n_estimators=n_estimators, random_state=random_state
        )
    elif name == "gradientboosting":
        model = ensemble.GradientBoostingClassifier(
            n_estimators=n_estimators, random_state=random_state
        )
    elif name == "xgboost":
        model = xgb.XGBClassifier(
            n_estimators=n_estimators, random_state=random_state
        )
    elif name == "lightgbm":
        model = lgb.LGBMClassifier(
            n_estimators=n_estimators, random_state=random_state, verbose=-1
        )
    else:
        raise ValueError(f"Unknown model name '{name}'")

    return model
