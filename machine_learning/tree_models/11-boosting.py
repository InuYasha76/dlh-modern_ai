#!/usr/bin/env python3
"""Initialize boosting classifiers using various algorithms."""
from sklearn import ensemble


def compare_boosting_classifiers(name, n_estimators, random_state):
    """Initialize and return an untrained boosting classifier.

    Args:
        name (str): Name of the boosting algorithm. Must be one of
            'adaboost', 'gradientboosting', 'xgboost', 'lightgbm'.
        n_estimators (int): Number of boosting iterations (trees).
        random_state (int): Random seed for reproducibility.

    Returns:
        Classifier instance: An untrained instance of the selected boosting
            classifier.

    Raises:
        ValueError: If the provided model name is invalid or not a string.
    """
    if not isinstance(name, str):
        raise ValueError("Model name must be a string")

    name = name.strip().lower()

    if name == "adaboost":
        model = ensemble.AdaBoostClassifier(
            n_estimators=n_estimators, random_state=random_state
        )
    elif name == "gradientboosting":
        model = ensemble.GradientBoostingClassifier(
            n_estimators=n_estimators, random_state=random_state
        )
    elif name == "xgboost":
        try:
            import xgboost as xgb
        except ImportError as e:
            raise ImportError(
                "xgboost is required for the 'xgboost' classifier"
            ) from e
        model = xgb.XGBClassifier(
            n_estimators=n_estimators, random_state=random_state
        )
    elif name == "lightgbm":
        try:
            import lightgbm as lgb
        except ImportError as e:
            raise ImportError(
                "lightgbm is required for the 'lightgbm' classifier"
            ) from e
        model = lgb.LGBMClassifier(
            n_estimators=n_estimators, random_state=random_state, verbose=-1
        )
    else:
        raise ValueError(f"Unknown model name '{name}'")

    return model
