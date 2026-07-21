#!/usr/bin/env python3
"""Pre-Pruning."""
from sklearn import model_selection


def prepruning(X, y, clf):
    """
    Uses Scikit-learn to perform a Grid Search for the best pre-pruning
    hyperparameters for a decision tree classifier.
    Arguments:
        X: Input features
        y: Target labels
        clf: An untrained DecisionTreeClassifier instance
    Returns:
        A dictionary containing the best combination of hyperparameters found
        during the grid search.
    """
    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": list(range(2, 5)),
        "min_samples_leaf": list(range(2, 5)),
        "min_samples_split": list(range(2, 5)),
    }
    grid_search = model_selection.GridSearchCV(
            estimator=clf, param_grid=param_grid
    )
    grid_search.fit(X, y)
    return grid_search.best_params_
