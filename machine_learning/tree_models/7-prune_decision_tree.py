#!/usr/bin/env python3
"""Train and evaluate decision trees with cost-complexity pruning."""
from sklearn import tree

try:
    train_tree = __import__('1-train').train_tree
except ImportError:
    def train_tree(clf, X, y):
        clf.fit(X, y)


def prune_and_evaluate_trees(
    X_train,
    y_train,
    X_test,
    y_test,
    ccp_alphas,
    random_state,
    min_samples_leaf,
    min_samples_split,
):
    """Train multiple decision tree classifiers over pruning alphas.

    Args:
        X_train: Training feature data.
        y_train: Training labels.
        X_test: Testing feature data.
        y_test: Testing labels.
        ccp_alphas: Array of pruning alpha values to evaluate.
        random_state (int): Seed used by the random number generator.
        min_samples_leaf (int): Minimum number of samples required at a leaf.
        min_samples_split (int): Minimum number of samples required to split.

    Returns:
        tuple[list, list, list]: A tuple containing:
            - clfs: List of trained DecisionTreeClassifier instances.
            - train_scores: List of training accuracy scores for each tree.
            - test_scores: List of testing accuracy scores for each tree.
    """
    clfs = []
    train_scores = []
    test_scores = []

    for alpha in ccp_alphas:
        clf = tree.DecisionTreeClassifier(
            random_state=random_state,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            ccp_alpha=alpha,
        )
        train_tree(clf, X_train, y_train)
        clfs.append(clf)
        train_scores.append(clf.score(X_train, y_train))
        test_scores.append(clf.score(X_test, y_test))
    return clfs, train_scores, test_scores
