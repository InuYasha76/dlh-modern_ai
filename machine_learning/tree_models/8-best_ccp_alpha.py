#!/usr/bin/env python3
"""Select the best ccp_alpha for pruning a decision tree."""


def get_best_alpha(clfs, train_scores, test_scores, ccp_alphas):
    """Select the best ccp_alpha based on the highest test accuracy,
    best generalization gap (smallest difference between training
    and test accuracy) and simplicity (largest alpha as tiebreaker).
    Args:
        clfs: List of trained DecisionTreeClassifier instances.
        train_scores: List of training accuracy scores.
        test_scores: List of test accuracy scores.
        ccp_alphas: List or array of ccp_alpha values used to train clfs.
    Returns:
        best_alpha: The selected ccp_alpha value.
        best_clf: The classifier associated with best_alpha.
    """
    max_test_score = max(test_scores)
    candidates = [i for i, score in enumerate(test_scores)
                  if score == max_test_score
                  ]
    min_gap = min(abs(train_scores[i] - test_scores[i]) for i in candidates)
    candidates = [i for i in candidates
                  if abs(train_scores[i] - test_scores[i]) == min_gap
                  ]
    best_index = max(candidates, key=lambda i: ccp_alphas[i])
    best_alpha = ccp_alphas[best_index]
    best_clf = clfs[best_index]
    return best_alpha, best_clf
