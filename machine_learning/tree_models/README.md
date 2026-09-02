# Tree-Based Models & CART

This module provides implementations and utilities for building, training, visualizing, pruning, and evaluating tree-based machine learning models using Scikit-learn, XGBoost, and LightGBM.

## Overview of Modules

| Script | Function | Description |
| :--- | :--- | :--- |
| `0-build.py` | `build_decision_tree` | Builds and configures a Decision Tree classifier. |
| `1-train.py` | `train_tree` | Fits a tree-based classifier on training features and labels. |
| `2-draw.py` | `draw` | Displays a text-based decision flowchart/rules of a trained tree. |
| `3-generate_predictions.py` | `generate_predictions` | Generates class predictions using a trained classifier. |
| `4-evaluate.py` | `evaluate` | Generates a detailed classification report (precision, recall, F1). |
| `5-pre_pruning.py` | `prepruning` | Performs Grid Search over pre-pruning hyperparameters. |
| `6-pruning_path.py` | `get_pruning_path` | Retrieves cost-complexity pruning alphas and leaf impurities. |
| `7-prune_decision_tree.py` | `prune_and_evaluate_trees` | Trains and evaluates decision trees across a range of `ccp_alpha` values. |
| `8-best_ccp_alpha.py` | `get_best_alpha` | Selects optimal `ccp_alpha` based on test score, gap, and simplicity. |
| `9-random_forest.py` | `random_forest` | Creates a Scikit-learn Random Forest classifier. |
| `10-feature_importance.py` | `feature_importance` | Calculates feature importances and ascending sorted indices. |
| `11-boosting.py` | `compare_boosting_classifiers` | Instantiates AdaBoost, Gradient Boosting, XGBoost, or LightGBM models. |

## Setup

Create `.venv` from the repository root and install dependencies with:

```bash
/usr/bin/python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

