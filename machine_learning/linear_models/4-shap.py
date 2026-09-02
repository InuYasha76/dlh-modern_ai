#!/usr/bin/env python3
"""SHAP Model Explainability Module."""


def get_shap_explainer_and_values(model, X_train, X_test):
    """Create a SHAP explainer and compute SHAP values for test data.

    Args:
        model: A trained regression or classification model.
        X_train: Input data used as background dataset.
        X_test: Input data to compute explanations for.

    Returns:
        tuple: A tuple containing:
            - explainer: Initialized shap.Explainer instance.
            - shap_values: Computed SHAP values for X_test predictions.

    Raises:
        ImportError: If the 'shap' package is not installed.
    """
    try:
        import shap
    except ImportError as e:
        raise ImportError(
            "The 'shap' library is required to compute SHAP values."
        ) from e

    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)
    return explainer, shap_values
