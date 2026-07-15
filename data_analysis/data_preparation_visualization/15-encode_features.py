#!/usr/bin/env python3
"""Module to encode categorical features for machine learning models."""

import pandas as pd
from sklearn import preprocessing


def encode_features(df):
    """
    Encodes features for modeling using Scikit-learn.
    Args:
        df (pandas.DataFrame): The input DataFrame.
    Returns:
        tuple: (encoded DataFrame, fitted LabelEncoder,
        fitted binary OrdinalEncoder, fitted TenureGroup OrdinalEncoder)
    """
    df_encoded = df.copy()

    # Step 1: LabelEncoder for Churn
    churn_le = preprocessing.LabelEncoder()
    df_encoded["Churn"] = churn_le.fit_transform(df_encoded["Churn"])

    # Step 2: OrdinalEncoder for binary columns
    binary_cols = [
        "Partner",
        "Dependents",
        "PaperlessBilling",
        "SeniorCitizen"
    ]
    binary_oe = preprocessing.OrdinalEncoder(categories=[["No", "Yes"]] * 4)
    df_encoded[binary_cols] = binary_oe.fit_transform(
        df_encoded[binary_cols]
    ).astype("int64")

    # Step 3: One-hot encoding for Contract and PaymentMethod
    one_hot_cols = ["Contract", "PaymentMethod"]
    df_encoded = pd.get_dummies(
        df_encoded,
        columns=one_hot_cols,
        drop_first=True,
        dtype="int64"
    )

    # Step 4: OrdinalEncoder for TenureGroup
    categories = sorted(df_encoded["TenureGroup"].unique().tolist())
    tenure_oe = preprocessing.OrdinalEncoder(categories=[categories])
    df_encoded["TenureGroup"] = tenure_oe.fit_transform(
        df_encoded[["TenureGroup"]]
    ).astype("int64")

    return df_encoded, churn_le, binary_oe, tenure_oe
