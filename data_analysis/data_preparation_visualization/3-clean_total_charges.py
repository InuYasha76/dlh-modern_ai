#!/usr/bin/env python3
"""
Module to clean missing values in the TotalCharges column.
"""


def clean_total_charges(df, method="drop"):
    """
    Handles missing values in the TotalCharges column using various strategies.
    Args:
        df (pandas.DataFrame): The DataFrame to modify.
        method (str): The strategy to use ('drop', 'median', 'impute').
    Returns:
        pandas.DataFrame: The modified DataFrame.
    """
    df_copy = df.copy()

    if method == "drop":
        return df_copy.dropna(subset=["TotalCharges"])

    elif method == "median":
        median = df_copy["TotalCharges"].median()
        df_copy.loc[:, "TotalCharges"] = df_copy["TotalCharges"].fillna(median)

    elif method == "impute":
        imput = df_copy["MonthlyCharges"] * df_copy["tenure"]
        df_copy.loc[:, "TotalCharges"] = df_copy["TotalCharges"].fillna(imput)

    else:
        raise ValueError(f"Unknown cleaning method: {method}")

    return df_copy
