#!/usr/bin/env python3
"""
Module to clean missing values in the TotalCharges column.
"""


def clean_total_charges(df, method='drop'):
    """
    Handles missing values in the TotalCharges column using various strategies.
    Args:
        df (pandas.DataFrame): The DataFrame to modify.
        method (str): The strategy to use ('drop', 'median', 'impute').
    Returns:
        pandas.DataFrame: The modified DataFrame.
    """
    match method:
        case 'drop':
            df = df.dropna(subset=['TotalCharges'])
        case 'median':
            median_TC = df['TotalCharges'].median()
            df.loc[:, 'TotalCharges'] = df['TotalCharges'].fillna(median_TC)
        case 'impute':
            imputation = df['MonthlyCharges'] * df['tenure']
            df.loc[:, 'TotalCharges'] = df['TotalCharges'].fillna(imputation)
        case _:
            raise ValueError(f"Unknown cleaning method: {method}")
    return df
