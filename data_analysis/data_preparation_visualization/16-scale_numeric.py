#!/usr/bin/env python3
"""
Module to scale numeric features for modeling.
"""
from sklearn import preprocessing


def scale_numeric(df):
    """
    Standardize MonthlyCharges and TotalCharges using StandardScaler.
    Args:
        df (pandas.DataFrame): The input DataFrame.
    Returns:
        pandas.DataFrame: The DataFrame with MonthlyCharges and
            TotalCharges scaled to mean=0, std=1.
    """
    numeric_cols = ['MonthlyCharges', 'TotalCharges']
    scaler = preprocessing.StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df
