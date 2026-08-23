#!/usr/bin/env python3
"""
Module to handle column type conversions in a DataFrame.
"""
import pandas as pd


def convert_columns(df):
    """
    Performs type conversion for specific columns:
        Converts TotalCharges to numeric, turning invalid strings into NaN.
        Maps SeniorCitizen numeric values (0 and 1) to "No" and "Yes".
    Args:
        df (pandas.DataFrame): The DataFrame to modify.
    Returns:
        pandas.DataFrame: The modified DataFrame.
    """
    if not isinstance(df, pd.DataFrame) or df.empty or len(df.columns) == 0:
        return 1
    if "TotalCharges" not in df.columns or "SeniorCitizen" not in df.columns:
        return 1
    # astype(float) does not handle missing values and crashes with value = ' '
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["SeniorCitizen"] = df["SeniorCitizen"].replace({
        0: "No",
        1: "Yes",
        "0": "No",
        "1": "Yes"
    })
    return df
