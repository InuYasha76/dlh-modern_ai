#!/usr/bin/env python3
"""
Module to handle column type conversions in a DataFrame.
"""
import pandas as pd


def convert_columns(df):
    """
    Performs type conversion for specific columns:
    - Converts TotalCharges to numeric, turning invalid strings into NaN.
    - Maps SeniorCitizen numeric values (0 and 1) to "No" and "Yes".
    Args:
        df (pandas.DataFrame): The DataFrame to modify.
    Returns:
        pandas.DataFrame: The modified DataFrame.
    """
    # astype(float) does not handle missing values and crashes with value = ' '
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
    return df
