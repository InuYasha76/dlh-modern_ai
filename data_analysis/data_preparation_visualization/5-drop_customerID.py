#!/usr/bin/env python3
"""
Module to remove non-predictive identifier columns from a DataFrame.
"""


def drop_customerID(df):
    """
    Drops the customerID column from a pandas DataFrame.
    Args:
        df (pandas.DataFrame): The DataFrame to process.
    Returns:
        pandas.DataFrame: The modified DataFrame without the customerID column.
    """
    return df.drop(columns=["customerID"])
