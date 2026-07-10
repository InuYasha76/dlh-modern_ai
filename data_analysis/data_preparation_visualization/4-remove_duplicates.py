#!/usr/bin/env python3
"""
Module to handle removing duplicate rows from a DataFrame.
"""
import pandas as pd


def remove_duplicates(df):
    """
    Drops all duplicate rows from a pandas DataFrame.
    Args:
        df (pandas.DataFrame): The DataFrame to process.
    Returns:
        pandas.DataFrame: The deduplicated DataFrame.
    """
    return df.drop_duplicates()
