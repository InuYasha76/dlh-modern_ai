#!/usr/bin/env python3
"""
This module visualizes missing values in a pandas DataFrame.
"""
import matplotlib.pyplot as plt
import numpy as np


def plot_missingness(df):
    """
    Generates a scatter plot showing the location of missing values in df.
    Args:
        df (pandas.DataFrame): DataFrame to analyze for missing values.
    Returns:
        None
    """
    if type(df).__name__ != "DataFrame":
        return 1
    if df.empty or len(df.columns) == 0:
        return 1
    row_idx, col_idx = np.where(df.isna())
    plt.figure(figsize=(12, 8))
    plt.scatter(row_idx, col_idx, marker="|")
    plt.yticks(range(len(df.columns)), df.columns)
    plt.title("Missingness Plot")
    plt.tight_layout()
    plt.show()
    return 0
