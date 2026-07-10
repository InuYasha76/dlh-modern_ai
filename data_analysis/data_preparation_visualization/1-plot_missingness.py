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
        - df (pandas.DataFrame): DataFrame to analyze for missing values.
    Returns:
        - None
    """
    plt.figure(figsize=(12, 8))

    for i, col in enumerate(df.columns):
        rows = df.index[df[col].isna()]
        if len(rows):
            plt.scatter(rows, [i] * len(rows), marker='|', color="blue")

    plt.yticks(range(len(df.columns)), df.columns)
    plt.title('Missingness Plot')

    plt.tight_layout()
    plt.show()
