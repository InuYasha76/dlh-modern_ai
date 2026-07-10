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
    plt.scatter(np.where(df.isna())[0], np.where(df.isna())[1], marker='|')
    plt.title('Missingness Plot')
    plt.tight_layout()
    plt.show()
