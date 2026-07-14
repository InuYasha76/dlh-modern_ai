#!/usr/bin/env python3
"""Module to visualize correlations between continuous numeric features."""

import seaborn as sns
import matplotlib.pyplot as plt


def plot_correlation_heatmap(df):
    """
    Computes pairwise correlations for continuous numeric features
    and generates an annotated heatmap.
    Args:
        df (pandas.DataFrame): The input dataframe containinng the data.
    Rerturns:
        None
        Displays the correlation heatmap.
        The darker the hue, the highest the correlation.
    """
    plt.figure(figsize=(6, 5))
    data = df.select_dtypes(include="number")
    corr_matrix = data.corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.tight_layout()
    plt.show()
