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
    data = df.select_dtypes(include=["number"])
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm",
                vmin=-1, vmax=1)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()
