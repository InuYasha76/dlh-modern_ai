#!/usr/bin/env python3
"""
Module to visualize the churn target variable distribution.
"""
import matplotlib.pyplot as plt


def plot_churn_distribution(df):
    """
    Generates a bar plot showing the class distribution of the Churn column.
    Args:
        df (pandas.DataFrame): DataFrame with a 'Churn' column.
    Returns:
        None
    """
    plt.figure(figsize=(12, 8))
    df["Churn"].value_counts().plot.bar(color=["skyblue", "salmon"])
    plt.title("Churn Distribution")
    plt.ylabel("Count")
    plt.show()
