#!/usr/bin/env python3
"""Module to visualize categorical features against churn rates."""
import matplotlib.pyplot as plt


def plot_categorical_vs_churn(df, col):
    """
    Visualizes churn rates per category.
    Args:
        df (pandas.DataFrame): The input DataFrame containing the dataset.
        col (str): The target categorical column name.
    Returns:
        None
        The function displays the generated bar plot directly.
    """
    plt.figure(figsize=(12, 8))
    churn_rate = (df['Churn'] == 'Yes').groupby(df[col]).mean()
    plt.bar(churn_rate.index, churn_rate.values)
    plt.title(f"Churn Rate by {col}")
    plt.ylabel("Churn Rate")
    plt.xticks(rotation=45)
    plt.show()