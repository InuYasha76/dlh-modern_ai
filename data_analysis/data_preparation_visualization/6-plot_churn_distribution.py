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
    churn_counts = df['Churn'].value_counts()
    colors = [
                'skyblue' if index == 'No'
                else 'salmon'
                for index in churn_counts.index
    ]
    plt.bar(churn_counts.index, churn_counts.values, color=colors)
    plt.title('Churn Distribution')
    plt.ylabel('Count')
    plt.show()
