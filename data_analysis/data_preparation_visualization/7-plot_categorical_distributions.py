import matplotlib.pyplot as plt


def plot_categorical_distributions(df, columns_to_plot=None):
    """
    Generates a grid of bar charts displaying the value counts of columns
    in a DataFrame. The function automatically determines the grid size
    based on a predetermined number of columns, flattens the 2D axis layout
    in 1D layout for easier rendering with one single for loop, delete empty
    axes, subplots, rotate the x-axis labels to 45 degrees, and positions
    the column names at the top of each subplot.
    Args:
        df (pandas.DataFrame): the input DataFrame containing the data to plot.
        columns_to_plot (list of str, optional): A specific list of column
        to plot. If None, all columns with 'object' data type, 'Churn" excluded
        are selected.
    Returns:
        None. Displays the generated plot grid on screen and saves the figure
        locally as 'Task_7.png'.
    """
    if columns_to_plot is None:
        columns_to_plot = [
                col for col in df.columns
                if df[col].dtype == 'object' and col != 'Churn'
        ]
    else:
        if not isinstance(columns_to_plot, str):
            columns_to_plot = list(columns_to_plot)
    n_cols = 3
    n_rows = (len(columns_to_plot) + 2) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()
    for i, col in enumerate(columns_to_plot):
        df[col].value_counts().plot(kind='bar', ax=axes[i])
        axes[i].xaxis.set_label_position('top')
        axes[i].tick_params(axis='x', rotation=45)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.savefig("Task_7.png")
    plt.show()
