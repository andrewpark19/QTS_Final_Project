import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_metric_heatmaps(df, metric):
    """
    Generates heatmaps for a given metric across different values of j and k, for each unique value of p.
    Arranges the subplots in a 2x3 grid, with low values in red and high values in green.
    Adds a main title in bold and makes the figure size smaller.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing 'j', 'k', 'p', and the metric column.
    metric (str): The metric to visualize (e.g., 'total_return', 'ann_sharpe', etc.).
    """
    unique_p_values = sorted(df['p'].unique())  # Get unique values of p
    num_p = len(unique_p_values)

    # Define subplot grid dimensions (2 rows, 3 columns)
    rows, cols = 2, 3
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(14, 9))  # Adjusted figure size to make it smaller

    # Flatten axes array for easy indexing
    axes = axes.flatten()

    for idx, p_value in enumerate(unique_p_values):
        df_filtered = df[df['p'] == p_value]
        
        # Pivot DataFrame for heatmap format
        heatmap_data = df_filtered.pivot(index='j', columns='k', values=metric)

        # Plot heatmap with custom colormap (Low values red, high values green)
        sns.heatmap(heatmap_data, annot=True, cmap="RdYlGn", fmt=".2f", linewidths=0.5, ax=axes[idx])

        # Titles and labels
        axes[idx].set_title(f"{metric} (p = {p_value})")
        axes[idx].set_xlabel("k values")
        axes[idx].set_ylabel("j values")

    # Remove extra empty subplots if num_p < 6
    for idx in range(num_p, rows * cols):
        fig.delaxes(axes[idx])

    # Add a main title for the entire figure in bold
    plt.suptitle(f"{metric} Heatmaps Across Different Values of j, k, and p", fontweight='bold', fontsize=16)

    # Adjust layout to fit everything
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust to fit the main title
    plt.show()

# Example usage:
# plot_metric_heatmaps(df, 'total_return')

import matplotlib.pyplot as plt

def plot_metric_distributions(datasets, titles,metric):
    """
    Plots histograms of 'downside_beta_BTC' for 6 different datasets in a 2x3 subplot layout.

    Parameters:
    datasets (list of pd.DataFrame): A list of 6 DataFrames containing the column 'downside_beta_BTC'.
    titles (list of str): A list of 6 titles corresponding to each dataset.
    """
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 7))  # 2 rows, 3 columns
    axes = axes.flatten()  # Flatten to easily index

    for i, (df, title) in enumerate(zip(datasets, titles)):
        axes[i].hist(df[metric], bins=30, color='blue', alpha=0.7, edgecolor='black')  # Histogram
        axes[i].set_title(title, fontweight='bold')  # Bold subplot title
        axes[i].set_xlabel(metric)  # X-axis label
        axes[i].set_ylabel('Frequency')  # Y-axis label

    # Main title for the entire figure
    plt.suptitle(f'Distribution of {metric} Across Strategies', fontweight='bold', fontsize=16)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Ensure spacing is adjusted properly
    plt.show()