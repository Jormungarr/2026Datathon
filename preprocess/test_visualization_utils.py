import pandas as pd
from visualization_utils import plot_corner_heatmap
from data_utils import import_orginal_dataset

def test_plot_corner_heatmap():
    # Load the dataset
    df = import_orginal_dataset()
    # Select relevant columns
    columns = ['passengers', 'nsmiles', 'large_ms', 'TotalFaredPax_city1', ]
    df = df[columns].dropna()
    feature_names = ['Year', 'quarter', 'passengers', 'nsmiles', 'large_ms', 'TotalFaredPax_city1']
    hue = df["Year"]
    X = df[feature_names]
    plot_corner_heatmap(X, feature_names=feature_names, hue=hue, hue_labels=None, palette="Set2", figsize=(10, 10), bins=30, legend_title="city1")

if __name__ == "__main__":
    test_plot_corner_heatmap()
