from preprocess.visualization_utils import plot_corner_heatmap
from preprocess.data_utils import import_orginal_dataset, import_unit_removed_dataset
import pandas as pd
if __name__ == "__main__":
    df = import_unit_removed_dataset()
    df = df[df["Year"] == 2022]
    feature_names = ['Year', 'quarter', 'passengers', 'nsmiles', 'large_ms', 'TotalFaredPax_city1', 'fare', 'fare_lg']
    X = df[feature_names].dropna()
    print(len(X))

    plot_corner_heatmap(X, feature_names=['fare', 'fare_lg', 'passengers', 'nsmiles', 'large_ms', 'TotalFaredPax_city1'], hue=df['quarter'])
    