from preprocess.visualization_utils import plot_corner_heatmap
from preprocess.data_utils import import_orginal_dataset, import_unit_removed_dataset
import pandas as pd
if __name__ == "__main__":
    df = import_unit_removed_dataset()
    df["pax"] = 
    #merge year and quarter to single time index column
    df['time_index'] = df['Year'].astype(str) + ' Q' + df['quarter'].astype(str)
    #create dataframe that store slices of datatset based on time_index
    df_slices = {time_idx: df[df['time_index'] == time_idx] for time_idx in df['time_index'].unique()}
    feature_names = ['passengers', 'nsmiles', 'large_ms', 'TotalFaredPax_city1', 'fare', 'fare_lg']
    X = df[feature_names].dropna()
    print(len(X))

    plot_corner_heatmap(X, feature_names=['fare', 'fare_lg', 'passengers', 'nsmiles', 'large_ms', 'TotalFaredPax_city1'], hue=df['quarter'])
    