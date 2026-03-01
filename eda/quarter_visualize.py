from preprocess.visualization_utils import plot_corner_heatmap
from preprocess.data_utils import import_orginal_dataset, import_unit_removed_dataset
import pandas as pd
import matplotlib.pyplot as plt
if __name__ == "__main__":
    df = import_unit_removed_dataset()
    df["city1_pax_strength"] = df.groupby(['Year', 'quarter', 'citymarketid_1'])['passengers'].transform('sum')
    df["city2_pax_strength"] = df.groupby(['Year', 'quarter', 'citymarketid_2'])['passengers'].transform('sum')
    
    #calculate relative pax_strength
    df["relative_pax_strength"] = abs(df["city1_pax_strength"] - df["city2_pax_strength"])
    df["total_pax_strength"] = df["city1_pax_strength"] + df["city2_pax_strength"]
    #merge year and quarter to single time index column
    df['time_index'] = df['Year'].astype(str) + ' Q' + df['quarter'].astype(str)
    feature_names = ['passengers', 'nsmiles', 'relative_pax_strength', 'total_pax_strength', 'large_ms', 'lf_ms', 'fare', 'fare_lg', 'fare_low']
    index_name = ['time_index']
    hue_names = ['carrier_lg']
    X = df[feature_names+hue_names + index_name].dropna()
    print(f"Dropped {len(df[feature_names+hue_names+index_name]) - len(X)} rows with missing values")   
    print(f"len(X): {len(X)}")
    #create dataframe that store slices of datatset based on time_index
    df_slices = {time_idx: X[X['time_index'] == time_idx].drop(columns='time_index') for time_idx in X['time_index'].unique()}
    



    for time_idx, df_slice in df_slices.items():
        print(f"Plotting for {time_idx}...")
        plot_corner_heatmap(
            df_slice,
            feature_names=feature_names,
            save_path=f"./results/corner_heatmap_{time_idx}.png",
            hue=df_slice[hue_names[0]]
        ) 



