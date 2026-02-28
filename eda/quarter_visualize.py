from preprocess.visualization_utils import plot_corner_heatmap
from preprocess.data_utils import import_orginal_dataset
import pandas as pd
if __name__ == "__main__":
    df = import_orginal_dataset()
    df =  
    plot_corner_heatmap(df, feature_names=[, 'quarter', 'passengers', 'nsmiles', 'large_ms', 'TotalFaredPax_city1'], hue=df['quarter'])
    