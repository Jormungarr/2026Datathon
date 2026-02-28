from preprocess.data_utils import import_unit_removed_dataset
if __name__ == "__main__":
    df = import_unit_removed_dataset()
    #count total combined unique cities under variable city1 and city2
    #print all strings of all unique cities combined
    all_unique_cities = set(df['city1'].unique()).union(set(df['city2'].unique()))
    print(f"Number of unique cities: {len(all_unique_cities)}")
    for city in sorted(all_unique_cities):
        print(city)
    