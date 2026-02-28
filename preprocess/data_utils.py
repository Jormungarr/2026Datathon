import pandas as pd
def import_orginal_dataset():
    df = pd.read_excel("./data/airline_ticket_dataset.xlsx")
    return df
def import_unit_removed_dataset():
    df = import_orginal_dataset()
    df['fare'] = df['fare'].astype(str).str.replace('$', '', regex=False).astype(float)
    df['fare_lg'] = df['fare_lg'].astype(str).str.replace('$', '', regex=False).astype(float)
    return df

def get_unique_cities():
    df = import_unit_removed_dataset()
    all_unique_cities = set(df['city1'].unique()).union(set(df['city2'].unique()))
    return list(all_unique_cities)

COLUMN_UNIT_MAP = {
    "Year": "year",
    "quarter": "quarter",
    "citymarketid_1": "id",
    "citymarketid_2": "id",
    "city1": None,
    "city2": None,
    "nsmiles": "mile",
    "passengers": "passenger",
    "fare": "USD",
    "carrier_lg": None,
    "large_ms": "fraction",
    "fare_lg": "USD",
    "carrier_low": None,
    "lf_ms": "fraction",
    "fare_low": "USD",
    "TotalFaredPax_city1": "passenger",
    "TotalPerLFMkts_city1": "fraction",
    "TotalPerPrem_city1": "fraction",
    "TotalFaredPax_city2": "passenger",
    "TotalPerLFMkts_city2": "fraction",
    "TotalPerPrem_city2": "fraction"
}