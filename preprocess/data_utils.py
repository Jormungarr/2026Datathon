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

def get_unique_cities_geocoding():
    df = pd.read_csv("./data/geocoded_cities.csv")
    return df

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def make_test_and_stratified_folds(
    feature_cols,
    import_fn,
    target_col="fare",
    test_ratio=0.1,
    n_splits=10,
    shuffle=True,
    random_state=42,
):
    """
    One-shot function.

    Steps:
      1) df = import_fn(); df = df.dropna()
      2) build strength features:
           city1_pax_strength, city2_pax_strength, rl_pax_str, tot_pax_str
         and time_index = "Year Qquarter"
      3) stratified sample test_ratio fraction as test set
      4) StratifiedKFold on remaining rows using time_index as strat label
      5) return (X_test, y_test, folds)

    Returns
    -------
    X_test : np.ndarray, shape (test_n, p)
    y_test : np.ndarray, shape (test_n,)
    folds  : list of dict, length = n_splits
        folds[i]["train"] = [X_train, y_train]
        folds[i]["val"]   = [X_val,   y_val]
    """
    # ---------- load & clean ----------
    df = import_fn()
    df = df.dropna().copy()

    # ---------- feature engineering ----------
    df["city1_pax_strength"] = (
        df.groupby(["Year", "quarter", "citymarketid_1"])["passengers"].transform("sum")
    )
    df["city2_pax_strength"] = (
        df.groupby(["Year", "quarter", "citymarketid_2"])["passengers"].transform("sum")
    )
    df["rl_pax_str"] = (df["city1_pax_strength"] - df["city2_pax_strength"]).abs()
    df["tot_pax_str"] = df["city1_pax_strength"] + df["city2_pax_strength"]
    df["time_index"] = df["Year"].astype(str) + " Q" + df["quarter"].astype(int).astype(str)

    # ---------- checks ----------
    needed = set(feature_cols + [target_col, "time_index"])
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in df: {missing}")

    n = len(df)
    if not (0 < test_ratio < 1):
        raise ValueError(f"test_ratio must be in (0, 1), got {test_ratio}")
    test_n = int(n * test_ratio)
    if test_n < 1 or test_n >= n:
        raise ValueError(f"test_ratio results in invalid test set size: {test_n}")

    # StratifiedKFold feasibility: each class count >= n_splits
    vc = df["time_index"].astype(str).value_counts()
    too_small = vc[vc < n_splits]
    if len(too_small) > 0:
        raise ValueError(
            f"Some strata have count < n_splits={n_splits}, cannot stratify.\n"
            f"Examples:\n{too_small.head(10).to_string()}"
        )

    # ---------- stratified sample test set ----------
    from sklearn.model_selection import StratifiedShuffleSplit
    strat_y = df["time_index"].astype(str).to_numpy()
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_state)
    test_idx, rest_idx = next(sss.split(df, strat_y))

    df_test = df.iloc[test_idx]
    df_rest = df.iloc[rest_idx]

    X_test = df_test[feature_cols].to_numpy()
    y_test = df_test[target_col].to_numpy()

    X_all = df_rest[feature_cols].to_numpy()
    y_all = df_rest[target_col].to_numpy()
    strat_y_rest = df_rest["time_index"].astype(str).to_numpy()

    # ---------- stratified k-fold ----------
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    folds = []
    for tr_idx, va_idx in skf.split(X_all, strat_y_rest):
        X_train, X_val = X_all[tr_idx], X_all[va_idx]
        y_train, y_val = y_all[tr_idx], y_all[va_idx]
        folds.append({
            "train": [X_train, y_train],
            "val":   [X_val,   y_val],
        })

    return X_test, y_test, folds, df_test, df_rest

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