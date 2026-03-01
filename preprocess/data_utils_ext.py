# data_utils_ext.py
# Expanded utilities for Datathon2026 (Excel-first compatible)
#
# Goals:
# - Load xlsx/csv consistently (money parsing included)
# - Centralize leakage-safe feature engineering (pax strength)
# - Provide graph-level time splits (for quarter-graph training)
# - Keep dependencies light: pandas/numpy/sklearn only

from __future__ import annotations

import os
from typing import Dict, List, Sequence, Tuple, Union, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, KFold


# -------------------------
# Defaults
# -------------------------

DEFAULT_XLSX_PATH = os.path.join("data", "airline_ticket_dataset.xlsx")
DEFAULT_CSV_PATH  = os.path.join("data", "train.csv")


# -------------------------
# Money / unit parsing
# -------------------------

def coerce_money_to_float(series: pd.Series) -> pd.Series:
    """
    Convert money-like strings to float:
      "$1,234.56" -> 1234.56
      "  123 "    -> 123
    Non-parsable -> NaN
    """
    if series.dtype.kind in "if":
        return series.astype(float)

    s = series.astype(str)
    # Keep digits, dot, minus; drop $, commas, spaces, etc.
    s = s.str.replace(r"[^\d\.\-]+", "", regex=True)
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    return pd.to_numeric(s, errors="coerce")


# -------------------------
# Loading
# -------------------------

def import_excel_dataset(
    path: str = DEFAULT_XLSX_PATH,
    *,
    sheet_name: Union[str, int] = 0,
    money_cols: Sequence[str] = ("fare", "fare_lg", "fare_low"),
) -> pd.DataFrame:
    """Load an Excel dataset and coerce money columns."""
    df = pd.read_excel(path, sheet_name=sheet_name)
    for c in money_cols:
        if c in df.columns:
            df[c] = coerce_money_to_float(df[c])
    return df


def import_csv_dataset(
    path: str = DEFAULT_CSV_PATH,
    *,
    low_memory: bool = False,
    money_cols: Sequence[str] = ("fare", "fare_lg", "fare_low"),
) -> pd.DataFrame:
    """Load a CSV dataset and coerce money columns."""
    df = pd.read_csv(path, low_memory=low_memory)
    for c in money_cols:
        if c in df.columns:
            df[c] = coerce_money_to_float(df[c])
    return df


def load_dataset_auto(
    path: str,
    *,
    sheet_name: Union[str, int] = 0,
    money_cols: Sequence[str] = ("fare", "fare_lg", "fare_low"),
) -> pd.DataFrame:
    """Auto-detect xlsx/xls/csv."""
    lp = path.lower()
    if lp.endswith(".xlsx") or lp.endswith(".xls"):
        return import_excel_dataset(path, sheet_name=sheet_name, money_cols=money_cols)
    if lp.endswith(".csv"):
        return import_csv_dataset(path, money_cols=money_cols)
    raise ValueError(f"Unsupported file format: {path}")


# -------------------------
# Leakage control / feature utils
# -------------------------

def drop_leaky_cols(
    df: pd.DataFrame,
    *,
    leaky_cols: Sequence[str] = ("fare_lg", "fare_low"),
    errors: str = "ignore",
) -> pd.DataFrame:
    """Drop known leaky columns if present."""
    return df.drop(columns=list(leaky_cols), errors=errors).copy()


def add_pax_strength_features(
    df: pd.DataFrame,
    *,
    year_col: str = "Year",
    quarter_col: str = "quarter",
    city1_col: str = "citymarketid_1",
    city2_col: str = "citymarketid_2",
    pax_col: str = "passengers",
    out_city1: str = "city1_pax_strength",
    out_city2: str = "city2_pax_strength",
    out_rl: str = "rl_pax_str",
    out_tot: str = "tot_pax_str",
) -> pd.DataFrame:
    """
    Non-leaky strength features derived ONLY from passengers.
    Mirrors your layer.py logic but centralized.
    """
    df = df.copy()

    # Require the minimal columns; leave other NAs untouched.
    need = [year_col, quarter_col, city1_col, city2_col, pax_col]
    df = df.dropna(subset=need).copy()

    df[out_city1] = (
        df.groupby([year_col, quarter_col, city1_col])[pax_col].transform("sum")
    )
    df[out_city2] = (
        df.groupby([year_col, quarter_col, city2_col])[pax_col].transform("sum")
    )
    df[out_rl] = (df[out_city1] - df[out_city2]).abs()
    df[out_tot] = df[out_city1] + df[out_city2]
    return df


def basic_clean_for_graph_training(
    path: str,
    *,
    sheet_name: Union[str, int] = 0,
    money_cols: Sequence[str] = ("fare", "fare_lg", "fare_low"),
    leaky_cols: Sequence[str] = ("fare_lg", "fare_low"),
    dropna_subset: Sequence[str] = ("Year", "quarter", "citymarketid_1", "citymarketid_2", "fare", "passengers"),
) -> pd.DataFrame:
    """
    One-call loader for the MetaLayer quarter-graph pipeline:
    - load xlsx/csv
    - coerce money columns
    - drop leaky columns
    - add pax strength features
    - dropna on minimal required subset for graph construction + target
    """
    df = load_dataset_auto(path, sheet_name=sheet_name, money_cols=money_cols)
    df = drop_leaky_cols(df, leaky_cols=leaky_cols)
    df = add_pax_strength_features(df)
    df = df.dropna(subset=list(dropna_subset)).copy()
    return df


# -------------------------
# Graph-level splitting helpers
# -------------------------

def get_graph_keys_by_time(
    df: pd.DataFrame,
    *,
    year_col: str = "Year",
    quarter_col: str = "quarter",
) -> List[Tuple[int, int]]:
    """Return sorted unique (Year, quarter) keys."""
    keys = (
        df[[year_col, quarter_col]]
        .drop_duplicates()
        .sort_values([year_col, quarter_col])
    )
    return list(map(lambda t: (int(t[0]), int(t[1])), keys.to_numpy()))


def split_graph_keys_by_time_holdout(
    keys: Sequence[Tuple[int, int]],
    *,
    val_ratio: float = 0.2,
    min_val: int = 1,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Time holdout at graph-key level (last val_ratio fraction)."""
    keys = list(keys)
    n = len(keys)
    if n == 0:
        return [], []
    n_val = max(min_val, int(np.ceil(n * val_ratio)))
    if n > 1:
        n_val = min(n_val, n - 1)
    train_keys = keys[:-n_val] if n_val < n else []
    val_keys = keys[-n_val:]
    return train_keys, val_keys


def graph_stratified_kfold_split(
    keys: Sequence[Tuple[int, int]],
    *,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
) -> List[Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]]:
    """
    Stratified KFold over graph keys, strata = Year*10 + quarter.
    Falls back to plain KFold if some strata are too rare.
    """
    keys = list(keys)
    if len(keys) == 0:
        return []

    y = np.array([y * 10 + q for (y, q) in keys], dtype=int)
    uniq, cnt = np.unique(y, return_counts=True)

    if np.any(cnt < 2):
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        out = []
        for tr_idx, va_idx in kf.split(keys):
            out.append(([keys[i] for i in tr_idx], [keys[i] for i in va_idx]))
        return out

    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    out = []
    for tr_idx, va_idx in skf.split(keys, y):
        out.append(([keys[i] for i in tr_idx], [keys[i] for i in va_idx]))
    return out


def filter_df_by_graph_keys(
    df: pd.DataFrame,
    keys: Sequence[Tuple[int, int]],
    *,
    year_col: str = "Year",
    quarter_col: str = "quarter",
) -> pd.DataFrame:
    """Filter rows to those whose (Year, quarter) is in keys."""
    key_set = set((int(y), int(q)) for (y, q) in keys)
    mask = df.apply(lambda r: (int(r[year_col]), int(r[quarter_col])) in key_set, axis=1)
    return df.loc[mask].copy()


# -------------------------
# Sanity helpers
# -------------------------

def sanity_report(df: pd.DataFrame) -> Dict[str, Union[int, float]]:
    out: Dict[str, Union[int, float]] = {}
    out["n_rows"] = int(df.shape[0])
    out["n_cols"] = int(df.shape[1])

    for c in ["fare", "passengers", "city1_pax_strength", "tot_pax_str"]:
        if c in df.columns:
            out[f"{c}_nan_rate"] = float(df[c].isna().mean())

    if "Year" in df.columns and "quarter" in df.columns:
        out["n_graph_keys"] = int(df[["Year", "quarter"]].drop_duplicates().shape[0])
    return out
