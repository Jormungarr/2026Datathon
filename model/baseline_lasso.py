# baseline_lasso_no_fe.py
# LassoCV baseline: NO engineered variables (no pax_strength / rl_pax_str / tot_pax_str / time_index feature, etc.)
# Uses only columns already in the dataset as features.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import lasso_path

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ----------------------------
# Data loading (re-use your existing loader if you want)
# Option A: read Excel directly (recommended, no dependency on data_utils)
# ----------------------------
XLSX_PATH = os.path.join("data", "airline_ticket_dataset.xlsx")

def coerce_money_to_float(series: pd.Series) -> pd.Series:
    if series.dtype.kind in "if":
        return series.astype(float)
    s = series.astype(str)
    s = s.str.replace(r"[^\d\.\-]+", "", regex=True)
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    return pd.to_numeric(s, errors="coerce")

def load_df(path=XLSX_PATH) -> pd.DataFrame:
    df = pd.read_excel(path)
    for c in ["fare", "fare_lg", "fare_low"]:
        if c in df.columns:
            df[c] = coerce_money_to_float(df[c])
    return df


# ----------------------------
# Config
# ----------------------------
TARGET_COL = "fare"

# ONLY use variables that already exist in the dataset (no constructed strength vars)
BASE_FEATURE_COLS = [
    "passengers",
    "nsmiles",
    "large_ms",
    "lf_ms",
    "TotalFaredPax_city1",
    "TotalPerLFMkts_city1",
    "TotalPerPrem_city1",
    "TotalFaredPax_city2",
    "TotalPerLFMkts_city2",
    "TotalPerPrem_city2",
]

ENGINEERED_FEATURE_COLS = [
    "city1_pax_strength",
    "city2_pax_strength",
    "rl_pax_str",
    "tot_pax_str",
]

FEATURE_COLS = BASE_FEATURE_COLS + ENGINEERED_FEATURE_COLS

# Split settings
TEST_RATIO = 0.10
N_SPLITS = 10
SHUFFLE = True
RANDOM_STATE = 63

# Lasso grid
ALPHAS = np.logspace(-3, 1, 60)


def main():
    # ---------- load ----------
    df = load_df()
    # ---------- engineered features ----------
    df["city1_pax_strength"] = (
        df.groupby(["Year", "quarter", "citymarketid_1"])["passengers"].transform("sum")
    )

    df["city2_pax_strength"] = (
        df.groupby(["Year", "quarter", "citymarketid_2"])["passengers"].transform("sum")
    )

    df["rl_pax_str"] = (
        df["city1_pax_strength"] - df["city2_pax_strength"]
    ).abs()

    df["tot_pax_str"] = (
        df["city1_pax_strength"] + df["city2_pax_strength"]
    )

    # ---------- minimal cleaning ----------
    needed_cols = ["Year", "quarter"] + FEATURE_COLS + [TARGET_COL]
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # dropna only on required subset (do NOT drop rows because of unrelated cols)
    df = df.dropna(subset=needed_cols).copy()

    # ---------- strat label (NOT used as a feature; only for split) ----------
    # Using Year & quarter to stratify keeps quarter distribution stable.
    strat_y = (df["Year"].astype(int).astype(str) + "_Q" + df["quarter"].astype(int).astype(str)).to_numpy()

    # Check feasibility for StratifiedKFold on the training pool later
    vc = pd.Series(strat_y).value_counts()
    # For later KFold, each stratum should have >= N_SPLITS in the *rest* set.
    # We'll do a quick conservative check on full df first:
    if (vc < 2).any():
        # still OK for StratifiedShuffleSplit; just warning-like behavior
        # but to keep it strict we raise (you can relax if needed)
        rare = vc[vc < 2].head(10).to_string()
        raise ValueError(f"Some strata have < 2 samples; stratified split may be unstable.\n{rare}")

    # ---------- stratified test split ----------
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_RATIO, random_state=RANDOM_STATE)
    rest_idx, test_idx = next(sss.split(df, strat_y))

    df_test = df.iloc[test_idx].copy()
    df_rest = df.iloc[rest_idx].copy()

    X_test = df_test[FEATURE_COLS].to_numpy()
    y_test = df_test[TARGET_COL].to_numpy()

    X_rest = df_rest[FEATURE_COLS].to_numpy()
    y_rest = df_rest[TARGET_COL].to_numpy()
    strat_rest = (
        df_rest["Year"].astype(int).astype(str) + "_Q" + df_rest["quarter"].astype(int).astype(str)
    ).to_numpy()

    # feasibility for StratifiedKFold on rest
    vc_rest = pd.Series(strat_rest).value_counts()
    too_small = vc_rest[vc_rest < N_SPLITS]
    if len(too_small) > 0:
        raise ValueError(
            f"Some strata in rest set have count < n_splits={N_SPLITS}; cannot StratifiedKFold.\n"
            f"Examples:\n{too_small.head(10).to_string()}"
        )

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=SHUFFLE, random_state=RANDOM_STATE)

    # ---------- model ----------
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("lasso", LassoCV(
            alphas=ALPHAS,
            cv=cv.split(X_rest, strat_rest),   # explicit CV splits, stratified by Year/quarter
            max_iter=20000,
            n_jobs=None,                      # avoid loky multiprocessing issues
            random_state=RANDOM_STATE,
        ))
    ])

    pipe.fit(X_rest, y_rest)

    lasso = pipe.named_steps["lasso"]
    best_alpha = float(lasso.alpha_)
    # ---------- regularization path (coef vs alpha) ----------
    # Reuse the fitted preprocessor to ensure consistency with the trained model
    imputer = pipe.named_steps["imputer"]
    scaler = pipe.named_steps["scaler"]
    lasso = pipe.named_steps["lasso"]

    X_imp = imputer.transform(X_rest)      # (n, p)
    X_z = scaler.transform(X_imp)          # standardized features

    # Use the same alpha grid as LassoCV used (already sorted descending typically)
    alphas = lasso.alphas_
    # Compute coefficient path
    alphas_path, coefs_path, _ = lasso_path(X_z, y_rest, alphas=alphas)


    # Ensure results_dir is defined before saving the plot
    results_dir = os.path.join("results", "baseline_lasso")
    os.makedirs(results_dir, exist_ok=True)

    # Plot
    plt.figure(figsize=(10, 6))
    x = np.log10(alphas_path)

    for j, name in enumerate(FEATURE_COLS):
        plt.plot(x, coefs_path[j, :], linewidth=1, alpha=0.9)

    # Mark chosen alpha
    plt.axvline(np.log10(lasso.alpha_), linestyle="--", linewidth=2)
    plt.xlabel("log10(alpha)")
    plt.ylabel("Coefficient")
    plt.title(f"Lasso Regularization Path (best alpha = {lasso.alpha_:.4g})")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)

    # Optional: legend only for non-zero-at-best features (avoids unreadable legend)
    best_coef = lasso.coef_
    nz = np.where(np.abs(best_coef) > 1e-12)[0]
    if len(nz) > 0 and len(nz) <= 15:
        plt.legend([FEATURE_COLS[i] for i in nz], loc="best", fontsize=8)

    out_path = os.path.join(results_dir, "lasso_regularization_path.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")

    # ----- CV report (approx): use mse_path_ at best alpha index -----
    best_i = int(np.argmin(np.abs(lasso.alphas_ - lasso.alpha_)))
    fold_mse = lasso.mse_path_[best_i]  # shape: (n_folds,)
    mean_mse = float(fold_mse.mean())
    std_mse  = float(fold_mse.std(ddof=1)) if len(fold_mse) > 1 else 0.0
    mean_rmse = float(np.sqrt(mean_mse))

    print(f"[CV] alpha={best_alpha:.6g} | RMSE={mean_rmse:.6f} | MSE={mean_mse:.6f} ± {std_mse:.6f}")
    print(f"[CV] fold_mse={np.round(fold_mse, 6)}")

    # ---------- test eval ----------
    y_pred = pipe.predict(X_test)
    test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    test_mae  = float(mean_absolute_error(y_test, y_pred))
    print(f"[TEST] RMSE={test_rmse:.6f} | MAE={test_mae:.6f} (test_n={len(df_test)})")

    # ---------- save predictions ----------
    out = df_test[["Year", "quarter", "citymarketid_1", "citymarketid_2"]].copy()
    out["y_true"] = y_test
    out["y_pred"] = y_pred
    out["residual"] = out["y_true"] - out["y_pred"]
    out["stratum"] = (
        df_test["Year"].astype(int).astype(str) + "_Q" + df_test["quarter"].astype(int).astype(str)
    )

    out_path = os.path.join(results_dir, "lasso_no_engineered_features_test_predictions.csv")
    out.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()