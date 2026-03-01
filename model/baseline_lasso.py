# lasso_stratified_cv.py
import os
import numpy as np
import pandas as pd

import preprocess.data_utils as data_utils

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, mean_absolute_error

# -------------------------
# Config
# -------------------------
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 10
TEST_N = 100

# 与 data_utils.py 一致的 alpha 网格（你也可以改密一点）
ALPHAS = np.logspace(-3, 1, 60)

# -------------------------
# 1) Load & clean (align with data_utils.import_unit_removed_dataset)
# -------------------------
df = data_utils.import_unit_removed_dataset()
df = df.dropna().copy()

# -------------------------
# 2) Feature engineering (align with data_utils.make_test_and_stratified_folds)
# -------------------------
df["city1_pax_strength"] = (
    df.groupby(["Year", "quarter", "citymarketid_1"])["passengers"].transform("sum")
)
df["city2_pax_strength"] = (
    df.groupby(["Year", "quarter", "citymarketid_2"])["passengers"].transform("sum")
)
df["rl_pax_str"] = (df["city1_pax_strength"] - df["city2_pax_strength"]).abs()
df["tot_pax_str"] = df["city1_pax_strength"] + df["city2_pax_strength"]
df["time_index"] = df["Year"].astype(str) + " Q" + df["quarter"].astype(int).astype(str)

# -------------------------
# 3) Select features / target
# -------------------------
target_col = "fare"

# 你可以按需要增删：这里用 data_utils.py 里常用的“strength + 市占率 + 距离/客流”骨架
feature_cols = [
    "passengers",
    "nsmiles",
    "large_ms",
    "lf_ms",
    "rl_pax_str",
    "tot_pax_str",
    # 可选：加城市宏观变量（如果你认为有用）
    "TotalFaredPax_city1", "TotalPerLFMkts_city1", "TotalPerPrem_city1",
    "TotalFaredPax_city2", "TotalPerLFMkts_city2", "TotalPerPrem_city2",
]

needed = set(feature_cols + [target_col, "time_index"])
missing = [c for c in needed if c not in df.columns]
if missing:
    raise KeyError(f"Missing columns in df: {missing}")

# -------------------------
# 4) Sample test set (align with data_utils.make_test_and_stratified_folds)
# -------------------------
n = len(df)
if not (1 <= TEST_N < n):
    raise ValueError(f"TEST_N must be in [1, {n-1}], got {TEST_N}")

rng = np.random.default_rng(RANDOM_STATE)
all_idx = np.arange(n)
test_idx = rng.choice(all_idx, size=TEST_N, replace=False)

mask_test = np.zeros(n, dtype=bool)
mask_test[test_idx] = True

df_test = df.loc[mask_test].copy()
df_rest = df.loc[~mask_test].copy()

X_test = df_test[feature_cols].to_numpy()
y_test = df_test[target_col].to_numpy()
X_all  = df_rest[feature_cols].to_numpy()
y_all  = df_rest[target_col].to_numpy()
strat_y = df_rest["time_index"].astype(str).to_numpy()

# StratifiedKFold feasibility check (same as data_utils)
vc = pd.Series(strat_y).value_counts()
too_small = vc[vc < N_SPLITS]
if len(too_small) > 0:
    raise ValueError(
        f"Some strata have count < n_splits={N_SPLITS}, cannot stratify.\n"
        f"Examples:\n{too_small.head(10).to_string()}"
    )

# -------------------------
# 5) Build CV splits (same stratification rule)
# -------------------------
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
cv_splits = list(skf.split(X_all, strat_y))

# -------------------------
# 6) Pipeline + LassoCV (no joblib Parallel issues)
# -------------------------
pipe = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LassoCV(
        alphas=ALPHAS,
        cv=cv_splits,
        max_iter=20000,
        random_state=RANDOM_STATE,
        n_jobs=None,  # 显式禁用并行，避免你之前那类 loky 报错
    ))
])

pipe.fit(X_all, y_all)

m = pipe.named_steps["model"]
best_alpha = float(m.alpha_)

# report CV summary (same mse_path_ trick)
best_i = int(np.argmin(np.abs(m.alphas_ - m.alpha_)))
fold_mse = m.mse_path_[best_i]  # (n_folds,)
cv_mse_mean = float(np.mean(fold_mse))
cv_mse_std  = float(np.std(fold_mse, ddof=1)) if len(fold_mse) > 1 else 0.0
cv_rmse_mean = float(np.sqrt(cv_mse_mean))

print(f"[CV] RMSE={cv_rmse_mean:.6f}  MSE={cv_mse_mean:.6f} ± {cv_mse_std:.6f}  alpha={best_alpha:.6g}")
print("[CV] fold_mse:", ", ".join(f"{x:.6f}" for x in fold_mse))

# -------------------------
# 7) Evaluate on held-out test
# -------------------------
pred_test = pipe.predict(X_test)
rmse = float(np.sqrt(mean_squared_error(y_test, pred_test)))
mae  = float(mean_absolute_error(y_test, pred_test))
print(f"[TEST] RMSE={rmse:.6f}  MAE={mae:.6f}  (test_n={TEST_N})")

# -------------------------
# 8) Save predictions
# -------------------------
out = df_test[["Year", "quarter", "citymarketid_1", "citymarketid_2", "time_index"]].copy()
out["y_true"] = y_test
out["y_pred"] = pred_test
out["residual"] = out["y_true"] - out["y_pred"]
out_path = os.path.join(OUT_DIR, "lasso_time_stratified_test_predictions.csv")
out.to_csv(out_path, index=False)
print("Saved:", out_path)