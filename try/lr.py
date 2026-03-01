import preprocess.data_utils as data_utils
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ============ 1) 读数据 ============
df_raw = data_utils.import_orginal_dataset()


# quarter_id: 可排序季度索引（16个季度）
df_raw["quarter_id"] = df_raw["Year"].astype(int) * 4 + df_raw["quarter"].astype(int)

# ============ 2) 聚合到 (route, quarter) ============
keys = ["quarter_id", "Year", "quarter", "citymarketid_1", "citymarketid_2"]
g = df_raw.groupby(keys, sort=False)

# passengers sum
pax_sum = g["passengers"].sum(min_count=1).rename("passengers_sum")

# mean distance
nsmiles = g["nsmiles"].mean().rename("nsmiles")

# weighted average helper (vectorized)
def wavg(col):
    num = (df_raw[col] * df_raw["passengers"]).groupby([df_raw[k] for k in keys]).sum(min_count=1)
    den = df_raw["passengers"].groupby([df_raw[k] for k in keys]).sum(min_count=1)
    return (num / den)

fare_wavg = wavg("fare").rename("fare_wavg")
large_ms_wavg = wavg("large_ms").rename("large_ms_wavg")
lf_ms_wavg = wavg("lf_ms").rename("lf_ms_wavg")

# city macro (mean just for de-dup; if already constant it's fine)
c1 = g[["TotalFaredPax_city1","TotalPerLFMkts_city1","TotalPerPrem_city1"]].mean()
c2 = g[["TotalFaredPax_city2","TotalPerLFMkts_city2","TotalPerPrem_city2"]].mean()

# carrier labels (mode)
carrier_lg = g["carrier_lg"].agg(lambda s: s.mode().iloc[0] if s.notna().any() else "UNK").rename("carrier_lg")
carrier_low = g["carrier_low"].agg(lambda s: s.mode().iloc[0] if s.notna().any() else "UNK").rename("carrier_low")

df = pd.concat([pax_sum, nsmiles, fare_wavg, large_ms_wavg, lf_ms_wavg, c1, c2, carrier_lg, carrier_low], axis=1).reset_index()
df["log_pax"] = np.log1p(df["passengers_sum"])

# ============ 3) 可选：加一阶 lag（按航线） ============
df = df.sort_values(["citymarketid_1","citymarketid_2","quarter_id"]).reset_index(drop=True)
route_keys = ["citymarketid_1","citymarketid_2"]
for col in ["fare_wavg", "log_pax", "large_ms_wavg", "lf_ms_wavg"]:
    df[f"{col}_lag1"] = df.groupby(route_keys)[col].shift(1)

# 目标：当前季度 fare（如果你想预测“下一季度”，把 y 改成 shift(-1) 即可）
y = df["fare_wavg"].astype(float)

# ============ 4) 特征列 ============
num_features = [
    "nsmiles", "log_pax",
    "large_ms_wavg", "lf_ms_wavg",
    "TotalFaredPax_city1","TotalPerLFMkts_city1","TotalPerPrem_city1",
    "TotalFaredPax_city2","TotalPerLFMkts_city2","TotalPerPrem_city2",
    # lag features（没有历史的会 NaN，下面会 impute）
    "fare_wavg_lag1","log_pax_lag1","large_ms_wavg_lag1","lf_ms_wavg_lag1",
]
cat_features = ["quarter", "carrier_lg", "carrier_low"]

X = df[num_features + cat_features]
groups = df["quarter_id"].to_numpy()

# ============ 5) 以“季度”为单位做 KFold ============
# 先在 16 个季度 id 上做 KFold，再映射回样本索引
unique_q = np.sort(np.unique(groups))
K = 5  # 你想要几折就改这里：4/5/8/16 都行（16≈LOQO）
kf = KFold(n_splits=K, shuffle=True, random_state=42)

splits = []
for tr_q_idx, va_q_idx in kf.split(unique_q):
    tr_q = set(unique_q[tr_q_idx])
    va_q = set(unique_q[va_q_idx])
    tr_idx = np.flatnonzero(np.isin(groups, list(tr_q)))
    va_idx = np.flatnonzero(np.isin(groups, list(va_q)))
    splits.append((tr_idx, va_idx))

# ============ 6) LassoCV + 预处理管道 ============
pre = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), num_features),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
        ]), cat_features),
    ]
)

model = LassoCV(
    alphas=np.logspace(-3, 1, 40),
    cv=splits,
    max_iter=20000,
    random_state=42,
)

pipe = Pipeline([("pre", pre), ("model", model)])

# ============ 7) 训练 + CV 结果 ============
pipe.fit(X, y)

# 用同样 splits 评估 out-of-fold（简单做法：逐 fold 预测）
oof = np.full(len(df), np.nan, dtype=float)

fold_id_arr = np.full(len(df), -1, dtype=int)

for fold_id, (tr_idx, va_idx) in enumerate(splits, start=1):
    model = LassoCV(alphas=np.logspace(-3, 1, 40), cv=5, max_iter=20000, random_state=42)
    pipe = Pipeline([("pre", pre), ("model", model)])

    pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    oof[va_idx] = pipe.predict(X.iloc[va_idx])
    fold_id_arr[va_idx] = fold_id

mse = mean_squared_error(y[~np.isnan(oof)], oof[~np.isnan(oof)])
rmse = np.sqrt(mse)
mae = mean_absolute_error(y[~np.isnan(oof)], oof[~np.isnan(oof)])

print(f"K={K} quarter-fold Lasso | OOF RMSE={rmse:.4f}  MAE={mae:.4f}")
print(f"Chosen alpha={pipe.named_steps['model'].alpha_}")

# ====== 保存拟合数据（OOF）======
mask = ~np.isnan(oof)

fit_df = df.loc[mask, ["Year", "quarter", "quarter_id", "citymarketid_1", "citymarketid_2"]].copy()
fit_df["fold_id"] = fold_id_arr[mask]
fit_df["y_true"] = y.loc[mask].to_numpy()
fit_df["y_pred"] = oof[mask]

# 可选：误差列
fit_df["residual"] = fit_df["y_true"] - fit_df["y_pred"]
fit_df["abs_error"] = np.abs(fit_df["residual"])

out_path = "outputs/lasso_oof_fitdata.csv"
import os
os.makedirs("outputs", exist_ok=True)
fit_df.to_csv(out_path, index=False)
print("Saved fit data:", out_path)