import os
import numpy as np
import pandas as pd
import preprocess.data_utils as data_utils

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, mean_absolute_error

# -------------------------
# Config
# -------------------------
VAL_YEAR = 2025
os.makedirs("outputs", exist_ok=True)

def quarter_id(year, q):
    return int(year) * 4 + int(q)

def canonical_route(a, b):
    a = int(a); b = int(b)
    return (a, b) if a <= b else (b, a)

def mode_or_unk(s: pd.Series) -> str:
    s = s.dropna()
    if len(s) == 0:
        return "UNK"
    m = s.mode()
    return str(m.iloc[0]) if len(m) else "UNK"

# ============ 1) 读数据 ============
df_raw = data_utils.import_orginal_dataset()

# quarter_id: 可排序季度索引
df_raw["quarter_id"] = df_raw.apply(lambda r: quarter_id(r["Year"], r["quarter"]), axis=1)

# ---- 用 canonical 无向 route（推荐：和 GNN/edge regression 对齐）
rt = df_raw.apply(lambda r: canonical_route(r["citymarketid_1"], r["citymarketid_2"]),
                  axis=1, result_type="expand")
df_raw["u_city"] = rt[0].astype(int)
df_raw["v_city"] = rt[1].astype(int)

# ============ 2) 聚合到 (route, quarter) ============
keys = ["quarter_id", "Year", "quarter", "u_city", "v_city"]
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
fare_lg_wavg = wavg("fare_lg").rename("fare_lg_wavg")
fare_low_wavg = wavg("fare_low").rename("fare_low_wavg")

# city macro (mean just for de-dup)
c1 = g[["TotalFaredPax_city1", "TotalPerLFMkts_city1", "TotalPerPrem_city1"]].mean()
c2 = g[["TotalFaredPax_city2", "TotalPerLFMkts_city2", "TotalPerPrem_city2"]].mean()

# carrier labels (mode)
carrier_lg = g["carrier_lg"].agg(mode_or_unk).rename("carrier_lg")
carrier_low = g["carrier_low"].agg(mode_or_unk).rename("carrier_low")

df = pd.concat(
    [pax_sum, nsmiles, fare_wavg, large_ms_wavg, lf_ms_wavg, fare_lg_wavg, fare_low_wavg,
     c1, c2, carrier_lg, carrier_low],
    axis=1
).reset_index()

df["log_pax"] = np.log1p(df["passengers_sum"].astype(float))

# ============ 3) 构造 t→t+1 预测任务 ============
df = df.sort_values(["u_city", "v_city", "quarter_id"]).reset_index(drop=True)

# label: 下一季度 fare（同一路线）
df["y_fare_t1"] = df.groupby(["u_city", "v_city"])["fare_wavg"].shift(-1)

# 一阶滞后（可选，但一般有益）
df["fare_lag1"] = df.groupby(["u_city", "v_city"])["fare_wavg"].shift(1)
df["log_pax_lag1"] = df.groupby(["u_city", "v_city"])["log_pax"].shift(1)
df["large_ms_lag1"] = df.groupby(["u_city", "v_city"])["large_ms_wavg"].shift(1)
df["lf_ms_lag1"] = df.groupby(["u_city", "v_city"])["lf_ms_wavg"].shift(1)

# 只保留有 label 的样本
df = df.dropna(subset=["y_fare_t1"]).copy()

# 年份（t）和目标年份（t+1）
df["year_t"] = (df["quarter_id"] // 4).astype(int)
df["year_t1"] = ((df["quarter_id"] + 1) // 4).astype(int)

# ============ 4) 2025 holdout 切分（与 edge regression 对齐） ============
# Train pairs: year_t < 2025 且 year_t1 < 2025 （不使用 2025 的 label）
train_df = df[(df["year_t"] < VAL_YEAR) & (df["year_t1"] < VAL_YEAR)].copy()

# Val pairs: 目标季度在 2025（t+1 属于 2025）：覆盖 2024Q4→2025Q1、2025Q1→Q2、Q2→Q3、Q3→Q4
val_df = df[df["year_t1"] == VAL_YEAR].copy()

print("Train pairs:", train_df.shape, "Val pairs:", val_df.shape)

# ============ 5) 特征列 ============
num_features = [
    "nsmiles", "log_pax",
    "large_ms_wavg", "lf_ms_wavg",
    "fare_wavg",          # t 时刻 fare（强 baseline）
    "fare_lg_wavg", "fare_low_wavg",
    "TotalFaredPax_city1", "TotalPerLFMkts_city1", "TotalPerPrem_city1",
    "TotalFaredPax_city2", "TotalPerLFMkts_city2", "TotalPerPrem_city2",
    "fare_lag1", "log_pax_lag1", "large_ms_lag1", "lf_ms_lag1",
]
cat_features = ["quarter", "carrier_lg", "carrier_low"]

X_train = train_df[num_features + cat_features]
y_train = train_df["y_fare_t1"].astype(float)

X_val = val_df[num_features + cat_features]
y_val = val_df["y_fare_t1"].astype(float)

# ============ 6) Pipeline + LassoCV（CV 只在训练集内部） ============
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
    alphas=np.logspace(-3, 1, 50),
    cv=5,
    max_iter=20000,
    random_state=42,
)

pipe = Pipeline([("pre", pre), ("model", model)])
pipe.fit(X_train, y_train)

pred = pipe.predict(X_val)

# sklearn 旧版兼容：sqrt(MSE)
mse = mean_squared_error(y_val, pred)
rmse = float(np.sqrt(mse))
mae = float(mean_absolute_error(y_val, pred))

print(f"[2025 holdout] RMSE={rmse:.4f}  MAE={mae:.4f}  alpha={pipe.named_steps['model'].alpha_:.6g}")

# ============ 7) 导出用于拟合图的数据 ============
fit_df = val_df[[
    "quarter_id", "Year", "quarter", "u_city", "v_city",
    "passengers_sum", "fare_wavg"
]].copy()

fit_df = fit_df.rename(columns={"fare_wavg": "fare_t"})
fit_df["fare_true_t1"] = y_val.to_numpy()
fit_df["fare_pred_t1"] = pred
fit_df["residual"] = fit_df["fare_true_t1"] - fit_df["fare_pred_t1"]
fit_df["abs_error"] = np.abs(fit_df["residual"])

out_path = "outputs/lasso_val_predictions_2025_holdout.csv"
fit_df.to_csv(out_path, index=False)
print("Saved:", out_path)