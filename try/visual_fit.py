import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ====== 输入：你可以按需改路径 ======
CANDIDATES = [
    ("lasso", "outputs/lasso_val_predictions_2025_holdout.csv"),
    ("gnn",   "outputs/gnn_edge_val_predictions_2025_holdout.csv"),
]

OUT_DIR = "outputs/holdout_fit_plots"
os.makedirs(OUT_DIR, exist_ok=True)


# --------- 统一列名适配 ---------
def normalize_schema(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    """
    返回统一 schema：
      quarter_id_t, fare_true_t1, fare_pred_t1, passengers_weight(optional)
    """
    d = df.copy()

    # quarter_id_t
    if "quarter_id" in d.columns:
        d["quarter_id_t"] = d["quarter_id"].astype(int)
    elif "quarter_id_t" in d.columns:
        d["quarter_id_t"] = d["quarter_id_t"].astype(int)
    else:
        raise ValueError(f"[{tag}] Missing quarter id column: expected quarter_id or quarter_id_t")

    # y_true / y_pred
    if "fare_true_t1" not in d.columns:
        # gnn 导出里是 fare_true_t1；lasso 我们也用了 fare_true_t1
        # 如果你有别的命名，在这里补映射即可
        raise ValueError(f"[{tag}] Missing fare_true_t1")
    if "fare_pred_t1" not in d.columns:
        raise ValueError(f"[{tag}] Missing fare_pred_t1")

    d["fare_true_t1"] = d["fare_true_t1"].astype(float)
    d["fare_pred_t1"] = d["fare_pred_t1"].astype(float)

    # passengers 权重（可选）
    # lasso 文件里是 passengers_sum；gnn 默认没带 pax（除非你改过导出）
    wcol = None
    for c in ["passengers_sum_t", "passengers_sum", "pax_t", "weight"]:
        if c in d.columns:
            wcol = c
            break
    if wcol is not None:
        d["pax_weight"] = d[wcol].astype(float)
    else:
        d["pax_weight"] = np.nan

    # 目标季度（t+1）
    d["quarter_id_t1"] = d["quarter_id_t"] + 1
    d["year_t1"] = (d["quarter_id_t1"] // 4).astype(int)
    d["q_t1"] = (d["quarter_id_t1"] % 4).astype(int)
    d.loc[d["q_t1"] == 0, "q_t1"] = 4

    # 仅保留 finite
    m = np.isfinite(d["fare_true_t1"]) & np.isfinite(d["fare_pred_t1"])
    d = d[m].copy()

    return d


def rmse_mae(y_true, y_pred):
    e = y_true - y_pred
    rmse = float(np.sqrt(np.mean(e * e)))
    mae = float(np.mean(np.abs(e)))
    return rmse, mae


def wavg(x, w):
    w = np.asarray(w, dtype=float)
    x = np.asarray(x, dtype=float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if m.sum() == 0:
        return np.nan
    return float(np.sum(x[m] * w[m]) / np.sum(w[m]))


def plot_true_vs_pred(d, title, out_png):
    y_true = d["fare_true_t1"].to_numpy()
    y_pred = d["fare_pred_t1"].to_numpy()

    if len(y_true) == 0:
        return

    rmse, mae = rmse_mae(y_true, y_pred)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=10, alpha=0.30)
    plt.plot([lo, hi], [lo, hi], linewidth=2)
    plt.xlabel("True fare (t+1)")
    plt.ylabel("Predicted fare (t+1)")
    plt.title(f"{title}\nRMSE={rmse:.3f}  MAE={mae:.3f}  n={len(y_true)}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def plot_quarter_timeseries(agg, title, out_png):
    # agg columns: year_t1, q_t1, true, pred
    agg = agg.sort_values(["year_t1", "q_t1"]).reset_index(drop=True)
    x = np.arange(len(agg))

    plt.figure(figsize=(8, 4))
    plt.plot(x, agg["true"].to_numpy(), marker="o")
    plt.plot(x, agg["pred"].to_numpy(), marker="o")
    plt.xticks(x, [f"{y}Q{q}" for y, q in zip(agg["year_t1"], agg["q_t1"])])
    plt.xlabel("Target quarter (t+1)")
    plt.ylabel("Fare")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def process_one(tag, path):
    if not os.path.exists(path):
        print(f"[skip] {tag}: not found -> {path}")
        return

    df = pd.read_csv(path)
    d = normalize_schema(df, tag)

    base = os.path.join(OUT_DIR, tag)
    os.makedirs(base, exist_ok=True)

    # 1) 全体拟合图
    plot_true_vs_pred(d, f"{tag.upper()} - ALL (2025 holdout)", os.path.join(base, "fit_all.png"))
    print(f"[{tag}] saved:", os.path.join(base, "fit_all.png"))

    # 2) 按目标季度（t+1）分图
    for (yy, qq), sub in d.groupby(["year_t1", "q_t1"], sort=True):
        out_png = os.path.join(base, f"fit_{yy}Q{qq}.png")
        plot_true_vs_pred(sub, f"{tag.upper()} - Target {yy}Q{qq}", out_png)
    print(f"[{tag}] per-quarter plots saved in:", base)

    # 3) 季度均值折线（mean）
    agg_mean = d.groupby(["year_t1", "q_t1"], sort=True).agg(
        true=("fare_true_t1", "mean"),
        pred=("fare_pred_t1", "mean"),
        n=("fare_true_t1", "size"),
    ).reset_index()
    plot_quarter_timeseries(
        agg_mean, f"{tag.upper()} - Quarter MEAN fare (t+1): True vs Pred",
        os.path.join(base, "ts_quarter_mean.png")
    )
    print(f"[{tag}] saved:", os.path.join(base, "ts_quarter_mean.png"))

    # 4) 季度 pax 加权折线（如果有 pax_weight）
    if np.isfinite(d["pax_weight"]).any():
        rows = []
        for (yy, qq), sub in d.groupby(["year_t1", "q_t1"], sort=True):
            rows.append({
                "year_t1": yy,
                "q_t1": qq,
                "true": wavg(sub["fare_true_t1"].to_numpy(), sub["pax_weight"].to_numpy()),
                "pred": wavg(sub["fare_pred_t1"].to_numpy(), sub["pax_weight"].to_numpy()),
                "n": len(sub),
            })
        agg_w = pd.DataFrame(rows)
        plot_quarter_timeseries(
            agg_w, f"{tag.upper()} - Quarter PAX-WEIGHTED fare (t+1): True vs Pred",
            os.path.join(base, "ts_quarter_pax_weighted.png")
        )
        print(f"[{tag}] saved:", os.path.join(base, "ts_quarter_pax_weighted.png"))
    else:
        print(f"[{tag}] no pax weight column -> skip pax-weighted timeseries")

    # 5) 保存一个 summary
    rmse, mae = rmse_mae(d["fare_true_t1"].to_numpy(), d["fare_pred_t1"].to_numpy())
    summary = pd.DataFrame([{
        "model": tag,
        "n": len(d),
        "rmse": rmse,
        "mae": mae,
        "source_csv": path
    }])
    summary.to_csv(os.path.join(base, "summary.csv"), index=False)
    print(f"[{tag}] summary:", summary.to_dict(orient="records")[0])


def main():
    for tag, path in CANDIDATES:
        process_one(tag, path)

    print("\nAll outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()