# visual_fit_time_stratified.py
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ====== 输入：你可以按需改路径 ======
CANDIDATES = [
    ("lasso", "outputs/lasso_time_stratified_test_predictions.csv"),
    # 之后你把其它模型也导出成同样 schema（time_index,y_true,y_pred）就能一起画
    # ("gnn",   "outputs/gnn_time_stratified_test_predictions.csv"),
]

OUT_DIR = "outputs/time_stratified_fit_plots"
os.makedirs(OUT_DIR, exist_ok=True)


# -------------------------
# Schema normalization
# -------------------------
def normalize_schema(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    """
    统一成：
      time_index(str), Year(int), quarter(int),
      y_true(float), y_pred(float), pax_weight(optional)
    """
    d = df.copy()

    # --- time_index ---
    if "time_index" not in d.columns:
        # 尝试从 Year/quarter 拼
        if ("Year" in d.columns) and ("quarter" in d.columns):
            d["time_index"] = d["Year"].astype(int).astype(str) + " Q" + d["quarter"].astype(int).astype(str)
        else:
            raise ValueError(f"[{tag}] Missing time_index (or Year+quarter).")

    d["time_index"] = d["time_index"].astype(str)

    # --- Year / quarter ---
    if ("Year" not in d.columns) or ("quarter" not in d.columns):
        # 从 time_index 解析：YYYY Qk
        m = d["time_index"].str.extract(r"^\s*(\d{4})\s*Q\s*([1-4])\s*$")
        if m.isna().any().any():
            bad = d.loc[m.isna().any(axis=1), "time_index"].head(10).tolist()
            raise ValueError(f"[{tag}] time_index parse failed for examples: {bad}")
        d["Year"] = m[0].astype(int)
        d["quarter"] = m[1].astype(int)
    else:
        d["Year"] = d["Year"].astype(int)
        d["quarter"] = d["quarter"].astype(int)

    # --- y_true / y_pred ---
    # 兼容不同命名：y_true/y_pred 或 fare_true/fare_pred 或 fare_true_t1/fare_pred_t1
    true_candidates = ["y_true", "fare_true", "fare_true_t1"]
    pred_candidates = ["y_pred", "fare_pred", "fare_pred_t1"]

    tcol = next((c for c in true_candidates if c in d.columns), None)
    pcol = next((c for c in pred_candidates if c in d.columns), None)
    if tcol is None or pcol is None:
        raise ValueError(f"[{tag}] Missing true/pred columns. Need one of {true_candidates} and {pred_candidates}.")

    d["y_true"] = d[tcol].astype(float)
    d["y_pred"] = d[pcol].astype(float)

    # --- pax weight (optional) ---
    wcol = None
    for c in ["pax_weight", "passengers", "passengers_sum", "passengers_sum_t", "weight"]:
        if c in d.columns:
            wcol = c
            break
    d["pax_weight"] = d[wcol].astype(float) if wcol is not None else np.nan

    # --- keep finite only ---
    m = np.isfinite(d["y_true"]) & np.isfinite(d["y_pred"])
    d = d.loc[m].copy()

    # residual
    d["residual"] = d["y_true"] - d["y_pred"]

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


# -------------------------
# Plots
# -------------------------
def plot_true_vs_pred(d, title, out_png):
    y_true = d["y_true"].to_numpy()
    y_pred = d["y_pred"].to_numpy()
    if len(y_true) == 0:
        return

    rmse, mae = rmse_mae(y_true, y_pred)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=10, alpha=0.30)
    plt.plot([lo, hi], [lo, hi], linewidth=2)
    plt.xlabel("True fare")
    plt.ylabel("Predicted fare")
    plt.title(f"{title}\nRMSE={rmse:.3f}  MAE={mae:.3f}  n={len(y_true)}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def plot_residuals(d, title, out_png):
    r = d["residual"].to_numpy()
    if len(r) == 0:
        return
    rmse, mae = rmse_mae(d["y_true"].to_numpy(), d["y_pred"].to_numpy())

    plt.figure(figsize=(7, 4))
    plt.hist(r, bins=60)
    plt.xlabel("Residual (true - pred)")
    plt.ylabel("Count")
    plt.title(f"{title}\nRMSE={rmse:.3f}  MAE={mae:.3f}  n={len(r)}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def plot_residual_vs_pred(d, title, out_png):
    yp = d["y_pred"].to_numpy()
    r = d["residual"].to_numpy()
    if len(r) == 0:
        return

    plt.figure(figsize=(7, 4))
    plt.scatter(yp, r, s=10, alpha=0.25)
    plt.axhline(0.0, linewidth=2)
    plt.xlabel("Predicted fare")
    plt.ylabel("Residual (true - pred)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def plot_time_series(agg, title, out_png, xlabels):
    # agg columns: true, pred
    x = np.arange(len(agg))
    plt.figure(figsize=(10, 4))
    plt.plot(x, agg["true"].to_numpy(), marker="o")
    plt.plot(x, agg["pred"].to_numpy(), marker="o")
    plt.xticks(x, xlabels, rotation=45, ha="right")
    plt.xlabel("time_index")
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

    # 0) 总体指标
    rmse, mae = rmse_mae(d["y_true"].to_numpy(), d["y_pred"].to_numpy())

    # 1) 全体拟合图
    plot_true_vs_pred(d, f"{tag.upper()} - ALL", os.path.join(base, "fit_all.png"))
    print(f"[{tag}] saved:", os.path.join(base, "fit_all.png"))

    # 2) residual 分析
    plot_residuals(d, f"{tag.upper()} - Residual histogram", os.path.join(base, "residual_hist.png"))
    plot_residual_vs_pred(d, f"{tag.upper()} - Residual vs Pred", os.path.join(base, "residual_vs_pred.png"))
    print(f"[{tag}] saved:", os.path.join(base, "residual_hist.png"))
    print(f"[{tag}] saved:", os.path.join(base, "residual_vs_pred.png"))

    # 3) 按 time_index 分散点图（每个季度单独一张）
    for (yy, qq), sub in d.groupby(["Year", "quarter"], sort=True):
        out_png = os.path.join(base, f"fit_{yy}Q{qq}.png")
        plot_true_vs_pred(sub, f"{tag.upper()} - {yy}Q{qq}", out_png)
    print(f"[{tag}] per-quarter fit plots saved in:", base)

    # 4) time_index 的均值曲线（mean）
    agg_mean = d.groupby(["Year", "quarter"], sort=True).agg(
        true=("y_true", "mean"),
        pred=("y_pred", "mean"),
        n=("y_true", "size"),
    ).reset_index()

    xlabels = [f"{y}Q{q}" for y, q in zip(agg_mean["Year"], agg_mean["quarter"])]
    plot_time_series(
        agg_mean,
        f"{tag.upper()} - Quarter MEAN fare: True vs Pred",
        os.path.join(base, "ts_quarter_mean.png"),
        xlabels,
    )
    print(f"[{tag}] saved:", os.path.join(base, "ts_quarter_mean.png"))

    # 5) pax-weighted 曲线（如果有 pax_weight）
    if np.isfinite(d["pax_weight"]).any():
        rows = []
        for (yy, qq), sub in d.groupby(["Year", "quarter"], sort=True):
            rows.append({
                "Year": yy,
                "quarter": qq,
                "true": wavg(sub["y_true"].to_numpy(), sub["pax_weight"].to_numpy()),
                "pred": wavg(sub["y_pred"].to_numpy(), sub["pax_weight"].to_numpy()),
                "n": len(sub),
            })
        agg_w = pd.DataFrame(rows).sort_values(["Year", "quarter"]).reset_index(drop=True)
        xlabels_w = [f"{y}Q{q}" for y, q in zip(agg_w["Year"], agg_w["quarter"])]

        plot_time_series(
            agg_w,
            f"{tag.upper()} - Quarter PAX-WEIGHTED fare: True vs Pred",
            os.path.join(base, "ts_quarter_pax_weighted.png"),
            xlabels_w,
        )
        print(f"[{tag}] saved:", os.path.join(base, "ts_quarter_pax_weighted.png"))
    else:
        print(f"[{tag}] no pax weight column -> skip pax-weighted timeseries")

    # 6) per-quarter metrics (no groupby.apply -> no FutureWarning)
    g = d.groupby(["Year", "quarter"], sort=True)

    perq = g.agg(
    n=("y_true", "size"),
    mse=("residual", lambda x: float(np.mean(np.square(x.to_numpy())))),
    mae=("residual", lambda x: float(np.mean(np.abs(x.to_numpy())))),
    bias_mean_residual=("residual", "mean"),
).reset_index()

    perq["rmse"] = np.sqrt(perq["mse"])
    perq = perq.drop(columns=["mse"])

    perq.to_csv(os.path.join(base, "per_quarter_metrics.csv"), index=False)
    print(f"[{tag}] saved:", os.path.join(base, "per_quarter_metrics.csv"))

    # 7) summary
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