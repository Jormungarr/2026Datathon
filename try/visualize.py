import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

IN_CSV = "outputs/lasso_oof_fitdata.csv"
OUT_DIR = "outputs/fold_plots"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(IN_CSV)

def plot_true_vs_pred(d, title, out_png):
    y_true = d["y_true"].to_numpy()
    y_pred = d["y_pred"].to_numpy()

    mse = np.mean((y_true - y_pred) ** 2)
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    lo = float(np.min([y_true.min(), y_pred.min()]))
    hi = float(np.max([y_true.max(), y_pred.max()]))

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=10, alpha=0.35)
    plt.plot([lo, hi], [lo, hi], linewidth=2)
    plt.xlabel("True fare")
    plt.ylabel("Predicted fare")
    plt.title(f"{title} | RMSE={rmse:.3f} MAE={mae:.3f}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

# 1) 每个 fold 一张图
for fold_id in sorted(df["fold_id"].unique()):
    d = df[df["fold_id"] == fold_id]
    out_png = os.path.join(OUT_DIR, f"fold_{int(fold_id):02d}_fit.png")
    plot_true_vs_pred(d, f"Fold {int(fold_id)}", out_png)
    print("Saved:", out_png, "| n=", len(d))

# 2) 总 OOF 一张图
plot_true_vs_pred(df, "OOF (all folds)", os.path.join("outputs", "oof_fit.png"))
print("Saved: outputs/oof_fit.png")