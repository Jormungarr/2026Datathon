# model/Metalayer/predict_test.py

from __future__ import annotations

import math
import torch
import numpy as np
import pandas as pd

from .config_gnn import (
    MAKE_UNDIRECTED,
    CKPT_PATH,
    EDGE_NUM_COLS,   # 你 config 里那套 edge 特征列
)

from .graph_builder import (
    build_graphs_by_time,
    standardize_graphs,
)

from .gnn_model import FareGNN

# 关键：直接用你现有的 data_utils.py（不改它）
from preprocess.data_utils import (
    import_unit_removed_dataset,
    make_test_and_stratified_folds,
)


@torch.no_grad()
def predict_edges(model: torch.nn.Module, graphs) -> pd.DataFrame:
    """
    Only output ORIGINAL edges (undirected_mask == True) so meta_u_city/meta_v_city aligns.
    If y_edge exists (fare), output fare_true for evaluation only.
    """
    model.eval()
    rows = []

    for g in graphs:
        pred = model(g).detach().cpu().numpy().reshape(-1)

        mask = getattr(g, "undirected_mask", None)
        mask_np = np.ones_like(pred, dtype=bool) if mask is None else mask.detach().cpu().numpy().astype(bool)
        pred = pred[mask_np]

        if hasattr(g, "y_edge") and g.y_edge is not None:
            y = g.y_edge.detach().cpu().numpy().reshape(-1)
            y = y[mask_np]
        else:
            y = np.full_like(pred, np.nan, dtype=float)

        u_city = g.meta_u_city.detach().cpu().numpy().astype(int)
        v_city = g.meta_v_city.detach().cpu().numpy().astype(int)

        for uu, vv, yt, yp in zip(u_city, v_city, y, pred):
            rows.append({
                "Year": int(g.year),
                "quarter": int(g.quarter),
                "citymarketid_1": int(uu),
                "citymarketid_2": int(vv),
                "fare_true": float(yt) if np.isfinite(yt) else np.nan,  # 仅用于检验
                "fare_pred": float(yp),
            })

    return pd.DataFrame(rows)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(np.mean((y_true - y_pred) ** 2)))


def main():
    # ---------------------------
    # 1) 用 data_utils 生成 test/rest
    # ---------------------------
    # make_test_and_stratified_folds 需要 feature_cols，但我们最终只用它返回的 df_test/df_rest
    # 注意：该函数内部会 df.dropna()，并自动生成 strength 特征列
    feature_cols = [
        # 你模型 edge_attr 需要的列（其中 strength 会在函数内部生成）
        "nsmiles", "passengers", "large_ms", "lf_ms",
        "city1_pax_strength", "city2_pax_strength", "rl_pax_str", "tot_pax_str",
    ]

    X_test, y_test, folds, X_all, y_all, df_test, df_rest = make_test_and_stratified_folds(
        feature_cols=feature_cols,
        categorical_encode_cols=None,
        import_fn=import_unit_removed_dataset,  # 去掉 $，保证 fare 是 float
        target_col="fare",
        test_ratio=0.1,          # 和你当时的设定一致即可
        n_splits=10,
        shuffle=True,
        random_state=42,
    )

    # df_test / df_rest 已经包含 strength 特征（由 data_utils 生成）
    # 但可能还包含 fare_lg/fare_low 等列；我们不会用到它们。

    # ---------------------------
    # 2) 建图：rest 用来 fit 标准化器；test 用来预测
    # ---------------------------
    train_graphs = build_graphs_by_time(df_rest, make_undirected=MAKE_UNDIRECTED)
    test_graphs  = build_graphs_by_time(df_test, make_undirected=MAKE_UNDIRECTED)

    if len(train_graphs) == 0:
        raise RuntimeError("No train graphs built from df_rest. Check data / filters.")
    if len(test_graphs) == 0:
        raise RuntimeError("No test graphs built from df_test. Check test_ratio / min edges per quarter filter.")

    # 用 train_graphs 作为标准化拟合来源（避免 test 信息泄露）
    standardize_graphs(train_graphs, test_graphs)

    # ---------------------------
    # 3) 加载模型 checkpoint
    # ---------------------------
    node_dim = int(train_graphs[0].x.size(1))
    edge_dim = int(train_graphs[0].edge_attr.size(1))

    model = FareGNN(node_dim=node_dim, edge_dim=edge_dim)
    state = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict(state)

    # ---------------------------
    # 4) 预测 + 评估（fare 仅用于检验）
    # ---------------------------
    df_pred = predict_edges(model, test_graphs)

    # 如果 test 有 fare_true，则只用于评估
    if df_pred["fare_true"].notna().any():
        m = df_pred["fare_true"].notna().values
        y_true = df_pred.loc[m, "fare_true"].to_numpy(dtype=float)
        y_pred = df_pred.loc[m, "fare_pred"].to_numpy(dtype=float)
        rmse = _rmse(y_true, y_pred)
        mae = float(np.mean(np.abs(y_true - y_pred)))
        print(f"[test metrics] rmse={rmse:.4f}  mae={mae:.4f}  n={len(y_true)}")

    out_path = "test_predictions.csv"
    df_pred.to_csv(out_path, index=False)
    print(f"[ok] wrote {len(df_pred)} rows -> {out_path}")


if __name__ == "__main__":
    main()