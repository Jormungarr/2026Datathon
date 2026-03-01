# model/report_test_gnn.py
from __future__ import annotations

import os
import math
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error

from .config_gnn import DATA_PATH, MAKE_UNDIRECTED, VAL_RATIO, CKPT_PATH
from .graph_builder import load_and_build_graphs
from .gnn_model import FareGNN


@torch.no_grad()
def predict_edges_df(model: torch.nn.Module, graphs) -> pd.DataFrame:
    model.eval()
    rows = []

    for g in graphs:
        pred = model(g).detach().cpu().numpy().reshape(-1)
        y = g.y_edge.detach().cpu().numpy().reshape(-1)

        mask = getattr(g, "undirected_mask", None)
        mask_np = np.ones_like(y, dtype=bool) if mask is None else mask.detach().cpu().numpy().astype(bool)

        pred = pred[mask_np]
        y = y[mask_np]

        u_city = g.meta_u_city.detach().cpu().numpy().astype(int)
        v_city = g.meta_v_city.detach().cpu().numpy().astype(int)

        for uu, vv, yt, yp in zip(u_city, v_city, y, pred):
            rows.append({
                "Year": int(g.year),
                "quarter": int(g.quarter),
                "citymarketid_1": int(uu),
                "citymarketid_2": int(vv),
                "y_true": float(yt),
                "y_pred": float(yp),
                "residual": float(yt - yp),
            })

    return pd.DataFrame(rows)


def main():
    # 1) load graphs exactly like training (data_utils_ext inside)
    df, graphs, train_graphs, test_graphs = load_and_build_graphs(
        DATA_PATH, make_undirected=MAKE_UNDIRECTED, val_ratio=VAL_RATIO
    )

    if len(graphs) == 0:
        raise RuntimeError("No graphs built. Check DATA_PATH and REQUIRED_COLS filters.")

    # 2) build model + load ckpt
    node_dim = int(graphs[0].x.size(1))
    edge_dim = int(graphs[0].edge_attr.size(1))

    model = FareGNN(node_dim=node_dim, edge_dim=edge_dim)
    state = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict(state)

    # 3) predict on test_graphs (time-holdout)
    out = predict_edges_df(model, test_graphs)

    # 4) metrics
    y_true = out["y_true"].to_numpy(dtype=float)
    y_pred = out["y_pred"].to_numpy(dtype=float)

    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"[TEST] rmse={rmse:.4f}  mae={mae:.4f}  n={len(out)}  graphs={len(test_graphs)}")

    # 5) save
    results_dir = os.path.join("results", "metalayer")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "test_predictions_gnn.csv")
    out.to_csv(out_path, index=False)
    print(f"[ok] wrote {len(out)} rows -> {out_path}")


if __name__ == "__main__":
    main()