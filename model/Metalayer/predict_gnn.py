# model/predict_gnn.py

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
import torch

from .config_gnn import (
    DATA_PATH, MAKE_UNDIRECTED, VAL_RATIO, CKPT_PATH, EDGE_NUM_COLS
)
from .graph_builder import load_and_build_graphs
from .gnn_model import FareGNN


@torch.no_grad()
def predict_edges(model, graphs) -> pd.DataFrame:
    model.eval()
    rows = []

    pax_idx = EDGE_NUM_COLS.index("passengers") if "passengers" in EDGE_NUM_COLS else None

    for g in graphs:
        pred = model(g).detach().cpu().numpy().reshape(-1)
        y = g.y_edge.detach().cpu().numpy().reshape(-1)

        mask = getattr(g, "undirected_mask", None)
        mask_np = np.ones_like(y, dtype=bool) if mask is None else mask.detach().cpu().numpy().astype(bool)

        # only output original edges (aligns with meta_u_city/meta_v_city)
        pred = pred[mask_np]
        y = y[mask_np]

        u_city = g.meta_u_city.detach().cpu().numpy().astype(int)
        v_city = g.meta_v_city.detach().cpu().numpy().astype(int)

        if pax_idx is not None:
            pax_std = g.edge_attr.detach().cpu().numpy()[mask_np, pax_idx]
        else:
            pax_std = np.full_like(pred, np.nan, dtype=float)

        for uu, vv, yt, yp, pw in zip(u_city, v_city, y, pred, pax_std):
            rows.append({
                "Year": int(g.year),
                "quarter": int(g.quarter),
                "citymarketid_1": int(uu),
                "citymarketid_2": int(vv),
                "y_true": float(yt),
                "y_pred": float(yp),
                "passengers_std": float(pw),
            })

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=DATA_PATH)
    ap.add_argument("--ckpt", type=str, default=CKPT_PATH)
    ap.add_argument("--out", type=str, default="val_predictions.csv")
    ap.add_argument("--scope", choices=["val", "train", "all"], default="val")
    args = ap.parse_args()

    df, graphs, train_graphs, val_graphs = load_and_build_graphs(
        args.data, make_undirected=MAKE_UNDIRECTED, val_ratio=VAL_RATIO
    )

    node_dim = int(graphs[0].x.size(1))
    edge_dim = int(graphs[0].edge_attr.size(1))

    model = FareGNN(node_dim=node_dim, edge_dim=edge_dim, hidden=64, n_layers=2)
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state)

    target = val_graphs if args.scope == "val" else train_graphs if args.scope == "train" else graphs
    out = predict_edges(model, target)
    out.to_csv(args.out, index=False)
    print(f"[ok] wrote {len(out)} rows -> {args.out}")


if __name__ == "__main__":
    main()