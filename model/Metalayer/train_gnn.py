# model/train_gnn.py

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error

from .config_gnn import (
    DATA_PATH, MAKE_UNDIRECTED, VAL_RATIO, SEED,
    HIDDEN, N_LAYERS, LR, WEIGHT_DECAY, EPOCHS, CKPT_PATH,
    EDGE_NUM_COLS
)
from .graph_builder import seed_everything, load_and_build_graphs
from .gnn_model import FareGNN


def _apply_eval_mask(g, pred):
    y = g.y_edge
    mask = getattr(g, "undirected_mask", None)
    if mask is None:
        return pred, y, g.edge_attr
    return pred[mask], y[mask], g.edge_attr[mask]


def train_one_epoch(model, graphs, opt, *, pax_weight: bool = False, pax_col: str = "passengers") -> float:
    model.train()
    losses = []
    pax_idx = EDGE_NUM_COLS.index(pax_col) if (pax_weight and pax_col in EDGE_NUM_COLS) else None

    for g in graphs:
        opt.zero_grad()
        pred = model(g)
        pred_m, y_m, edge_attr_m = _apply_eval_mask(g, pred)

        if pax_idx is None:
            loss = F.mse_loss(pred_m, y_m)
        else:
            w = torch.log1p(edge_attr_m[:, pax_idx].clamp_min(0.0)).view(-1, 1)
            w = w / (w.mean() + 1e-8)
            loss = ((pred_m - y_m) ** 2 * w).mean()

        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def evaluate(model, graphs):
    model.eval()
    ys, ps = [], []
    for g in graphs:
        pred = model(g)
        pred_m, y_m, _ = _apply_eval_mask(g, pred)
        ys.append(y_m.detach().cpu().numpy().reshape(-1))
        ps.append(pred_m.detach().cpu().numpy().reshape(-1))

    if not ys:
        return {"rmse": float("nan"), "mae": float("nan")}

    y = np.concatenate(ys)
    p = np.concatenate(ps)
    rmse = math.sqrt(mean_squared_error(y, p))
    mae = float(np.mean(np.abs(y - p)))
    return {"rmse": float(rmse), "mae": mae}


def main():
    seed_everything(SEED)

    df, graphs, train_graphs, val_graphs = load_and_build_graphs(
        DATA_PATH, make_undirected=MAKE_UNDIRECTED, val_ratio=VAL_RATIO
    )
    print(f"[data] rows={len(df)}  graphs={len(graphs)}  train={len(train_graphs)}  val={len(val_graphs)}")

    node_dim = int(graphs[0].x.size(1))
    edge_dim = int(graphs[0].edge_attr.size(1))

    model = FareGNN(node_dim=node_dim, edge_dim=edge_dim, hidden=HIDDEN, n_layers=N_LAYERS)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best = float("inf")
    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(model, train_graphs, opt, pax_weight=False)
        m = evaluate(model, val_graphs)
        if m["rmse"] < best:
            best = m["rmse"]
            torch.save(model.state_dict(), CKPT_PATH)

        if epoch == 1 or epoch % 5 == 0:
            print(f"[epoch {epoch:03d}] loss={tr_loss:.4f}  val_rmse={m['rmse']:.4f}  val_mae={m['mae']:.4f}  best={best:.4f}")

    print(f"[done] best_rmse={best:.4f}  ckpt={CKPT_PATH}")


if __name__ == "__main__":
    main()