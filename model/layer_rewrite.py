# layer_rewrite.py
# One-file, no-leak MetaLayer (edge-state MPNN) for fare prediction, refactored to use data_utils_ext.
#
# Requirements:
#   pip/conda install torch torch_geometric pandas openpyxl scikit-learn
#
# Key safety changes vs original:
# - Centralize loading/cleaning/strength features in data_utils_ext (xlsx supported)
# - Avoid full-table dropna; only drop minimal subset needed
# - If make_undirected=True, attach undirected_mask so loss/metrics are computed on original edges only
# - Default-remove suspicious "*Fared*" columns from EDGE_NUM_COLS (you can re-add after verifying no leakage)

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import MetaLayer
from torch_geometric.utils import scatter

from sklearn.metrics import mean_squared_error

from preprocess.data_utils_ext import basic_clean_for_graph_training, sanity_report


# -----------------------------
# Config
# -----------------------------

DATA_PATH = "./data/airline_ticket_dataset.xlsx"   # xlsx input
MAKE_UNDIRECTED = True
VAL_RATIO = 0.2
SEED = 42

# If you discover any target-encoded / price-aggregated columns, add here.
LEAKY_COLS = ["fare_lg", "fare_low"]

# Safe edge numeric features (NO fare/fare_* allowed)
EDGE_NUM_COLS = [
    "nsmiles",
    "passengers",
    "large_ms",
    "lf_ms",
    # strength features (computed from passengers only)
    "city1_pax_strength",
    "city2_pax_strength",
    "rl_pax_str",
    "tot_pax_str",
    # NOTE: removed by default due to leakage risk from the name "*Fared*".
    # Re-add only after verifying they are not derived from fare / target encodings:
    # "TotalPerLFMkts_city1",
    # "TotalPerPrem_city1",
    # "TotalPerLFMkts_city2",
    # "TotalPerPrem_city2",
]

# Minimal columns needed to build graphs + y
REQUIRED_COLS = ["Year", "quarter", "citymarketid_1", "citymarketid_2", "fare", "passengers"]


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Graph building
# -----------------------------

def _node_features_from_edges(gdf: pd.DataFrame, nodes: np.ndarray) -> np.ndarray:
    """
    Build simple node features from edges only (leakage-safe):
      - out_degree, in_degree
      - avg city1_pax_strength, avg city2_pax_strength across incident edges
    """
    node_to_idx = {int(n): i for i, n in enumerate(nodes)}
    n = len(nodes)

    out_deg = np.zeros(n, dtype=np.float32)
    in_deg  = np.zeros(n, dtype=np.float32)

    # Strength aggregates
    s1_sum = np.zeros(n, dtype=np.float32)
    s1_cnt = np.zeros(n, dtype=np.float32)
    s2_sum = np.zeros(n, dtype=np.float32)
    s2_cnt = np.zeros(n, dtype=np.float32)

    c1 = gdf["citymarketid_1"].astype(int).to_numpy()
    c2 = gdf["citymarketid_2"].astype(int).to_numpy()

    city1_strength = gdf["city1_pax_strength"].astype(float).to_numpy()
    city2_strength = gdf["city2_pax_strength"].astype(float).to_numpy()

    for u, v, s1, s2 in zip(c1, c2, city1_strength, city2_strength):
        iu = node_to_idx.get(int(u))
        iv = node_to_idx.get(int(v))
        if iu is None or iv is None:
            continue
        out_deg[iu] += 1.0
        in_deg[iv]  += 1.0

        s1_sum[iu] += float(s1); s1_cnt[iu] += 1.0
        s2_sum[iv] += float(s2); s2_cnt[iv] += 1.0

    s1_mean = np.divide(s1_sum, np.maximum(s1_cnt, 1.0))
    s2_mean = np.divide(s2_sum, np.maximum(s2_cnt, 1.0))

    x = np.stack([out_deg, in_deg, s1_mean, s2_mean], axis=1).astype(np.float32)
    return x


def build_graphs_by_time(df: pd.DataFrame, *, make_undirected: bool = True) -> List[Data]:
    graphs: List[Data] = []
    df = df.copy()

    # Ensure required cols exist
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Sort by time
    df = df.sort_values(["Year", "quarter"]).reset_index(drop=True)

    for (yy, qq), gdf in df.groupby(["Year", "quarter"], sort=True):
        if len(gdf) < 10:
            continue

        # Nodes
        nodes = np.unique(
            np.concatenate([
                gdf["citymarketid_1"].astype(int).to_numpy(),
                gdf["citymarketid_2"].astype(int).to_numpy(),
            ])
        )
        nodes = np.sort(nodes)
        node_to_idx = {int(n): i for i, n in enumerate(nodes)}

        # Edge index (directed)
        u = gdf["citymarketid_1"].astype(int).map(node_to_idx).to_numpy()
        v = gdf["citymarketid_2"].astype(int).map(node_to_idx).to_numpy()
        edge_index = torch.tensor(np.stack([u, v], axis=0), dtype=torch.long)

        # Edge attrs
        edge_attr = gdf[EDGE_NUM_COLS].astype(float).to_numpy(dtype=np.float32)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

        # Target
        y_edge = torch.tensor(gdf["fare"].astype(float).to_numpy(dtype=np.float32), dtype=torch.float32).view(-1, 1)

        # Node feats (derived from edges only)
        x_np = _node_features_from_edges(gdf, nodes)
        x = torch.tensor(x_np, dtype=torch.float32)

        # Mask: original edges are True; reversed copies are False
        undir_mask = torch.ones((edge_index.size(1),), dtype=torch.bool)

        if make_undirected:
            E = edge_index.size(1)
            rev_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
            edge_index = torch.cat([edge_index, rev_edge_index], dim=1)
            edge_attr  = torch.cat([edge_attr, edge_attr], dim=0)
            y_edge     = torch.cat([y_edge, y_edge], dim=0)
            undir_mask = torch.cat(
                [torch.ones(E, dtype=torch.bool), torch.zeros(E, dtype=torch.bool)],
                dim=0,
            )

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y_edge=y_edge)
        data.year = int(yy)
        data.quarter = int(qq)
        data.undirected_mask = undir_mask
        graphs.append(data)

    return graphs


def split_by_time(graphs: List[Data], *, val_ratio: float = 0.2) -> Tuple[List[Data], List[Data]]:
    graphs = sorted(graphs, key=lambda g: (int(g.year), int(g.quarter)))
    n = len(graphs)
    n_val = max(1, int(math.ceil(n * val_ratio)))
    if n > 1:
        n_val = min(n_val, n - 1)
    return graphs[:-n_val], graphs[-n_val:]


# -----------------------------
# Standardization (train stats only)
# -----------------------------

@dataclass
class Standardizer:
    mean: torch.Tensor
    std: torch.Tensor

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


def fit_standardizer(xs: List[torch.Tensor]) -> Standardizer:
    X = torch.cat(xs, dim=0)
    mean = X.mean(dim=0, keepdim=True)
    std = X.std(dim=0, keepdim=True)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)
    return Standardizer(mean=mean, std=std)


def standardize_graphs(train_graphs: List[Data], val_graphs: List[Data]) -> None:
    node_std = fit_standardizer([g.x for g in train_graphs])
    edge_std = fit_standardizer([g.edge_attr for g in train_graphs])

    for g in train_graphs + val_graphs:
        g.x = node_std.transform(g.x)
        g.edge_attr = edge_std.transform(g.edge_attr)


# -----------------------------
# Model
# -----------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, p_drop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EdgeModel(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden: int):
        super().__init__()
        self.mlp = MLP(node_dim * 2 + edge_dim, hidden, edge_dim)

    def forward(self, src, dst, edge_attr, u=None, batch=None):
        h = torch.cat([src, dst, edge_attr], dim=1)
        return self.mlp(h)


class NodeModel(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden: int):
        super().__init__()
        self.mlp = MLP(node_dim + edge_dim, hidden, node_dim)

    def forward(self, x, edge_index, edge_attr, u=None, batch=None):
        _, col = edge_index  # messages to node col
        # agg = scatter_add(edge_attr, col, dim=0, dim_size=x.size(0))
        agg = scatter(edge_attr, col, dim=0, dim_size=x.size(0), reduce="sum")
        h = torch.cat([x, agg], dim=1)
        return self.mlp(h)


class FareGNN(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden: int = 64, n_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([
            MetaLayer(EdgeModel(node_dim, edge_dim, hidden), NodeModel(node_dim, edge_dim, hidden), None)
            for _ in range(n_layers)
        ])
        self.readout = MLP(edge_dim, hidden, 1)

    def forward(self, g: Data) -> torch.Tensor:
        x, edge_index, edge_attr = g.x, g.edge_index, g.edge_attr
        for layer in self.layers:
            x, edge_attr, _ = layer(x, edge_index, edge_attr, None, None)
        return self.readout(edge_attr)  # (E, 1)


# -----------------------------
# Train / Eval
# -----------------------------

def _apply_eval_mask(g: Data, pred: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    y = g.y_edge
    mask = getattr(g, "undirected_mask", None)
    if mask is None:
        return pred, y, g.edge_attr
    return pred[mask], y[mask], g.edge_attr[mask]


def train_one_epoch(
    model: nn.Module,
    graphs: List[Data],
    opt: torch.optim.Optimizer,
    *,
    pax_weight: bool = False,
    pax_col: str = "passengers",
) -> float:
    model.train()
    losses = []

    # optional pax weighting (log1p(passengers))
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
def evaluate(model: nn.Module, graphs: List[Data]) -> Dict[str, float]:
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


# -----------------------------
# Main
# -----------------------------

def main():
    seed_everything(SEED)

    df = basic_clean_for_graph_training(
        DATA_PATH,
        money_cols=("fare", "fare_lg", "fare_low"),
        leaky_cols=LEAKY_COLS,
        dropna_subset=REQUIRED_COLS,
    )

    print("[data] sanity:", sanity_report(df))

    graphs = build_graphs_by_time(df, make_undirected=MAKE_UNDIRECTED)
    print(f"[graphs] built: {len(graphs)}")

    train_graphs, val_graphs = split_by_time(graphs, val_ratio=VAL_RATIO)
    print(f"[split] train={len(train_graphs)}  val={len(val_graphs)}")

    standardize_graphs(train_graphs, val_graphs)

    # Model dims from first graph
    node_dim = int(train_graphs[0].x.size(1))
    edge_dim = int(train_graphs[0].edge_attr.size(1))

    model = FareGNN(node_dim=node_dim, edge_dim=edge_dim, hidden=64, n_layers=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best = float("inf")
    for epoch in range(1, 51):
        tr_loss = train_one_epoch(model, train_graphs, opt, pax_weight=False)
        metrics = evaluate(model, val_graphs)
        if metrics["rmse"] < best:
            best = metrics["rmse"]
        if epoch % 5 == 0 or epoch == 1:
            print(f"[epoch {epoch:03d}] loss={tr_loss:.4f}  val_rmse={metrics['rmse']:.4f}  val_mae={metrics['mae']:.4f}  best={best:.4f}")


if __name__ == "__main__":
    main()
