# model/graph_builder.py

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from preprocess.data_utils_ext import basic_clean_for_graph_training
from .config_gnn import EDGE_NUM_COLS, REQUIRED_COLS, LEAKY_COLS


def seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _node_features_from_edges(gdf: pd.DataFrame, nodes: np.ndarray) -> np.ndarray:
    node_to_idx = {int(n): i for i, n in enumerate(nodes)}
    n = len(nodes)

    out_deg = np.zeros(n, dtype=np.float32)
    in_deg = np.zeros(n, dtype=np.float32)

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
        in_deg[iv] += 1.0
        s1_sum[iu] += float(s1); s1_cnt[iu] += 1.0
        s2_sum[iv] += float(s2); s2_cnt[iv] += 1.0

    s1_mean = np.divide(s1_sum, np.maximum(s1_cnt, 1.0))
    s2_mean = np.divide(s2_sum, np.maximum(s2_cnt, 1.0))
    return np.stack([out_deg, in_deg, s1_mean, s2_mean], axis=1).astype(np.float32)


def build_graphs_by_time(df: pd.DataFrame, *, make_undirected: bool = True) -> List[Data]:
    graphs: List[Data] = []
    df = df.copy().sort_values(["Year", "quarter"]).reset_index(drop=True)

    for (yy, qq), gdf in df.groupby(["Year", "quarter"], sort=True):
        if len(gdf) < 10:
            continue

        nodes = np.unique(np.concatenate([
            gdf["citymarketid_1"].astype(int).to_numpy(),
            gdf["citymarketid_2"].astype(int).to_numpy(),
        ]))
        nodes = np.sort(nodes)
        node_to_idx = {int(n): i for i, n in enumerate(nodes)}

        u = gdf["citymarketid_1"].astype(int).map(node_to_idx).to_numpy()
        v = gdf["citymarketid_2"].astype(int).map(node_to_idx).to_numpy()
        edge_index = torch.tensor(np.stack([u, v], axis=0), dtype=torch.long)

        edge_attr = torch.tensor(gdf[EDGE_NUM_COLS].astype(float).to_numpy(np.float32), dtype=torch.float32)
        y_edge = torch.tensor(gdf["fare"].astype(float).to_numpy(np.float32), dtype=torch.float32).view(-1, 1)
        x = torch.tensor(_node_features_from_edges(gdf, nodes), dtype=torch.float32)

        # original-edge mask (avoid double counting if undirected)
        undir_mask = torch.ones((edge_index.size(1),), dtype=torch.bool)

        # store original city ids for prediction export
        data_u_city = torch.tensor(gdf["citymarketid_1"].astype(int).to_numpy(), dtype=torch.long)
        data_v_city = torch.tensor(gdf["citymarketid_2"].astype(int).to_numpy(), dtype=torch.long)

        if make_undirected:
            E = edge_index.size(1)
            rev_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
            edge_index = torch.cat([edge_index, rev_edge_index], dim=1)
            edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
            y_edge = torch.cat([y_edge, y_edge], dim=0)
            undir_mask = torch.cat([torch.ones(E, dtype=torch.bool), torch.zeros(E, dtype=torch.bool)], dim=0)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y_edge=y_edge)
        data.year = int(yy)
        data.quarter = int(qq)
        data.undirected_mask = undir_mask
        data.meta_u_city = data_u_city
        data.meta_v_city = data_v_city
        graphs.append(data)

    return graphs


def split_by_time(graphs: List[Data], *, val_ratio: float = 0.2) -> Tuple[List[Data], List[Data]]:
    graphs = sorted(graphs, key=lambda g: (int(g.year), int(g.quarter)))
    n = len(graphs)
    n_val = max(1, int(math.ceil(n * val_ratio)))
    if n > 1:
        n_val = min(n_val, n - 1)
    return graphs[:-n_val], graphs[-n_val:]


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


def standardize_graphs(train_graphs: List[Data], graphs: List[Data]) -> None:
    node_std = fit_standardizer([g.x for g in train_graphs])
    edge_std = fit_standardizer([g.edge_attr for g in train_graphs])

    for g in graphs:
        g.x = node_std.transform(g.x)
        g.edge_attr = edge_std.transform(g.edge_attr)


def load_and_build_graphs(data_path: str, *, make_undirected: bool, val_ratio: float):
    df = basic_clean_for_graph_training(
        data_path,
        money_cols=("fare", "fare_lg", "fare_low"),
        leaky_cols=LEAKY_COLS,
        dropna_subset=REQUIRED_COLS,
    )
    graphs = build_graphs_by_time(df, make_undirected=make_undirected)
    train_graphs, val_graphs = split_by_time(graphs, val_ratio=val_ratio)
    standardize_graphs(train_graphs, graphs)
    return df, graphs, train_graphs, val_graphs