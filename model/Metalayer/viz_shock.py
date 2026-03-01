# model/Metalayer/viz_shockwave.py
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import networkx as nx

from .config_gnn import DATA_PATH, MAKE_UNDIRECTED, VAL_RATIO, CKPT_PATH, EDGE_NUM_COLS
from .graph_builder import load_and_build_graphs, build_graphs_by_time, standardize_graphs
from .gnn_model import FareGNN


def _recompute_strength_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["city1_pax_strength"] = df.groupby(["Year", "quarter", "citymarketid_1"])["passengers"].transform("sum")
    df["city2_pax_strength"] = df.groupby(["Year", "quarter", "citymarketid_2"])["passengers"].transform("sum")
    df["rl_pax_str"] = (df["city1_pax_strength"] - df["city2_pax_strength"]).abs()
    df["tot_pax_str"] = df["city1_pax_strength"] + df["city2_pax_strength"]
    return df


@torch.no_grad()
def _predict_one_graph(model: FareGNN, g) -> pd.DataFrame:
    """
    Return dataframe with (u_city, v_city, pred) on ORIGINAL edges only.
    """
    model.eval()
    pred = model(g).detach().cpu().numpy().reshape(-1)

    mask = getattr(g, "undirected_mask", None)
    mask_np = np.ones_like(pred, dtype=bool) if mask is None else mask.detach().cpu().numpy().astype(bool)

    pred = pred[mask_np]
    u_city = g.meta_u_city.detach().cpu().numpy().astype(int)
    v_city = g.meta_v_city.detach().cpu().numpy().astype(int)

    return pd.DataFrame({"u": u_city, "v": v_city, "pred": pred})


def _load_geopos(csv_path: str = "./data/geocoded_cities.csv"):
    if not os.path.exists(csv_path):
        return None
    geo = pd.read_csv(csv_path)
    # 你文件的列名可能不同；常见是 citymarketid / lat / lon
    # 这里做一个宽松匹配
    col_id = None
    for c in ["citymarketid", "citymarketid_1", "id", "city_id"]:
        if c in geo.columns:
            col_id = c
            break
    if col_id is None:
        return None

    col_lat = None
    col_lon = None
    for c in ["lat", "latitude", "LAT", "Latitude"]:
        if c in geo.columns:
            col_lat = c
            break
    for c in ["lon", "lng", "longitude", "LON", "Longitude"]:
        if c in geo.columns:
            col_lon = c
            break
    if col_lat is None or col_lon is None:
        return None

    pos = {}
    for _, r in geo.iterrows():
        try:
            cid = int(r[col_id])
            pos[cid] = (float(r[col_lon]), float(r[col_lat]))  # (x=lon, y=lat)
        except Exception:
            continue
    return pos if len(pos) > 0 else None


def shockwave_plot(
    year: int,
    quarter: int,
    source_citymarketid: int,
    passengers_multiplier: float = 1.5,   # 冲击幅度：把 incident edges 的 passengers 乘以这个系数
    topk_edges: int = 250,                # 为了清晰：只画最重要的 top-k 航线
    save_path: str = "shockwave.png",
):
    # 1) load graphs the same way as training (df already cleaned by data_utils_ext inside)
    df, graphs, train_graphs, _ = load_and_build_graphs(
        DATA_PATH, make_undirected=MAKE_UNDIRECTED, val_ratio=VAL_RATIO
    )

    # 2) filter target quarter from df (raw, before graph build)
    d0 = df[(df["Year"] == year) & (df["quarter"] == quarter)].copy()
    if len(d0) == 0:
        available = df.groupby(["Year", "quarter"]).size().reset_index().rename({0: "count"}, axis=1)
        print("No rows for Year=", year, ", quarter=", quarter)
        print("Available (Year, quarter) pairs:")
        print(available[["Year", "quarter", "count"]])
        raise ValueError(f"No rows for Year={year}, quarter={quarter}. See above for available pairs.")

    # 3) build baseline graph from this quarter only (then standardize using train_graphs stats)
    d0 = _recompute_strength_features(d0)
    g0_list = build_graphs_by_time(d0, make_undirected=MAKE_UNDIRECTED)
    if len(g0_list) != 1:
        # build_graphs_by_time groups by (Year,quarter); with one quarter should be 1
        raise RuntimeError(f"Expected 1 graph, got {len(g0_list)}")
    g0 = g0_list[0]
    standardize_graphs(train_graphs, [g0])

    # 4) build shocked df: scale passengers on edges incident to the source city
    d1 = d0.copy()
    mask_incident = (d1["citymarketid_1"] == source_citymarketid) | (d1["citymarketid_2"] == source_citymarketid)
    if mask_incident.sum() == 0:
        raise ValueError(f"source_citymarketid={source_citymarketid} not present in this quarter graph.")
    d1.loc[mask_incident, "passengers"] = d1.loc[mask_incident, "passengers"].astype(float) * passengers_multiplier
    d1 = _recompute_strength_features(d1)

    g1_list = build_graphs_by_time(d1, make_undirected=MAKE_UNDIRECTED)
    g1 = g1_list[0]
    standardize_graphs(train_graphs, [g1])

    # 5) load model
    node_dim = int(g0.x.size(1))
    edge_dim = int(g0.edge_attr.size(1))
    model = FareGNN(node_dim=node_dim, edge_dim=edge_dim)
    state = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict(state)

    # 6) predict baseline vs shocked
    p0 = _predict_one_graph(model, g0)
    p1 = _predict_one_graph(model, g1)

    m = p0.merge(p1, on=["u", "v"], how="inner", suffixes=("_base", "_shock"))
    m["delta"] = m["pred_shock"] - m["pred_base"]
    m["abs_delta"] = m["delta"].abs()

    # 7) choose edges to draw (keep figure readable)
    m_draw = m.sort_values("abs_delta", ascending=False).head(topk_edges).copy()

    # node impact = sum |delta| over incident edges
    node_impact = {}
    for _, r in m.iterrows():
        node_impact[r["u"]] = node_impact.get(r["u"], 0.0) + float(r["abs_delta"])
        node_impact[r["v"]] = node_impact.get(r["v"], 0.0) + float(r["abs_delta"])

    # 8) build networkx graph for plotting
    G = nx.Graph()
    for _, r in m_draw.iterrows():
        G.add_edge(int(r["u"]), int(r["v"]), delta=float(r["delta"]), abs_delta=float(r["abs_delta"]))

    # positions: geo if available, else spring
    pos_geo = _load_geopos("./data/geocoded_cities.csv")
    if pos_geo is not None and all(n in pos_geo for n in G.nodes()):
        pos = {n: pos_geo[n] for n in G.nodes()}
    else:
        pos = nx.spring_layout(G, seed=42)

    # 9) plot (two panels)
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    # --- Left: Baseline network (edge width by abs_delta just for visibility; or use constant)
    ax1.set_title(f"Baseline Network (Year={year} Q{quarter})")
    ax1.axis("off")
    nx.draw_networkx_nodes(G, pos, ax=ax1, node_size=30)
    nx.draw_networkx_edges(G, pos, ax=ax1, width=0.8, alpha=0.6)
    nx.draw_networkx_nodes(
        G, pos, ax=ax1,
        nodelist=[source_citymarketid] if source_citymarketid in G.nodes() else [],
        node_size=120
    )

    # --- Right: Shockwave (edge width ~ abs_delta, edge color ~ sign(delta), node size ~ node_impact)
    ax2.set_title(f"Shockwave: multiply passengers ×{passengers_multiplier} at city {source_citymarketid}")
    ax2.axis("off")

    nodes = list(G.nodes())
    sizes = []
    for n in nodes:
        s = node_impact.get(n, 0.0)
        # scale for visibility
        sizes.append(20 + 600 * (s / (max(node_impact.values()) + 1e-12)))

    edges = list(G.edges(data=True))
    widths = [0.5 + 6.0 * (d["abs_delta"] / (max(m_draw["abs_delta"]) + 1e-12)) for (_, _, d) in edges]

    # 用灰度不太能表达正负号；但如果你允许颜色，就把正负分开两类
    pos_edges = [(u, v) for (u, v, d) in edges if d["delta"] >= 0]
    neg_edges = [(u, v) for (u, v, d) in edges if d["delta"] < 0]

    nx.draw_networkx_nodes(G, pos, ax=ax2, nodelist=nodes, node_size=sizes)
    nx.draw_networkx_edges(G, pos, ax=ax2, edgelist=pos_edges, width=widths, alpha=0.7)
    nx.draw_networkx_edges(G, pos, ax=ax2, edgelist=neg_edges, width=widths, alpha=0.7, style="dashed")

    nx.draw_networkx_nodes(
        G, pos, ax=ax2,
        nodelist=[source_citymarketid] if source_citymarketid in G.nodes() else [],
        node_size=180
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"[ok] saved -> {save_path}")


if __name__ == "__main__":
    # 示例：你需要换成你想展示的季度 + hub 城市 id
    shockwave_plot(
        year=2022,
        quarter=1,
        source_citymarketid=30194,  # 请确保此 citymarketid 存在于 2022 Q1 的数据中
        passengers_multiplier=1.5,
        topk_edges=250,
        save_path="shockwave.png",
    )