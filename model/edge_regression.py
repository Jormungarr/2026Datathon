import os
import math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric import data

from preprocess import data_utils
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv


# =========================
# Config
# =========================
DATA_PATH = "data/airline_ticket_dataset.xlsx" 
VAL_YEAR = 2025
SEED = 42

EPOCHS = 80
LR = 2e-3
WEIGHT_DECAY = 1e-5
HIDDEN = 96
DROPOUT = 0.15
PATIENCE = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs("outputs", exist_ok=True)


# =========================
# Utils
# =========================
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def qid(year: int, q: int) -> int:
    # q in {1..4}
    return int(year) * 4 + int(q)


def qid_to_year(qid_: int) -> int:
    return int(qid_ // 4)


def canonical_route(a: int, b: int):
    a = int(a); b = int(b)
    return (a, b) if a <= b else (b, a)


def mode_or_unk(s: pd.Series) -> str:
    s = s.dropna()
    if len(s) == 0:
        return "UNK"
    m = s.mode()
    if len(m) == 0:
        return "UNK"
    return str(m.iloc[0])


def masked_rmse_mae(pred: torch.Tensor, y: torch.Tensor):
    mask = torch.isfinite(y)
    if mask.sum().item() == 0:
        return None
    e = pred[mask] - y[mask]
    rmse = torch.sqrt(torch.mean(e * e)).item()
    mae = torch.mean(torch.abs(e)).item()
    return rmse, mae, int(mask.sum().item())


class Standardizer:
    """Fit on train only; transform numpy arrays (float32)."""
    def __init__(self, eps=1e-6):
        self.eps = eps
        self.mean = None
        self.std = None

    def fit(self, X: np.ndarray):
        m = np.nanmean(X, axis=0)
        s = np.nanstd(X, axis=0)
        s = np.where(s < self.eps, 1.0, s)
        self.mean = m.astype(np.float32)
        self.std = s.astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = X.astype(np.float32)
        X = np.where(np.isfinite(X), X, np.nan)
        # impute NaN to mean
        X = np.where(np.isfinite(X), X, self.mean[None, :])
        return (X - self.mean[None, :]) / self.std[None, :]


# =========================
# 1) Read + aggregate to route-quarter
# =========================
def build_route_quarter(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    df["quarter_id"] = df.apply(lambda r: qid(r["Year"], r["quarter"]), axis=1)

    rt = df.apply(lambda r: canonical_route(r["citymarketid_1"], r["citymarketid_2"]),
                  axis=1, result_type="expand")
    df["u_city"] = rt[0].astype(int)
    df["v_city"] = rt[1].astype(int)

    keys = ["quarter_id", "Year", "quarter", "u_city", "v_city"]
    g = df.groupby(keys, sort=False)

    pax_sum = g["passengers"].sum(min_count=1).rename("passengers_sum")
    nsmiles = g["nsmiles"].mean().rename("nsmiles")

    # weighted avg with passengers
    def wavg(col):
        num = (df[col] * df["passengers"]).groupby([df[k] for k in keys]).sum(min_count=1)
        den = df["passengers"].groupby([df[k] for k in keys]).sum(min_count=1)
        return (num / den)

    fare_wavg = wavg("fare").rename("fare_wavg")
    large_ms_wavg = wavg("large_ms").rename("large_ms_wavg")
    lf_ms_wavg = wavg("lf_ms").rename("lf_ms_wavg")
    fare_lg_wavg = wavg("fare_lg").rename("fare_lg_wavg")
    fare_low_wavg = wavg("fare_low").rename("fare_low_wavg")

    carrier_lg = g["carrier_lg"].agg(mode_or_unk).rename("carrier_lg")
    carrier_low = g["carrier_low"].agg(mode_or_unk).rename("carrier_low")

    out = pd.concat(
        [pax_sum, nsmiles, fare_wavg, large_ms_wavg, lf_ms_wavg,
         fare_lg_wavg, fare_low_wavg, carrier_lg, carrier_low],
        axis=1
    ).reset_index()

    out["log_pax_edge"] = np.log1p(out["passengers_sum"].astype(float))
    return out


def build_city_quarter_macro(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Stack city1/city2 macro into one table: (quarter_id, citymarketid) -> macro features."""
    df = df_raw.copy()
    df["quarter_id"] = df.apply(lambda r: qid(r["Year"], r["quarter"]), axis=1)

    a = df[["quarter_id", "Year", "quarter", "citymarketid_1",
            "TotalFaredPax_city1", "TotalPerLFMkts_city1", "TotalPerPrem_city1"]].copy()
    a = a.rename(columns={
        "citymarketid_1": "citymarketid",
        "TotalFaredPax_city1": "TotalFaredPax_city",
        "TotalPerLFMkts_city1": "TotalPerLFMkts_city",
        "TotalPerPrem_city1": "TotalPerPrem_city",
    })

    b = df[["quarter_id", "Year", "quarter", "citymarketid_2",
            "TotalFaredPax_city2", "TotalPerLFMkts_city2", "TotalPerPrem_city2"]].copy()
    b = b.rename(columns={
        "citymarketid_2": "citymarketid",
        "TotalFaredPax_city2": "TotalFaredPax_city",
        "TotalPerLFMkts_city2": "TotalPerLFMkts_city",
        "TotalPerPrem_city2": "TotalPerPrem_city",
    })

    c = pd.concat([a, b], axis=0, ignore_index=True)
    c["citymarketid"] = c["citymarketid"].astype(int)

    gm = c.groupby(["quarter_id", "Year", "quarter", "citymarketid"], sort=False)[
        ["TotalFaredPax_city", "TotalPerLFMkts_city", "TotalPerPrem_city"]
    ].mean().reset_index()

    return gm


# =========================
# 2) Build per-quarter graphs with y_edge = fare(t+1)
# =========================
def build_graphs(route_q: pd.DataFrame, city_q: pd.DataFrame):
    qids = sorted(route_q["quarter_id"].unique())

    # global city set for stable node indexing across time
    all_cities = pd.Index(pd.unique(route_q[["u_city", "v_city"]].values.ravel()))
    city2idx = {int(c): i for i, c in enumerate(all_cities)}
    idx2city = {i: int(c) for c, i in city2idx.items()}
    n_nodes = len(all_cities)

    # (quarter_id, u_city, v_city) -> fare_wavg
    fare_map = {
        (int(r.quarter_id), int(r.u_city), int(r.v_city)): float(r.fare_wavg)
        for r in route_q.itertuples(index=False)
        if np.isfinite(r.fare_wavg)
    }

    # (quarter_id, citymarketid) -> macro(3)
    macro_map = {}
    for r in city_q.itertuples(index=False):
        macro_map[(int(r.quarter_id), int(r.citymarketid))] = (
            float(r.TotalFaredPax_city) if np.isfinite(r.TotalFaredPax_city) else np.nan,
            float(r.TotalPerLFMkts_city) if np.isfinite(r.TotalPerLFMkts_city) else np.nan,
            float(r.TotalPerPrem_city) if np.isfinite(r.TotalPerPrem_city) else np.nan,
        )

    graphs = []
    for qi in qids:
        dft = route_q[route_q["quarter_id"] == qi].copy()
        if len(dft) == 0:
            continue

        # ----- edges (E) in canonical undirected form -----
        u_idx = dft["u_city"].map(city2idx).to_numpy()
        v_idx = dft["v_city"].map(city2idx).to_numpy()

        # Create explicit bidirectional edges (2E) with aligned edge_attr and y_edge
        src = np.concatenate([u_idx, v_idx])
        dst = np.concatenate([v_idx, u_idx])
        edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)

        # edge numeric features (E,F)
        edge_num_raw = np.column_stack([
            dft["log_pax_edge"].to_numpy(dtype=float),
            dft["nsmiles"].to_numpy(dtype=float),
            dft["large_ms_wavg"].to_numpy(dtype=float),
            dft["lf_ms_wavg"].to_numpy(dtype=float),
            dft["fare_wavg"].to_numpy(dtype=float),       # lag feature at t
            dft["fare_lg_wavg"].to_numpy(dtype=float),
            dft["fare_low_wavg"].to_numpy(dtype=float),
        ]).astype(np.float32)

        fare_t_raw = dft["fare_wavg"].to_numpy(dtype=np.float32)
        fare_t = torch.tensor(np.concatenate([fare_t_raw, fare_t_raw]), dtype=torch.float)


        edge_num = torch.tensor(np.vstack([edge_num_raw, edge_num_raw]), dtype=torch.float)

        # ---- undirected mask: only count each (u,v) once ----
        E = edge_num_raw.shape[0]  # number of undirected routes in this quarter
        undirected_mask = np.zeros(2 * E, dtype=bool)
        undirected_mask[:E] = True  # first E are the canonical direction (u_city <= v_city)

        undirected_mask = torch.tensor(undirected_mask, dtype=torch.bool)

        # carrier categorical indices (2E,)
        # store as strings now; will map to ids later
        carrier_lg = np.concatenate([dft["carrier_lg"].astype(str).to_numpy(),
                                     dft["carrier_lg"].astype(str).to_numpy()])
        carrier_low = np.concatenate([dft["carrier_low"].astype(str).to_numpy(),
                                      dft["carrier_low"].astype(str).to_numpy()])

        # y_edge: fare at (t+1) for the same canonical route; missing -> NaN
        y_raw = []
        for r in dft.itertuples(index=False):
            y = fare_map.get((int(qi + 1), int(r.u_city), int(r.v_city)), np.nan)
            y_raw.append(y)

        # y_edge = fare(t+1) - fare(t)
        y_raw = np.array(y_raw, dtype=np.float32)               # fare(t+1) for E undirected edges
        delta_raw = y_raw - fare_t_raw                           # (E,)
        y_edge = torch.tensor(np.concatenate([delta_raw, delta_raw]), dtype=torch.float)
        # y_raw = np.array(y_raw, dtype=np.float32)

        # y_edge = torch.tensor(np.concatenate([y_raw, y_raw]), dtype=torch.float)

        # ----- node features (N, 3+3) -----
        # strength / degree / HHI computed from current quarter edges
        pax = dft["passengers_sum"].to_numpy(dtype=float)
        strength = np.zeros(n_nodes, dtype=float)
        neigh = [set() for _ in range(n_nodes)]
        inc = [[] for _ in range(n_nodes)]

        for uu, vv, w in zip(u_idx, v_idx, pax):
            strength[uu] += w
            strength[vv] += w
            neigh[uu].add(vv); neigh[vv].add(uu)
            inc[uu].append(w); inc[vv].append(w)

        degree = np.array([len(s) for s in neigh], dtype=float)
        hhi = np.zeros(n_nodes, dtype=float)
        for i in range(n_nodes):
            tot = sum(inc[i])
            if tot > 0:
                shares = np.array(inc[i], dtype=float) / tot
                hhi[i] = float(np.sum(shares * shares))

        # macro per node
        macro = np.full((n_nodes, 3), np.nan, dtype=float)
        for i in range(n_nodes):
            cid = idx2city[i]
            if (qi, cid) in macro_map:
                macro[i, :] = np.array(macro_map[(qi, cid)], dtype=float)

        node_x = np.column_stack([
            np.log1p(strength),
            degree,
            hhi,
            macro
        ]).astype(np.float32)

        data = Data(
            x=torch.tensor(node_x, dtype=torch.float),
            edge_index=edge_index,
            edge_num=edge_num,         # numeric edge features
            y_edge=y_edge
        )

        data.fare_t = fare_t

        data.undirected_mask = undirected_mask
        data.E_undirected = E  # optional, for debugging

        data.quarter_id = int(qi)
        data.year = int(dft["Year"].iloc[0])

        # keep raw carrier strings for mapping
        data.carrier_lg_str = carrier_lg
        data.carrier_low_str = carrier_low

        # keep edge endpoints in city ids for export (aligned with 2E)
        u_city = np.concatenate([dft["u_city"].to_numpy(dtype=int), dft["v_city"].to_numpy(dtype=int)])
        v_city = np.concatenate([dft["v_city"].to_numpy(dtype=int), dft["u_city"].to_numpy(dtype=int)])
        data.u_city = torch.tensor(u_city, dtype=torch.long)
        data.v_city = torch.tensor(v_city, dtype=torch.long)

        graphs.append(data)

    return graphs


# =========================
# 3) Year holdout split
# =========================
def split_train_val(graphs, val_year=2025):
    train = []
    val = []
    for g in graphs:
        yt = qid_to_year(g.quarter_id)
        yt1 = qid_to_year(g.quarter_id + 1)

        # Train pairs: both t and t+1 are before val_year
        if (yt < val_year) and (yt1 < val_year):
            train.append(g)

        # Val pairs: target quarter is in val_year (t+1 in 2025)
        if yt1 == val_year:
            val.append(g)

    return train, val


# =========================
# 4) Model (uses topology via message passing)
# =========================
class EdgeGNN(nn.Module):
    def __init__(self, node_in, edge_num_in, n_carrier_lg, n_carrier_low, hidden=96, dropout=0.15):
        super().__init__()
        emb_dim = 16
        self.emb_lg = nn.Embedding(n_carrier_lg, emb_dim)
        self.emb_low = nn.Embedding(n_carrier_low, emb_dim)

        self.edge_in = edge_num_in + 2 * emb_dim

        self.node_enc = nn.Sequential(
            nn.Linear(node_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )

        # project edge features to hidden so GINE message x_j + edge_attr matches dim
        self.edge_proj = nn.Sequential(
            nn.Linear(self.edge_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )

        nn_edge = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.conv = GINEConv(nn_edge)

        self.norm = nn.LayerNorm(hidden)

        # edge head predicts delta (scalar)
        self.edge_head = nn.Sequential(
            nn.Linear(hidden * 2 + self.edge_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

    def forward(self, data: Data):
        lg = self.emb_lg(data.carrier_lg_idx)
        lo = self.emb_low(data.carrier_low_idx)
        edge_attr_raw = torch.cat([data.edge_num, lg, lo], dim=1)   # (E, edge_in)
        edge_attr = self.edge_proj(edge_attr_raw)                   # (E, hidden)

        x0 = self.node_enc(data.x)                                  # (N, hidden)
        m = self.conv(x0, data.edge_index, edge_attr).relu()        # (N, hidden)
        x = self.norm(x0 + m)                                       # residual + norm

        src, dst = data.edge_index
        z = torch.cat([x[src], x[dst], edge_attr_raw], dim=1)
        delta = self.edge_head(z).squeeze(1)                        # predicted delta
        return delta


# =========================
# 5) Main training
# =========================
def main():
    set_seed(SEED)
    print("Device:", DEVICE)

    raw = data_utils.import_orginal_dataset()
    route_q = build_route_quarter(raw)
    city_q = build_city_quarter_macro(raw)
    graphs = build_graphs(route_q, city_q)

    train_graphs, val_graphs = split_train_val(graphs, val_year=VAL_YEAR)
    print(f"Graphs: total={len(graphs)} train_pairs={len(train_graphs)} val_pairs={len(val_graphs)}")

    # --- build carrier vocab using TRAIN ONLY (avoid leakage) ---
    lg_set = set(["UNK"])
    low_set = set(["UNK"])
    for g in train_graphs:
        lg_set.update(list(g.carrier_lg_str))
        low_set.update(list(g.carrier_low_str))

    lg_list = sorted(lg_set)
    low_list = sorted(low_set)
    lg2id = {s: i for i, s in enumerate(lg_list)}
    low2id = {s: i for i, s in enumerate(low_list)}
    print(f"carrier_lg vocab={len(lg_list)} carrier_low vocab={len(low_list)}")

    # map carrier strings -> ids for all graphs (UNK if unseen in train)
    for g in graphs:
        lg_idx = np.array([lg2id.get(s, lg2id["UNK"]) for s in g.carrier_lg_str], dtype=np.int64)
        low_idx = np.array([low2id.get(s, low2id["UNK"]) for s in g.carrier_low_str], dtype=np.int64)
        g.carrier_lg_idx = torch.tensor(lg_idx, dtype=torch.long)
        g.carrier_low_idx = torch.tensor(low_idx, dtype=torch.long)

        # free raw strings (optional)
        delattr(g, "carrier_lg_str")
        delattr(g, "carrier_low_str")

    # --- standardize node & edge numeric features using TRAIN ONLY ---
    node_stack = np.vstack([g.x.cpu().numpy() for g in train_graphs])
    edge_stack = np.vstack([g.edge_num.cpu().numpy() for g in train_graphs])

    node_scaler = Standardizer().fit(node_stack)
    edge_scaler = Standardizer().fit(edge_stack)

    for g in graphs:
        g.x = torch.tensor(node_scaler.transform(g.x.cpu().numpy()), dtype=torch.float)
        g.edge_num = torch.tensor(edge_scaler.transform(g.edge_num.cpu().numpy()), dtype=torch.float)

    # move graphs to device
    for g in graphs:
        g.x = g.x.to(DEVICE)
        g.edge_index = g.edge_index.to(DEVICE)
        g.edge_num = g.edge_num.to(DEVICE)
        g.carrier_lg_idx = g.carrier_lg_idx.to(DEVICE)
        g.carrier_low_idx = g.carrier_low_idx.to(DEVICE)
        g.y_edge = g.y_edge.to(DEVICE)
        g.u_city = g.u_city.to(DEVICE)
        g.v_city = g.v_city.to(DEVICE)
        g.undirected_mask = g.undirected_mask.to(DEVICE)   # <<< ADD THIS
        g.fare_t = g.fare_t.to(DEVICE)

    model = EdgeGNN(
        node_in=train_graphs[0].x.size(1),
        edge_num_in=train_graphs[0].edge_num.size(1),
        n_carrier_lg=len(lg_list),
        n_carrier_low=len(low_list),
        hidden=HIDDEN,
        dropout=DROPOUT
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val = float("inf")
    best_state = None
    patience = 0

    def run_epoch(graph_list, train: bool):
        if train:
            model.train()
        else:
            model.eval()

        losses = []
        rmses = []
        maes = []
        ns = []

        for g in graph_list:
            pred = model(g)
            y = g.y_edge

            mask = torch.isfinite(y) & g.undirected_mask  # torch.bool

            if mask.sum().item() == 0:
                continue

            # loss = F.mse_loss(pred[mask], y[mask])
            loss = F.smooth_l1_loss(pred[mask], y[mask], beta=1.0)  # 或者 beta=2.0

            if train:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()

            with torch.no_grad():
                e = pred[mask] - y[mask]
                rmse = torch.sqrt(torch.mean(e * e)).item()
                mae = torch.mean(torch.abs(e)).item()
                n = int(mask.sum().item())

            rmses.append(rmse); maes.append(mae); ns.append(n)

            losses.append(loss.item())

        if len(losses) == 0:
            return None

        # weighted by number of labeled edges
        if len(ns) > 0:
            w = np.array(ns, dtype=float)
            rmse_w = float(np.sum(w * np.array(rmses)) / np.sum(w))
            mae_w = float(np.sum(w * np.array(maes)) / np.sum(w))
        else:
            rmse_w = float("nan")
            mae_w = float("nan")

        return float(np.mean(losses)), rmse_w, mae_w

    for ep in range(1, EPOCHS + 1):
        tr = run_epoch(train_graphs, train=True)
        va = run_epoch(val_graphs, train=False)

        if tr is None or va is None:
            print("No labeled edges found. Check y_edge construction.")
            return

        tr_loss, tr_rmse, tr_mae = tr
        va_loss, va_rmse, va_mae = va

        print(f"Epoch {ep:03d} | "
              f"train RMSE={tr_rmse:.3f} MAE={tr_mae:.3f} | "
              f"val RMSE={va_rmse:.3f} MAE={va_mae:.3f}")

        if va_rmse < best_val:
            best_val = va_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                print("Early stop.")
                break

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # =====================
    # Export validation predictions (2025 holdout pairs)
    # =====================
    model.eval()
    rows = []
    with torch.no_grad():
        for g in val_graphs:
            # delta prediction / delta true
            delta_pred_np = model(g).detach().cpu().numpy()
            delta_true_np = g.y_edge.detach().cpu().numpy()

            # fare at time t (must exist)
            fare_t_np = g.fare_t.detach().cpu().numpy()

            # undirected mask
            undir_np = g.undirected_mask.detach().cpu().numpy().astype(bool)

            # valid rows
            mask = np.isfinite(delta_true_np) & np.isfinite(fare_t_np) & undir_np

            u = g.u_city.detach().cpu().numpy()
            v = g.v_city.detach().cpu().numpy()

            fare_true_t1 = fare_t_np + delta_true_np
            fare_pred_t1 = fare_t_np + delta_pred_np

            for uu, vv, ft, yt, yp in zip(
                u[mask], v[mask],
                fare_t_np[mask],
                fare_true_t1[mask],
                fare_pred_t1[mask],
            ):
                rows.append({
                    "quarter_id_t": g.quarter_id,
                    "year_t": qid_to_year(g.quarter_id),
                    "quarter_t": int((g.quarter_id % 4) if (g.quarter_id % 4) != 0 else 4),
                    "u_city": int(uu),
                    "v_city": int(vv),
                    "fare_t": float(ft),                # 可选：写出来方便诊断
                    "fare_true_t1": float(yt),
                    "fare_pred_t1": float(yp),
                    "abs_error": float(abs(yt - yp)),
                })

    out = pd.DataFrame(rows)
    out_path = "outputs/gnn_edge_val_predictions_2025_holdout.csv"
    out.to_csv(out_path, index=False)
    print("Saved:", out_path)

    # also print overall val metrics
    if len(out) > 0:
        rmse = math.sqrt(np.mean((out["fare_true_t1"] - out["fare_pred_t1"]) ** 2))
        mae = np.mean(np.abs(out["fare_true_t1"] - out["fare_pred_t1"]))
        print(f"[VAL 2025 holdout] RMSE={rmse:.4f} MAE={mae:.4f} | n={len(out)}")
    else:
        print("No validation rows exported (check mapping / missing t+1 fares).")


if __name__ == "__main__":
    main()