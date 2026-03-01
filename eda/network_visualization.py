import os
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from preprocess.data_utils import import_orginal_dataset

OUT_DIR = "./results/city_graph_with_weight"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# A) 工具函数：城市名清洗
# =========================
def clean_city_name(s: str) -> str:
    if pd.isna(s):
        return s
    s = str(s).strip()
    # 去掉 "(Metropolitan Area)" 等括号说明
    s = re.sub(r"\s*\(.*?\)\s*$", "", s)
    # 规范逗号后空格
    s = re.sub(r"\s*,\s*", ", ", s)
    return s

# =========================
# B) 读数据 & 必要列清洗
# =========================
df = import_orginal_dataset()

required = ["Year", "quarter", "citymarketid_1", "citymarketid_2", "passengers"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# 注意：不要全表 dropna，会删太多。只对建图必需列 dropna
data = df.dropna(subset=required).copy()

data["Year"] = data["Year"].astype(int)
data["quarter"] = data["quarter"].astype(int)
data["citymarketid_1"] = data["citymarketid_1"].astype(int)
data["citymarketid_2"] = data["citymarketid_2"].astype(int)
data["passengers"] = pd.to_numeric(data["passengers"], errors="coerce")
data = data.dropna(subset=["passengers"])
data["passengers"] = data["passengers"].astype(float)

opt_cols = {
    "fare": "fare",
    "nsmiles": "nsmiles",
    "lf_ms": "lf_ms",
    "large_ms": "large_ms",
    "carrier_lg": "carrier_lg",
    "fare_lg": "fare_lg",
    "carrier_low": "carrier_low",
    "fare_low": "fare_low",
    "TotalFaredPax_city1": "TotalFaredPax_city1",
    "TotalFaredPax_city2": "TotalFaredPax_city2",
    "TotalPerLFMkts_city1": "TotalPerLFMkts_city1",
    "TotalPerLFMkts_city2": "TotalPerLFMkts_city2",
    "TotalPerPrem_city1": "TotalPerPrem_city1",
    "TotalPerPrem_city2": "TotalPerPrem_city2",
    "city1": "city1",
    "city2": "city2",
}
present = {k: v for k, v in opt_cols.items() if k in data.columns}

print("Data columns used (optional present):", list(present.keys()))
print("cleaned data shape:", data.shape)

# =========================
# C) citymarketid -> city_clean 映射（从 city1/city2）
# =========================
if "city1" not in df.columns or "city2" not in df.columns:
    raise ValueError("Your dataset has no city1/city2 columns; cannot map citymarketid to geocoded city names.")

id_city_map = []
tmp = df[["citymarketid_1", "city1"]].dropna().copy()
tmp["city_clean"] = tmp["city1"].map(clean_city_name)
id_city_map.append(tmp[["citymarketid_1", "city_clean"]].rename(columns={"citymarketid_1": "citymarketid"}))

tmp = df[["citymarketid_2", "city2"]].dropna().copy()
tmp["city_clean"] = tmp["city2"].map(clean_city_name)
id_city_map.append(tmp[["citymarketid_2", "city_clean"]].rename(columns={"citymarketid_2": "citymarketid"}))

id_city = pd.concat(id_city_map, ignore_index=True).drop_duplicates()
id_city = (
    id_city.groupby(["citymarketid", "city_clean"])
    .size()
    .reset_index(name="cnt")
    .sort_values(["citymarketid", "cnt"], ascending=[True, False])
)
id_city = id_city.drop_duplicates("citymarketid")[["citymarketid", "city_clean"]]

# =========================
# D) 读 geocoded_cities.csv 并 merge 经纬度
# =========================
GEO_PATH = "./data/geocoded_cities.csv"
geo = pd.read_csv(GEO_PATH)
geo["city_clean"] = geo["city_clean"].map(clean_city_name)

city_geo = id_city.merge(geo[["city_clean", "lat", "lon"]], on="city_clean", how="left")
miss_geo = city_geo["lat"].isna().sum()
if miss_geo > 0:
    print(f"[WARN] {miss_geo} citymarketid cannot be matched to lat/lon via city_clean.")

# =========================
# E) 构建 edges_by_period（按季度聚合边权重）
# =========================
edge_keys = ["Year", "quarter", "citymarketid_1", "citymarketid_2"]
agg_dict = {"passengers": "sum"}

for col in ["fare", "nsmiles", "fare_lg", "fare_low", "lf_ms", "large_ms"]:
    if col in df.columns:
        agg_dict[col] = "mean"

def mode_or_nan(x):
    x = x.dropna()
    if len(x) == 0:
        return np.nan
    return x.value_counts().index[0]

for col in ["carrier_lg", "carrier_low"]:
    if col in df.columns:
        agg_dict[col] = mode_or_nan

edges = data.groupby(edge_keys, as_index=False).agg(agg_dict)

# 乘客加权 fare（若存在）
if "fare" in df.columns:
    t = data[edge_keys + ["passengers", "fare"]].dropna()
    t["pf"] = t["passengers"] * t["fare"]
    w = t.groupby(edge_keys, as_index=False).agg(pf=("pf", "sum"), p=("passengers", "sum"))
    w["fare_pax_wavg"] = w["pf"] / w["p"].replace(0, np.nan)
    edges = edges.merge(w[edge_keys + ["fare_pax_wavg"]], on=edge_keys, how="left")

edges = edges.merge(
    city_geo.rename(columns={"citymarketid": "citymarketid_1", "lat": "lat1", "lon": "lon1", "city_clean": "city1_clean"}),
    on="citymarketid_1",
    how="left",
)
edges = edges.merge(
    city_geo.rename(columns={"citymarketid": "citymarketid_2", "lat": "lat2", "lon": "lon2", "city_clean": "city2_clean"}),
    on="citymarketid_2",
    how="left",
)

edges_path = os.path.join(OUT_DIR, "edges_by_period.csv")
edges.to_csv(edges_path, index=False)
print("Saved:", edges_path)

# =========================
# F) 构建 nodes_by_period（节点权重：strength/degree/HHI + 城市宏观变量）
# =========================
left = edges[["Year", "quarter", "citymarketid_1", "passengers"]].rename(columns={"citymarketid_1": "citymarketid"})
right = edges[["Year", "quarter", "citymarketid_2", "passengers"]].rename(columns={"citymarketid_2": "citymarketid"})
inc = pd.concat([left, right], ignore_index=True)

node = inc.groupby(["Year", "quarter", "citymarketid"], as_index=False).agg(
    strength_pax=("passengers", "sum"),
    degree=("passengers", "size"),
)

inc2 = inc.copy()
inc2["share"] = inc2["passengers"] / inc2.groupby(["Year", "quarter", "citymarketid"])["passengers"].transform("sum").replace(0, np.nan)
inc2["share2"] = inc2["share"] ** 2
hhi = inc2.groupby(["Year", "quarter", "citymarketid"], as_index=False).agg(HHI_neighbors=("share2", "sum"))
node = node.merge(hhi, on=["Year", "quarter", "citymarketid"], how="left")

city_attr_cols_1 = [c for c in ["TotalFaredPax_city1", "TotalPerLFMkts_city1", "TotalPerPrem_city1"] if c in df.columns]
city_attr_cols_2 = [c for c in ["TotalFaredPax_city2", "TotalPerLFMkts_city2", "TotalPerPrem_city2"] if c in df.columns]

attrs = []
if city_attr_cols_1:
    t1 = data[["Year", "quarter", "citymarketid_1"] + city_attr_cols_1].copy().rename(columns={"citymarketid_1": "citymarketid"})
    t1 = t1.rename(
        columns={
            "TotalFaredPax_city1": "TotalFaredPax_city",
            "TotalPerLFMkts_city1": "TotalPerLFMkts_city",
            "TotalPerPrem_city1": "TotalPerPrem_city",
        }
    )
    attrs.append(t1)

if city_attr_cols_2:
    t2 = data[["Year", "quarter", "citymarketid_2"] + city_attr_cols_2].copy().rename(columns={"citymarketid_2": "citymarketid"})
    t2 = t2.rename(
        columns={
            "TotalFaredPax_city2": "TotalFaredPax_city",
            "TotalPerLFMkts_city2": "TotalPerLFMkts_city",
            "TotalPerPrem_city2": "TotalPerPrem_city",
        }
    )
    attrs.append(t2)

if attrs:
    city_attr = pd.concat(attrs, ignore_index=True).groupby(["Year", "quarter", "citymarketid"], as_index=False).mean(numeric_only=True)
    node = node.merge(city_attr, on=["Year", "quarter", "citymarketid"], how="left")

node = node.merge(city_geo, on="citymarketid", how="left")

nodes_path = os.path.join(OUT_DIR, "nodes_by_period.csv")
node.to_csv(nodes_path, index=False)
print("Saved:", nodes_path)

# =========================
# G) 地理动画图：按季度（全航司 / 指定航司）
# =========================
def make_geo_animation(edges_df: pd.DataFrame, nodes_df: pd.DataFrame, out_html: str, carrier_filter=None):
    dE = edges_df.copy()
    dN = nodes_df.copy()

    if carrier_filter is not None:
        if "carrier_lg" not in dE.columns:
            raise ValueError("carrier_lg not in edges; cannot filter by carrier.")
        dE = dE[dE["carrier_lg"] == carrier_filter].copy()

    dE = dE.dropna(subset=["lat1", "lon1", "lat2", "lon2", "passengers"])
    dN = dN.dropna(subset=["lat", "lon", "strength_pax"])

    frames_keys = dE[["Year", "quarter"]].drop_duplicates().sort_values(["Year", "quarter"]).to_records(index=False)
    if len(frames_keys) == 0:
        raise ValueError("No frames available after filtering. Check missing geocodes or carrier filter.")
    y0, q0 = frames_keys[0]

    def scale_sizes(x, smin, smax):
        x = np.asarray(x, dtype=float)
        if len(x) == 0:
            return x
        lo, hi = np.nanmin(x), np.nanmax(x)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
            return np.full_like(x, (smin + smax) / 2, dtype=float)
        return smin + (x - lo) * (smax - smin) / (hi - lo)

    # ---------- Static legend traces (kept in every frame) ----------
    # Node size legend (visual reference only; actual size is scaled per frame in [6,28])
    node_size_legend = [
        go.Scattergeo(
            lon=[-160], lat=[10],  # off-map-ish for USA scope; still OK for legend
            mode="markers",
            marker=dict(size=s, color="gray", opacity=0.85),
            name=lab,
            showlegend=True,
            hoverinfo="skip",
            visible="legendonly",
        )
        for s, lab in [
            (8, "City strength: small"),
            (16, "City strength: medium"),
            (26, "City strength: large"),
        ]
    ]

    # Edge color legend: low/med/high passengers tertiles within each frame
    edge_colors = ["#a6bddb", "#3690c0", "#034e7b"]  # low -> high
    edge_color_legend = [
        go.Scattergeo(
            lon=[-160], lat=[10],
            mode="lines",
            line=dict(width=3, color=c),
            name=lab,
            showlegend=True,
            hoverinfo="skip",
            visible="legendonly",
        )
        for c, lab in zip(
            edge_colors,
            ["Route passengers: low (frame tertile)", "Route passengers: mid (frame tertile)", "Route passengers: high (frame tertile)"],
        )
    ]

    meaning_note = go.Scattergeo(
        lon=[-160], lat=[10],
        mode="markers",
        marker=dict(size=10, color="white", opacity=0.0),
        name="Encoding: node size ∝ city strength_pax; edge width ∝ passengers",
        showlegend=True,
        hoverinfo="skip",
        visible="legendonly",
    )

    static_legend_traces = node_size_legend + edge_color_legend + [meaning_note]

    def build_traces(y, q):
        Ee = dE[(dE["Year"] == y) & (dE["quarter"] == q)]
        Nn = dN[(dN["Year"] == y) & (dN["quarter"] == q)]

        # ---------- edges: bin by passenger tertiles, width per-bin, color per-bin ----------
        edge_traces = []
        if len(Ee) > 0:
            bins = pd.qcut(Ee["passengers"], q=min(3, Ee["passengers"].nunique()), duplicates="drop")
            cats = list(bins.cat.categories)

            for idx, b in enumerate(cats):
                Ebin = Ee[bins == b]
                if len(Ebin) == 0:
                    continue

                # width based on mean passenger (scaled)
                width_b = float(np.clip(np.nanmean(scale_sizes(Ebin["passengers"].values, 0.8, 4.0)), 0.8, 4.0))

                lons_b, lats_b = [], []
                for lon1, lat1, lon2, lat2 in zip(Ebin["lon1"], Ebin["lat1"], Ebin["lon2"], Ebin["lat2"]):
                    lons_b += [lon1, lon2, None]
                    lats_b += [lat1, lat2, None]

                color_b = edge_colors[min(idx, len(edge_colors) - 1)]
                edge_traces.append(
                    go.Scattergeo(
                        lon=lons_b,
                        lat=lats_b,
                        mode="lines",
                        line=dict(width=width_b, color=color_b),
                        opacity=0.55,
                        hoverinfo="skip",
                        showlegend=False,  # legend handled by static traces
                        name="edges",
                    )
                )

        # ---------- nodes: size by strength_pax ----------
        ns = scale_sizes(Nn["strength_pax"].values, 6, 28)
        node_trace = go.Scattergeo(
            lon=Nn["lon"],
            lat=Nn["lat"],
            mode="markers",
            marker=dict(size=ns, opacity=0.85, color="black"),
            text=(
                "citymarketid=" + Nn["citymarketid"].astype(str)
                + "<br>strength_pax=" + Nn["strength_pax"].round(1).astype(str)
                + "<br>degree=" + Nn["degree"].astype(str)
                + "<br>HHI=" + Nn["HHI_neighbors"].round(3).astype(str)
            ),
            hoverinfo="text",
            name="nodes",
            showlegend=False,  # legend handled by static traces
        )

        # IMPORTANT: include static legend traces in every frame because we animate with redraw=True
        return edge_traces + [node_trace] + static_legend_traces

    base_traces = build_traces(y0, q0)
    title0 = (
        f"Geo Network | "
        f"{'All carriers' if carrier_filter is None else f'carrier={carrier_filter}'} | "
        f"{y0} Q{q0}"
    )
    fig = go.Figure(data=base_traces)

    frames = []
    for y, q in frames_keys:
        fr_traces = build_traces(y, q)
        frame_title = (
            f"Geo Network | "
            f"{'All carriers' if carrier_filter is None else f'carrier={carrier_filter}'} | "
            f"{y} Q{q}"
        )
        frames.append(
            go.Frame(
                data=fr_traces,
                name=f"{y}Q{q}",
                layout=go.Layout(title=frame_title),
            )
        )
    fig.frames = frames

    fig.update_layout(
        title=title0,
        geo=dict(scope="usa", projection_type="albers usa", showland=True),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=800, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=200),
                            ),
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                steps=[
                    dict(
                        method="animate",
                        args=[[f"{y}Q{q}"], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
                        label=f"{y} Q{q}",
                    )
                    for y, q in frames_keys
                ],
                active=0,
            )
        ],
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.9,
            xanchor="left",
            x=0.9,     # 超出右边界
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="green",
            borderwidth=1
        ),
        margin=dict(l=10, r=200, t=60, b=10)
    )

    fig.write_html(out_html)
    print("Saved:", out_html)

# 全航司动画
make_geo_animation(edges, node, os.path.join(OUT_DIR, "geo_network_allcarriers.html"), carrier_filter=None)

# 按航司分别动画（用 carrier_lg）
if "carrier_lg" in edges.columns:
    carriers = sorted([c for c in edges["carrier_lg"].dropna().unique()])
    for c in carriers:
        make_geo_animation(edges, node, os.path.join(OUT_DIR, f"geo_network_by_carrier_{c}.html"), carrier_filter=c)
else:
    print("[INFO] carrier_lg not found; skip per-carrier plots.")