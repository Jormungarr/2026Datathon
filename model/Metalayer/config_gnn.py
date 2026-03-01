# model/config_gnn.py

DATA_PATH = "./data/airline_ticket_dataset.xlsx"
MAKE_UNDIRECTED = True
VAL_RATIO = 0.2
SEED = 42

LEAKY_COLS = ["fare_lg", "fare_low"]

# 强烈建议：默认不放任何看起来与票价聚合有关的列（如 *Fared*）
EDGE_NUM_COLS = [
    "nsmiles",
    "passengers",
    "large_ms",
    "lf_ms",
    "city1_pax_strength",
    "city2_pax_strength",
    "rl_pax_str",
    "tot_pax_str",
]

REQUIRED_COLS = ["Year", "quarter", "citymarketid_1", "citymarketid_2", "fare", "passengers"]

# 模型超参
HIDDEN = 64
N_LAYERS = 2
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 50

CKPT_PATH = "best_model.pt"