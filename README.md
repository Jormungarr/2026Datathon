# 2026 Datathon  
## Airline Fare Market Structure Analysis  

# Overview

This project investigates structural determinants of airfare variation in the U.S. domestic airline market. While route distance and passenger volume are intuitive drivers of ticket prices, airfare dynamics are also shaped by deeper structural forces such as market concentration, hub dominance, and competitive intensity.

We use the U.S. Department of Transportation Domestic Airline Consumer Airfare dataset (2021–2025 Q2), we analyze how demand, competition, and network structure interact to influence route-level pricing.

# Problem Setting

- **Task:** Route-level airfare prediction  
- **Target variable:** Average fare  
- **Time horizon:** 2021–2025 Q2  
- **Market scope:** U.S. domestic city-pair routes  

We aim to understand:

- Which structural factors explain fare differences?
- Do nonlinear effects exist in market concentration?
- Does network structure improve predictive performance?

---

# Methodology

## 1. Exploratory Data Analysis (EDA)

- Fare distribution analysis  
- Competition and market share structure  
- Construction of city-level demand indicators (e.g., *City Strength*)

## 2. Principal Component Analysis (PCA)

To reduce multicollinearity and summarize market structure:

- **PC1:** Overall market scale & competitiveness  
- **PC2:** Route structure & hub dominance  
- **PC3:** Traffic intensity & asymmetry  

The first three components explain approximately **74% of total variance**.

## 3. Modeling Approaches

### Lasso Regression (Baseline)

- Handles multicollinearity  
- Performs variable selection  

### Generalized Additive Model (GAM)

- Captures nonlinear relationships  
- Maintains interpretability  

### Graph Neural Network (GNN)

- Models airline markets as a graph  
- Incorporates network topology  
- Captures structural dependencies between cities  

---

# Results

### Dataset

| Split | Samples | Ratio |
|:------|--------:|------:|
| Train | ~12,582 | 90%   |
| Test  | ~1,398  | 10%   |
| **Total** | **~13,980** | **100%** |

All models are evaluated on held-out test data (10% random split) using **Root Mean Squared Error (RMSE)**.

### Model Comparison

| Model | Features | Test RMSE |
|:------|:---------|----------:|
| **Lasso Regression (Baseline) 🥇** | **Raw** | **35.23** |
| *GAM (Raw + PCA-Guided Interactions) 🥈* | *Raw + PCA-Guided* | *36.33* |
| GAM (Raw + PCA-Guided Interactions + Quarter) | Raw + PCA-Guided + Quarter | 36.35 |
| GAM (Raw + Year + Quarter) | Raw + Year + Quarter | 36.73 |
| GAM (Raw + Year) | Raw + Year | 36.74 |
| GAM (Raw) | Raw | 36.75 |
| GAM (Raw + Quarter) | Raw + Quarter | 37.88 |
| GNN (MetaLayer) | Graph-Structured | 45.01 |
| GAM (PCA + Quarter) | PCA | 49.10 |
| GAM (PCA + Year) | PCA | 49.11 |
| GAM (PCA + Quarter + Year) | PCA | 49.11 |
| GAM (PCA) | PCA | 49.11 |

> **Key Findings:**
> - GAM with PCA-guided interaction terms achieves the best GAM performance (RMSE = 36.33), suggesting that PCA-informed feature engineering captures meaningful nonlinear market structure.


---

# Getting Started

### Prerequisites

- Python ≥ 3.9
- [Conda](https://docs.conda.io/) (recommended)

### Installation

```bash
git clone https://github.com/Jormungarr/2026Datathon.git
cd 2026Datathon
pip install -e .
```

### Project Structure

```
├── data/                # Raw data (flight edges, geocoded cities)
├── preprocess/          # Data loading, filtering, and feature engineering
├── eda/                 # Exploratory data analysis scripts and notebooks
├── pca/                 # PCA analysis notebooks
├── model/               # Model implementations
│   ├── baseline_lasso.py
│   ├── gam/             # GAM variants (raw, PCA, interaction terms)
│   └── Metalayer/       # Graph Neural Network (MetaLayer GNN)
└── results/             # Predictions, dashboards, and visualizations
```

---
