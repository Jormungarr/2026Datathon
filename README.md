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

# Evaluation

- **RMSE** (Root Mean Squared Error)  
- **MAE** (Mean Absolute Error)  
- Cross-validation performance  

---
