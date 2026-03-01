import os
import json
import numpy as np
from preprocess.data_utils import import_unit_removed_dataset
from pygam import GammaGAM, s
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from preprocess.data_utils import make_test_and_stratified_folds, import_unit_removed_dataset
from sklearn.model_selection import StratifiedKFold
import numpy as np
from preprocess.visualization_utils import plot_gam_terms, plot_gam_summary_dashboard, plot_gam_feature_importance

# Ensure year column exists
N_COMPONENTS =  3
# Load and prepare data
feature_names = ['passengers', 'nsmiles', 'rl_pax_str', 'tot_pax_str', 'large_ms', 'lf_ms']
pca_feature_names = [f'PC{i+1}' for i in range(N_COMPONENTS)]

X_test, y_test, folds, X_all, y_all, df_test, df_rest = make_test_and_stratified_folds(
    feature_cols=feature_names, 
    import_fn=import_unit_removed_dataset
)

# Hyperparameter tuning
mean_scores = []
for k in [15, 20, 25]:
    scores = []
    for fold in folds:
        X_train, y_train = fold["train"]
        X_val, y_val = fold["val"]
        
        # Fit scaler and PCA on training data only
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        pca = PCA(n_components=N_COMPONENTS)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_val_pca = pca.transform(X_val_scaled)
        
        # GAM with spline terms for each PC
        gam = GammaGAM(
            s(0, n_splines=k) + s(1, n_splines=k) + s(2, n_splines=k)
        ).gridsearch(X=X_train_pca, y=y_train)
        
        y_pred = gam.predict(X_val_pca)
        rmse = ((y_val - y_pred) ** 2).mean() ** 0.5
        scores.append(rmse)
    mean_scores.append((k, sum(scores) / len(scores)))

best_k = min(mean_scores, key=lambda x: x[1])[0]
best_rmse = min(mean_scores, key=lambda x: x[1])[1]
print(f"Best n_splines={best_k} with average RMSE={float(best_rmse):.4f}")

# from full prediction
# Fit final model on full training set
scaler = StandardScaler().fit(X_all)
X_all_scaled = scaler.transform(X_all)
X_test_scaled = scaler.transform(X_test)
pca = PCA()
X_all_pca = pca.fit_transform(X_all_scaled)
X_test_pca = pca.transform(X_test_scaled)
X_all_gam = X_all_pca[:, :3]
X_test_gam = X_test_pca[:, :3]

gam = GammaGAM(
    s(0, n_splines=best_k) + s(1, n_splines=best_k) + s(2, n_splines=best_k)# + 
    # s(3, n_splines=best_k) + s(4, n_splines=best_k) + s(5, n_splines=best_k)
).gridsearch(X=X_all_gam, y=y_all)

print("\n" + "="*80)
print("GAM MODEL SUMMARY")
print("="*80 + "\n")
gam.summary()

# Generate predictions
y_test_pred = gam.predict(X_test_gam)
test_rmse = np.sqrt(((y_test - y_test_pred) ** 2).mean())
print(f"\nTest Set RMSE: {test_rmse:.4f}")

# Industrial-grade visualizations
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80 + "\n")

pc_titles = ["PC1", "PC2", "PC3"]
# titles=pc_titles

# 1. Partial dependence plots for all terms
plot_gam_terms(
    gam, 
    titles=pc_titles,
    figsize=(14, 5),
    save_path="./results/gam_pca_partial_dependence.png"
)

# 2. Comprehensive diagnostic dashboard
plot_gam_summary_dashboard(
    gam, 
    X_test_gam, 
    y_test, 
    y_test_pred,
    titles=pc_titles,
    figsize=(16, 12),
    save_path="./results/gam_pca_summary_dashboard.png"
)

# 3. Feature importance (EDOF)
plot_gam_feature_importance(
    gam,
    figsize=(10, 6),
    save_path="./results/gam_pcafeature_importance.png"
)

print("\n✓ All visualizations generated successfully!")