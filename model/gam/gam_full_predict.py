from preprocess.data_utils import import_unit_removed_dataset
from pygam import GammaGAM, s
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from preprocess.data_utils import make_test_and_stratified_folds, import_unit_removed_dataset
from preprocess.visualization_utils import plot_gam_terms, plot_gam_summary_dashboard, plot_gam_feature_importance

# Load and prepare data
feature_names = ['passengers', 'nsmiles', 'rl_pax_str', 'tot_pax_str', 'large_ms', 'lf_ms']
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
        
        # Fit scaler on training data only
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Train GAM
        gam = GammaGAM(
            s(0, n_splines=k) + s(1, n_splines=k) + s(2, n_splines=k) + 
            s(3, n_splines=k) + s(4, n_splines=k) + s(5, n_splines=k)
        ).gridsearch(X=X_train_scaled, y=y_train)
        
        y_pred = gam.predict(X_val_scaled)
        rmse = np.sqrt(((y_val - y_pred) ** 2).mean())
        scores.append(rmse)
    
    mean_rmse = np.mean(scores)
    mean_scores.append((k, mean_rmse))
    print(f"k={k}: RMSE={mean_rmse:.4f}")

best_k = min(mean_scores, key=lambda x: x[1])[0]
best_rmse = min(mean_scores, key=lambda x: x[1])[1]
print(f"\n✓ Best n_splines={best_k} with average RMSE={best_rmse:.4f}\n")

# Fit final model on full training set
scaler = StandardScaler().fit(X_all)
X_all_scaled = scaler.transform(X_all)
X_test_scaled = scaler.transform(X_test)

gam = GammaGAM(
    s(0, n_splines=best_k) + s(1, n_splines=best_k) + s(2, n_splines=best_k) + 
    s(3, n_splines=best_k) + s(4, n_splines=best_k) + s(5, n_splines=best_k)
).gridsearch(X=X_all_scaled, y=y_all)

print("\n" + "="*80)
print("GAM MODEL SUMMARY")
print("="*80 + "\n")
gam.summary()

# Generate predictions
y_test_pred = gam.predict(X_test_scaled)
test_rmse = np.sqrt(((y_test - y_test_pred) ** 2).mean())
print(f"\nTest Set RMSE: {test_rmse:.4f}")

# Industrial-grade visualizations
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80 + "\n")

# 1. Partial dependence plots for all terms
plot_gam_terms(
    gam, 
    titles=feature_names,
    figsize=(14, 5),
    save_path="./results/gam_partial_dependence.png"
)

# 2. Comprehensive diagnostic dashboard
plot_gam_summary_dashboard(
    gam, 
    X_test_scaled, 
    y_test, 
    y_test_pred,
    titles=feature_names,
    figsize=(16, 12),
    save_path="./results/gam_summary_dashboard.png"
)

# 3. Feature importance (EDOF)
plot_gam_feature_importance(
    gam,
    figsize=(10, 6),
    save_path="./results/gam_feature_importance.png"
)

print("\n✓ All visualizations generated successfully!")