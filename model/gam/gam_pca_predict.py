import os
import json
import numpy as np
from preprocess.data_utils import import_unit_removed_dataset
from pygam import GammaGAM, s
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from preprocess.data_utils import make_test_and_stratified_folds
from preprocess.visualization_utils import (
    plot_gam_terms_plotly, 
    plot_gam_combined_dashboard_plotly,
    plot_gam_feature_importance_plotly
)

# Configuration variables for paths and naming
RESULTS_BASE_DIR = "./results/gam/pca/"
EXPERIMENT_PREFIX = "pca_3comp"
N_COMPONENTS = 3

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

# Save cross-validation results as JSON
results_dir = os.path.join(RESULTS_BASE_DIR, "combined")
os.makedirs(results_dir, exist_ok=True)

cv_results = {
    "cv_scores": [{"n_splines": k, "mean_rmse": float(rmse)} for k, rmse in mean_scores],
    "selected_n_splines": int(best_k),
    "selected_mean_rmse": float(best_rmse),
    "n_pca_components": N_COMPONENTS,
    "original_features": feature_names
}

cv_json_path = os.path.join(results_dir, f"{EXPERIMENT_PREFIX}_cv_results.json")
os.makedirs(os.path.dirname(cv_json_path), exist_ok=True)
with open(cv_json_path, "w") as f:
    json.dump(cv_results, f, indent=2)
print(f"✅ Cross-validation results saved to: {cv_json_path}")

# Fit final model on full training set
scaler = StandardScaler().fit(X_all)
X_all_scaled = scaler.transform(X_all)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=N_COMPONENTS)
X_all_pca = pca.fit_transform(X_all_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Save PCA info to JSON
pca_info = {
    "n_components": N_COMPONENTS,
    "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    "cumulative_variance_ratio": np.cumsum(pca.explained_variance_ratio_).tolist(),
    "original_features": feature_names,
    "loadings": {
        f"PC{i+1}": {feat: float(pca.components_[i, j]) for j, feat in enumerate(feature_names)}
        for i in range(N_COMPONENTS)
    }
}

pca_json_path = os.path.join(results_dir, f"{EXPERIMENT_PREFIX}_pca_info.json")
with open(pca_json_path, "w") as f:
    json.dump(pca_info, f, indent=2)
print(f"✅ PCA info saved to: {pca_json_path}")

# Print PCA summary
print("\n" + "="*80)
print("PCA SUMMARY")
print("="*80)
for i in range(N_COMPONENTS):
    print(f"  PC{i+1}: {pca.explained_variance_ratio_[i]*100:.2f}% variance")
print(f"  Total: {np.sum(pca.explained_variance_ratio_)*100:.2f}% variance explained")
print()

# Fit final GAM
gam = GammaGAM(
    s(0, n_splines=best_k) + s(1, n_splines=best_k) + s(2, n_splines=best_k)
).gridsearch(X=X_all_pca, y=y_all)

print("\n" + "="*80)
print("GAM MODEL SUMMARY")
print("="*80 + "\n")
gam.summary()

# Generate predictions for BOTH training and test sets
y_train_pred = gam.predict(X_all_pca)
y_test_pred = gam.predict(X_test_pca)

# Calculate performance metrics
train_rmse = np.sqrt(((y_all - y_train_pred) ** 2).mean())
test_rmse = np.sqrt(((y_test - y_test_pred) ** 2).mean())

print(f"\nTraining Set RMSE: {train_rmse:.4f}")
print(f"Test Set RMSE: {test_rmse:.4f}")
print(f"Overfitting Check: Test RMSE / Train RMSE = {test_rmse/train_rmse:.2f}")

def get_save_path(diagram_type):
    """Generate standardized save path for diagrams."""
    return os.path.join(results_dir, f"{EXPERIMENT_PREFIX}_{diagram_type}")

print("\n" + "="*80)
print("GENERATING COMBINED VISUALIZATIONS")
print("="*80 + "\n")

# 1. Partial dependence plots for PC1, PC2, PC3
plot_gam_terms_plotly(
    gam, 
    titles=pca_feature_names,
    plot_title=f"GAM Partial Dependence - PCA ({N_COMPONENTS} components, n_splines={best_k})",
    save_path=get_save_path("partial_dependence")
)

# 2. Combined diagnostic dashboard (train + test)
plot_gam_combined_dashboard_plotly(
    gam,
    X_all_pca, y_all, y_train_pred,       # Training data
    X_test_pca, y_test, y_test_pred,       # Test data
    titles=pca_feature_names,
    plot_title=f"GAM PCA Combined Analysis - Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f}",
    save_path=get_save_path("combined_dashboard")
)

# 3. Feature importance (model-based)
plot_gam_feature_importance_plotly(
    gam,
    feature_names=pca_feature_names,
    plot_title=f"GAM PCA Feature Importance (EDOF) - {N_COMPONENTS} Components",
    save_path=get_save_path("feature_importance")
)

print("\n" + "="*80)
print("VISUALIZATION SUMMARY")
print("="*80)
print(f"✅ PCA Combined Analysis Complete!")
print(f"   📁 Saved to: {results_dir}")
print(f"   📊 Training RMSE: {train_rmse:.4f}")
print(f"   📊 Test RMSE: {test_rmse:.4f}")
print(f"   📈 Training Samples: {len(y_all):,}")
print(f"   📈 Test Samples: {len(y_test):,}")
print(f"   🔬 PCA Components: {N_COMPONENTS} ({np.sum(pca.explained_variance_ratio_)*100:.1f}% variance)")
print(f"   📏 Generalization Ratio: {test_rmse/train_rmse:.2f}" + 
      (" ⚠️  (>1.2 indicates overfitting)" if test_rmse/train_rmse > 1.2 else " ✅ (good generalization)"))

