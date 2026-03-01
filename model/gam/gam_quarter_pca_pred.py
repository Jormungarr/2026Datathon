import os
import json
import numpy as np
from preprocess.data_utils import import_unit_removed_dataset
from pygam import GammaGAM, s, f
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
EXPERIMENT_PREFIX = "quarter_pca_3comp"
N_COMPONENTS = 3

# Define feature columns
feature_names = ['passengers', 'nsmiles', 'rl_pax_str', 'tot_pax_str', 'large_ms', 'lf_ms']
time_features = ['quarter']
pca_feature_names = [f'PC{i+1}' for i in range(N_COMPONENTS)]
all_feature_names = pca_feature_names + time_features  # PC1, PC2, PC3, quarter

# Load and prepare data
X_test, y_test, folds, X_all, y_all, df_test, df_rest = make_test_and_stratified_folds(
    feature_cols=feature_names + time_features,
    import_fn=import_unit_removed_dataset,
    categorical_encode_cols=time_features
)

n_numeric = len(feature_names)

# Hyperparameter tuning
mean_scores = []
for k in [15, 20, 25]:
    scores = []
    for fold in folds:
        X_train, y_train = fold["train"]
        X_val, y_val = fold["val"]

        # Split numeric and categorical features
        X_train_num = X_train[:, :n_numeric]
        X_val_num = X_val[:, :n_numeric]
        X_train_cat = X_train[:, n_numeric:n_numeric+1]  # Quarter column only
        X_val_cat = X_val[:, n_numeric:n_numeric+1]

        # Fit scaler and PCA on numeric features only
        scaler = StandardScaler().fit(X_train_num)
        X_train_num_scaled = scaler.transform(X_train_num)
        X_val_num_scaled = scaler.transform(X_val_num)

        pca = PCA(n_components=N_COMPONENTS)
        X_train_pca = pca.fit_transform(X_train_num_scaled)
        X_val_pca = pca.transform(X_val_num_scaled)

        # Combine PCA components + categorical quarter
        X_train_gam = np.hstack([X_train_pca, X_train_cat])
        X_val_gam = np.hstack([X_val_pca, X_val_cat])

        # GAM: spline terms for PCs + categorical term for quarter
        gam = GammaGAM(
            s(0, n_splines=k) + s(1, n_splines=k) + s(2, n_splines=k) + f(3)
        ).gridsearch(X=X_train_gam, y=y_train)

        y_pred = gam.predict(X_val_gam)
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
    "original_features": feature_names,
    "categorical_features": time_features
}

cv_json_path = os.path.join(results_dir, f"{EXPERIMENT_PREFIX}_cv_results.json")
with open(cv_json_path, "w") as json_file:
    json.dump(cv_results, json_file, indent=2)
print(f"✅ Cross-validation results saved to: {cv_json_path}")

# Fit final model on full training set
X_all_num = X_all[:, :n_numeric]
X_test_num = X_test[:, :n_numeric]
X_all_cat = X_all[:, n_numeric:n_numeric+1]   # Quarter column
X_test_cat = X_test[:, n_numeric:n_numeric+1]

scaler = StandardScaler().fit(X_all_num)
X_all_num_scaled = scaler.transform(X_all_num)
X_test_num_scaled = scaler.transform(X_test_num)

pca = PCA(n_components=N_COMPONENTS)
X_all_pca = pca.fit_transform(X_all_num_scaled)
X_test_pca = pca.transform(X_test_num_scaled)

# Combine PCA components + categorical quarter
X_all_gam = np.hstack([X_all_pca, X_all_cat])
X_test_gam = np.hstack([X_test_pca, X_test_cat])

# Save PCA info to JSON
pca_info = {
    "n_components": N_COMPONENTS,
    "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    "cumulative_variance_ratio": np.cumsum(pca.explained_variance_ratio_).tolist(),
    "original_features": feature_names,
    "categorical_features": time_features,
    "loadings": {
        f"PC{i+1}": {feat: float(pca.components_[i, j]) for j, feat in enumerate(feature_names)}
        for i in range(N_COMPONENTS)
    }
}

pca_json_path = os.path.join(results_dir, f"{EXPERIMENT_PREFIX}_pca_info.json")
with open(pca_json_path, "w") as json_file:
    json.dump(pca_info, json_file, indent=2)
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
    s(0, n_splines=best_k) + s(1, n_splines=best_k) + s(2, n_splines=best_k) + f(3)
).gridsearch(X=X_all_gam, y=y_all)

print("\n" + "="*80)
print("GAM MODEL SUMMARY")
print("="*80 + "\n")
gam.summary()

# Generate predictions for BOTH training and test sets
y_train_pred = gam.predict(X_all_gam)
y_test_pred = gam.predict(X_test_gam)

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

# 1. Partial dependence plots for PC1, PC2, PC3, quarter
plot_gam_terms_plotly(
    gam, 
    titles=all_feature_names,
    plot_title=f"GAM Partial Dependence - PCA + Quarter ({N_COMPONENTS} components, n_splines={best_k})",
    save_path=get_save_path("partial_dependence")
)

# 2. Combined diagnostic dashboard (train + test)
plot_gam_combined_dashboard_plotly(
    gam,
    X_all_gam, y_all, y_train_pred,       # Training data
    X_test_gam, y_test, y_test_pred,       # Test data
    titles=all_feature_names,
    plot_title=f"GAM PCA + Quarter Combined Analysis - Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f}",
    save_path=get_save_path("combined_dashboard")
)

# 3. Feature importance (model-based)
plot_gam_feature_importance_plotly(
    gam,
    feature_names=all_feature_names,
    plot_title=f"GAM PCA + Quarter Feature Importance (EDOF) - {N_COMPONENTS} Components",
    save_path=get_save_path("feature_importance")
)

print("\n" + "="*80)
print("VISUALIZATION SUMMARY")
print("="*80)
print(f"✅ PCA + Quarter Combined Analysis Complete!")
print(f"   📁 Saved to: {results_dir}")
print(f"   📊 Training RMSE: {train_rmse:.4f}")
print(f"   📊 Test RMSE: {test_rmse:.4f}")
print(f"   📈 Training Samples: {len(y_all):,}")
print(f"   📈 Test Samples: {len(y_test):,}")
print(f"   🔬 PCA Components: {N_COMPONENTS} ({np.sum(pca.explained_variance_ratio_)*100:.1f}% variance)")
print(f"   🗓️  Categorical: {time_features}")
print(f"   📏 Generalization Ratio: {test_rmse/train_rmse:.2f}" + 
      (" ⚠️  (>1.2 indicates overfitting)" if test_rmse/train_rmse > 1.2 else " ✅ (good generalization)"))
