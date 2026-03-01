import os
from preprocess.data_utils import import_unit_removed_dataset
from pygam import GammaGAM, s
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from preprocess.data_utils import make_test_and_stratified_folds, import_unit_removed_dataset
import numpy as np
from sklearn.model_selection import StratifiedKFold
from preprocess.visualization_utils import (
    plot_gam_terms_plotly, 
    plot_gam_combined_dashboard_plotly,
    plot_gam_feature_importance_plotly
)
import json 

# Configuration variables for paths and naming
RESULTS_BASE_DIR = "./results/gam/raw/"
EXPERIMENT_PREFIX = "no_quarter"  # Change this for different experiments
FIGURE_FORMATS = ".png"  # Could be .pdf, .svg, etc.

# Load and prepare data (NO categorical variables)
feature_names = ['passengers', 'nsmiles', 'rl_pax_str', 'tot_pax_str', 'large_ms', 'lf_ms']
all_feature_names = feature_names  # No time features added

# Load and prepare data (NO categorical encoding)
X_test, y_test, folds, X_all, y_all, df_test, df_rest = make_test_and_stratified_folds(
    feature_cols=feature_names, 
    import_fn=import_unit_removed_dataset
    # NO categorical_encode_cols parameter
)

# Hyperparameter tuning
mean_scores = []
for k in [15, 20, 25]:
    scores = []
    for fold in folds:
        X_train, y_train = fold["train"]
        X_val, y_val = fold["val"]
        
        # Standard scaling for ALL continuous features
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # GAM with only spline terms (NO categorical f() terms)
        gam = GammaGAM(
            s(0, n_splines=k) + s(1, n_splines=k) + s(2, n_splines=k) + 
            s(3, n_splines=k) + s(4, n_splines=k) + s(5, n_splines=k)
        ).gridsearch(X=X_train_scaled, y=y_train)
        
        y_pred = gam.predict(X_val_scaled)
        # Calculate RMSE
        rmse = ((y_val - y_pred) ** 2).mean() ** 0.5
        scores.append(rmse)
    mean_scores.append((k, sum(scores) / len(scores)))

best_k = min(mean_scores, key=lambda x: x[1])[0]
best_rmse = min(mean_scores, key=lambda x: x[1])[1]
print(f"Best n_splines={best_k} with average RMSE={float(best_rmse):.4f}")

# Prepare cross-validation results for saving
cv_results = {
    "cv_scores": [{"n_splines": k, "mean_rmse": float(rmse)} for k, rmse in mean_scores],
    "selected_n_splines": int(best_k),
    "selected_mean_rmse": float(best_rmse)
}

# Save as JSON
cv_json_path = os.path.join(RESULTS_BASE_DIR, f"{EXPERIMENT_PREFIX}_cv_results.json")
with open(cv_json_path, "w") as f:
    json.dump(cv_results, f, indent=2)
print(f"✅ Cross-validation results saved to: {cv_json_path}")

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

# Generate predictions for BOTH training and test sets
y_train_pred = gam.predict(X_all_scaled)  # Training predictions
y_test_pred = gam.predict(X_test_scaled)   # Test predictions

# Calculate performance metrics
train_rmse = np.sqrt(((y_all - y_train_pred) ** 2).mean())
test_rmse = np.sqrt(((y_test - y_test_pred) ** 2).mean())

print(f"\nTraining Set RMSE: {train_rmse:.4f}")
print(f"Test Set RMSE: {test_rmse:.4f}")
print(f"Overfitting Check: Test RMSE / Train RMSE = {test_rmse/train_rmse:.2f}")

# Single directory for all results
results_dir = os.path.join(RESULTS_BASE_DIR, "combined")
os.makedirs(results_dir, exist_ok=True)

def get_save_path(diagram_type):
    """Generate standardized save path for diagrams."""
    return os.path.join(results_dir, f"{EXPERIMENT_PREFIX}_{diagram_type}")

print("\n" + "="*80)
print("GENERATING COMBINED VISUALIZATIONS")
print("="*80 + "\n")

# 1. Partial dependence plots (model-based, same for both)
plot_gam_terms_plotly(
    gam, 
    titles=all_feature_names,
    plot_title=f"GAM Partial Dependence Analysis - No Quarter (n_splines={best_k})",
    save_path=get_save_path("partial_dependence")
)

# 2. Combined diagnostic dashboard (train + test)
plot_gam_combined_dashboard_plotly(
    gam,
    X_all_scaled, y_all, y_train_pred,      # Training data
    X_test_scaled, y_test, y_test_pred,     # Test data
    titles=all_feature_names,
    plot_title=f"GAM Combined Analysis - No Quarter - Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f}",
    save_path=get_save_path("combined_dashboard")
)

# 3. Feature importance (model-based)
plot_gam_feature_importance_plotly(
    gam,
    feature_names=all_feature_names,
    plot_title=f"GAM Feature Importance Analysis - No Quarter (EDOF)",
    save_path=get_save_path("feature_importance")
)

print("\n" + "="*80)
print("VISUALIZATION SUMMARY")
print("="*80)
print(f"✅ Combined Analysis Complete!")
print(f"   📁 Saved to: {results_dir}")
print(f"   📊 Training RMSE: {train_rmse:.4f}")
print(f"   📊 Test RMSE: {test_rmse:.4f}")
print(f"   📈 Training Samples: {len(y_all):,}")
print(f"   📈 Test Samples: {len(y_test):,}")
print(f"   📏 Generalization Ratio: {test_rmse/train_rmse:.2f}" + 
      (" ⚠️  (>1.2 indicates overfitting)" if test_rmse/train_rmse > 1.2 else " ✅ (good generalization)"))