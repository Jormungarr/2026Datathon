import os
from preprocess.data_utils import import_unit_removed_dataset
from pygam import GammaGAM, s, f
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from preprocess.data_utils import make_test_and_stratified_folds, import_unit_removed_dataset
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import StratifiedKFold
from preprocess.visualization_utils import (
    plot_gam_terms_plotly, 
    plot_gam_combined_dashboard_plotly,
    plot_gam_feature_importance_plotly
)


# Configuration variables for paths and naming
RESULTS_BASE_DIR = "./results/gam/raw/"
EXPERIMENT_PREFIX = "add_quarter"  # Change this for different experiments
FIGURE_FORMATS = ".png"  # Could be .pdf, .svg, etc.

# Load and prepare data
time_features = ['quarter']
feature_names = ['passengers', 'nsmiles', 'rl_pax_str', 'tot_pax_str', 'large_ms', 'lf_ms']
all_feature_names = feature_names + time_features

# Load and prepare data
X_test, y_test, folds , X_all, y_all, df_test, df_rest  = make_test_and_stratified_folds(
    feature_cols=feature_names+time_features, 
    import_fn=import_unit_removed_dataset, 
    categorical_encode_cols=time_features
)

# Hyperparameter tuning
mean_scores = []
for k in [15, 20, 25]:
    scores = []
    for fold in folds:
        X_train, y_train = fold["train"]
        X_val, y_val = fold["val"]
        # FIX: Skip 'quarter' column (last column), not first column
        continuous_cols = list(range(len(feature_names)))  # [0, 1, 2, 3, 4, 5]
        quarter_col = len(feature_names)  # 6
        scaler = StandardScaler().fit(X_train[:, continuous_cols])
        X_train_scaled = X_train.copy()
        X_val_scaled = X_val.copy()
        X_train_scaled[:, continuous_cols] = scaler.transform(X_train[:, continuous_cols])
        X_val_scaled[:, continuous_cols] = scaler.transform(X_val[:, continuous_cols])
        # Quarter column (index 6) remains unscaled
        # Fit and evaluate your model here
        gam = GammaGAM(
            s(0, n_splines=k) + s(1, n_splines=k) + s(2, n_splines=k) + s(3, n_splines=k) + s(4, n_splines=k) +
            s(5, n_splines=k) + f(6) 
        ).gridsearch(X=X_train_scaled, y=y_train)
        y_pred = gam.predict(X_val_scaled)
        # Calculate RMSE
        rmse = ((y_val - y_pred) ** 2).mean() ** 0.5
        scores.append(rmse)
    mean_scores.append((k, sum(scores) / len(scores)))

best_k = min(mean_scores, key=lambda x: x[1])[0]
best_rmse = min(mean_scores, key=lambda x: x[1])[1]
print(f"Best n_splines={best_k} with average RMSE={float(best_rmse):.4f}")

# Fit final model on full training set
# FIX: Skip quarter column for scaling
scaler = StandardScaler().fit(X_all[:, :-1])  # Skip last column (quarter)
X_all_scaled = X_all.copy()
X_test_scaled = X_test.copy()
X_all_scaled[:, :-1] = scaler.transform(X_all[:, :-1])     # Scale all except quarter
X_test_scaled[:, :-1] = scaler.transform(X_test[:, :-1])   # Scale all except quarter

gam = GammaGAM(
    s(0, n_splines=best_k) + s(1, n_splines=best_k) + s(2, n_splines=best_k) + 
    s(3, n_splines=best_k) + s(4, n_splines=best_k) + s(5, n_splines=best_k) + f(6)
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
    plot_title=f"GAM Partial Dependence Analysis (n_splines={best_k})",
    save_path=get_save_path("partial_dependence")
)

# 2. Combined diagnostic dashboard (train + test)
plot_gam_combined_dashboard_plotly(
    gam,
    X_all_scaled, y_all, y_train_pred,      # Training data
    X_test_scaled, y_test, y_test_pred,     # Test data
    titles=all_feature_names,
    plot_title=f"GAM Combined Analysis - Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f}",
    save_path=get_save_path("combined_dashboard")
)

# 3. Feature importance (model-based)
plot_gam_feature_importance_plotly(
    gam,
    feature_names=all_feature_names,
    plot_title=f"GAM Feature Importance Analysis (EDOF)",
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