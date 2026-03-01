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
    plot_gam_summary_dashboard_plotly,
    plot_gam_feature_importance_plotly  # Keep matplotlib for this one
)


# Configuration variables for paths and naming
RESULTS_BASE_DIR = "./results/gam/raw/"
EXPERIMENT_PREFIX = "add_quarter"  # Change this for different experiments
FIGURE_FORMATS = ".png"  # Could be .pdf, .svg, etc.

# Load and prepare data
time_features = ['quarter']
feature_names = ['passengers', 'nsmiles', 'rl_pax_str', 'tot_pax_str', 'large_ms', 'lf_ms']
all_feature_names = feature_names + time_features

# Create complete feature names including time features
all_feature_names = feature_names + time_features
# Result: ['passengers', 'nsmiles', 'rl_pax_str', 'tot_pax_str', 'large_ms', 'lf_ms', 'quarter']

# Then use this in your visualization calls:

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

# Generate predictions
y_test_pred = gam.predict(X_test_scaled)
test_rmse = np.sqrt(((y_test - y_test_pred) ** 2).mean())
print(f"\nTest Set RMSE: {test_rmse:.4f}")

# Industrial-grade visualizations
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80 + "\n")

# Create results directory if it doesn't exist
os.makedirs(RESULTS_BASE_DIR, exist_ok=True)

# Helper function to generate save paths
def get_save_path(diagram_type):
    """Generate standardized save path for diagrams."""
    return os.path.join(RESULTS_BASE_DIR, f"{EXPERIMENT_PREFIX}_{diagram_type}{FIGURE_FORMATS}")

# 1. Interactive partial dependence plots
plot_gam_terms_plotly(
    gam, 
    titles=all_feature_names,
    save_path=get_save_path("partial_dependence")  # Saves as .html
)

# 2. Interactive diagnostic dashboard
plot_gam_summary_dashboard_plotly(
    gam, 
    X_test_scaled, 
    y_test, 
    y_test_pred,
    titles=all_feature_names,
    save_path=get_save_path("summary_dashboard")  # Saves as .html
)
plot_gam_feature_importance_plotly(
    gam,
    feature_names=all_feature_names,
    save_path=get_save_path("feature_importance_interactive")
)
print(f"\n✓ All visualizations generated successfully!")
print(f"📁 Results saved to: {RESULTS_BASE_DIR}")
print(f"🏷️  Experiment prefix: {EXPERIMENT_PREFIX}")