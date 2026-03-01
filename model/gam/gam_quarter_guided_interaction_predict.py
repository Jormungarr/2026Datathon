import os
import json
import numpy as np
from preprocess.data_utils import import_unit_removed_dataset
from pygam import GammaGAM, s, f, te
from sklearn.preprocessing import StandardScaler
from preprocess.data_utils import make_test_and_stratified_folds
from preprocess.visualization_utils import (
    plot_gam_terms_plotly, 
    plot_gam_combined_dashboard_plotly,
    plot_gam_feature_importance_plotly
)

# Configuration variables for paths and naming
RESULTS_BASE_DIR = "./results/gam/guided_interaction/"
EXPERIMENT_PREFIX = "add_quarter"

# Feature columns:
#   0: passengers, 1: nsmiles, 2: rl_pax_str, 3: tot_pax_str, 4: large_ms, 5: lf_ms, 6: quarter
feature_names = ['passengers', 'nsmiles', 'rl_pax_str', 'tot_pax_str', 'large_ms', 'lf_ms']
time_features = ['quarter']
all_feature_names = feature_names + time_features

# Load and prepare data
X_test, y_test, folds, X_all, y_all, df_test, df_rest = make_test_and_stratified_folds(
    feature_cols=feature_names + time_features,
    import_fn=import_unit_removed_dataset,
    categorical_encode_cols=time_features
)

n_continuous = len(feature_names)  # 6 continuous columns (indices 0-5)
quarter_col = n_continuous          # quarter at index 6

# GAM formula builder: 6 spline terms + 2 tensor product interactions + categorical quarter
# Interaction 1: te(2, 3) = rl_pax_str × tot_pax_str
# Interaction 2: te(4, 5) = large_ms × lf_ms
def build_gam_formula(k):
    return (
        s(0, n_splines=k) + s(1, n_splines=k) + s(2, n_splines=k) +
        s(3, n_splines=k) + s(4, n_splines=k) + s(5, n_splines=k) +
        te(2, 3, n_splines=k) + te(4, 5, n_splines=k) +
        f(6)
    )

# Hyperparameter tuning
mean_scores = []
for k in [15, 20, 25]:
    scores = []
    for fold in folds:
        X_train, y_train = fold["train"]
        X_val, y_val = fold["val"]

        # Scale only continuous columns (0-5), leave quarter (6) unscaled
        continuous_cols = list(range(n_continuous))
        scaler = StandardScaler().fit(X_train[:, continuous_cols])
        X_train_scaled = X_train.copy()
        X_val_scaled = X_val.copy()
        X_train_scaled[:, continuous_cols] = scaler.transform(X_train[:, continuous_cols])
        X_val_scaled[:, continuous_cols] = scaler.transform(X_val[:, continuous_cols])

        # GAM with spline + tensor product interaction + categorical terms
        gam = GammaGAM(build_gam_formula(k)).gridsearch(X=X_train_scaled, y=y_train)

        y_pred = gam.predict(X_val_scaled)
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
    "interactions": [
        {"term": "te(rl_pax_str, tot_pax_str)", "indices": [2, 3]},
        {"term": "te(large_ms, lf_ms)", "indices": [4, 5]}
    ],
    "categorical_features": time_features
}

cv_json_path = os.path.join(results_dir, f"{EXPERIMENT_PREFIX}_cv_results.json")
with open(cv_json_path, "w") as json_file:
    json.dump(cv_results, json_file, indent=2)
print(f"✅ Cross-validation results saved to: {cv_json_path}")

# Fit final model on full training set
continuous_cols = list(range(n_continuous))
scaler = StandardScaler().fit(X_all[:, continuous_cols])
X_all_scaled = X_all.copy()
X_test_scaled = X_test.copy()
X_all_scaled[:, continuous_cols] = scaler.transform(X_all[:, continuous_cols])
X_test_scaled[:, continuous_cols] = scaler.transform(X_test[:, continuous_cols])

gam = GammaGAM(build_gam_formula(best_k)).gridsearch(X=X_all_scaled, y=y_all)

print("\n" + "="*80)
print("GAM MODEL SUMMARY")
print("="*80 + "\n")
gam.summary()

# Generate predictions for BOTH training and test sets
y_train_pred = gam.predict(X_all_scaled)
y_test_pred = gam.predict(X_test_scaled)

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

# Term titles: 6 main effects + 2 interactions + quarter (intercept filtered automatically)
term_titles = feature_names + ["rl_pax_str × tot_pax_str", "large_ms × lf_ms"] + time_features

# 1. Partial dependence plots
plot_gam_terms_plotly(
    gam,
    titles=term_titles,
    plot_title=f"GAM Partial Dependence - Quarter + Guided Interactions (n_splines={best_k})",
    save_path=get_save_path("partial_dependence")
)

# 2. Combined diagnostic dashboard (train + test)
plot_gam_combined_dashboard_plotly(
    gam,
    X_all_scaled, y_all, y_train_pred,
    X_test_scaled, y_test, y_test_pred,
    titles=term_titles,
    plot_title=f"GAM Quarter + Guided Interactions - Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f}",
    save_path=get_save_path("combined_dashboard")
)

# 3. Feature importance (model-based)
plot_gam_feature_importance_plotly(
    gam,
    feature_names=term_titles,
    plot_title=f"GAM Feature Importance - Quarter + Guided Interactions (EDOF)",
    save_path=get_save_path("feature_importance")
)

print("\n" + "="*80)
print("VISUALIZATION SUMMARY")
print("="*80)
print(f"✅ Quarter + Guided Interaction Analysis Complete!")
print(f"   📁 Saved to: {results_dir}")
print(f"   📊 Training RMSE: {train_rmse:.4f}")
print(f"   📊 Test RMSE: {test_rmse:.4f}")
print(f"   📈 Training Samples: {len(y_all):,}")
print(f"   📈 Test Samples: {len(y_test):,}")
print(f"   🔗 Interactions: te(rl_pax_str, tot_pax_str), te(large_ms, lf_ms)")
print(f"   🗓️  Categorical: {time_features}")
print(f"   📏 Generalization Ratio: {test_rmse/train_rmse:.2f}" + 
      (" ⚠️  (>1.2 indicates overfitting)" if test_rmse/train_rmse > 1.2 else " ✅ (good generalization)"))
