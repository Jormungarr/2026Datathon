from preprocess.data_utils import import_unit_removed_dataset
from pygam import GammaGAM, s
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from preprocess.data_utils import make_test_and_stratified_folds, import_unit_removed_dataset
from sklearn.model_selection import StratifiedKFold
# Ensure year column exists
# Load and prepare data
feature_names = ['passengers', 'nsmiles', 'rl_pax_str', 'tot_pax_str', 'large_ms', 'lf_ms']
X_test, y_test, _, df_test, df_rest = make_test_and_stratified_folds(feature_cols=feature_names, import_fn=import_unit_removed_dataset)

# Fit PCA
X = df_rest[feature_names].values
X = StandardScaler().fit_transform(X)
pca = PCA()
pc_scores = pca.fit_transform(X)
X_gam = pc_scores[:, :3]  # Only first 3 PCs

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
for train_idx, val_idx in skf.split(X_gam, df_rest["time_index"].to_numpy()):
    X_train, X_val = X_gam[train_idx], X_gam[val_idx]
    y_train, y_val = df_rest["fare"].to_numpy()[train_idx], df_rest["fare"].to_numpy()[val_idx]
    # Fit and evaluate your model here
    for k in [15, 20, 25]:
        gam = GammaGAM(
            s(0, n_splines=k) + s(1, n_splines=k) + s(2, n_splines=k)
        ).gridsearch(X=X_train, y=y_train)
        y_pred = gam.predict(X_val)
        # Calculate RMSE
        rmse = ((y_val - y_pred) ** 2).mean() ** 0.5
        print(f"RMSE for n_splines={k}: {rmse}")
