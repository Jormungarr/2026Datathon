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
X_test, y_test, folds , X_all, y_all, df_test, df_rest = make_test_and_stratified_folds(feature_cols=feature_names, import_fn=import_unit_removed_dataset)


# skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
# for train_idx, val_idx in skf.split(X_gam, df_rest["time_index"].to_numpy()):
#     X_train, X_val = X_gam[train_idx], X_gam[val_idx]
#     y_train, y_val = df_rest["fare"].to_numpy()[train_idx], df_rest["fare"].to_numpy()[val_idx]
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
        pca = PCA()
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_val_pca = pca.transform(X_val_scaled)
        X_train_gam = X_train_pca[:, :3]
        X_val_gam = X_val_pca[:, :3]
        # Fit and evaluate your model here
        gam = GammaGAM(
            s(0, n_splines=k) + s(1, n_splines=k) + s(2, n_splines=k)
        ).gridsearch(X=X_train_gam, y=y_train)
        y_pred = gam.predict(X_val_gam)
        # Calculate RMSE
        rmse = ((y_val - y_pred) ** 2).mean() ** 0.5
        scores.append(rmse)
    mean_scores.append((k, sum(scores) / len(scores)))
best_k = min(mean_scores, key=lambda x: x[1])[0]
best_rmse = min(mean_scores, key=lambda x: x[1])[1]
print(f"Best n_splines={best_k} with average RMSE={float(best_rmse):.4f}")

