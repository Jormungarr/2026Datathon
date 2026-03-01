from preprocess.data_utils import import_unit_removed_dataset
from pygam import GammaGAM, s, f
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from preprocess.data_utils import make_test_and_stratified_folds, import_unit_removed_dataset
from sklearn.preprocessing import LabelEncoder


from sklearn.model_selection import StratifiedKFold
# Ensure year column exists
# Load and prepare data
time_features = ['quarter', 'Year']
feature_names = ['passengers', 'nsmiles', 'rl_pax_str', 'tot_pax_str', 'large_ms', 'lf_ms']
X_test, y_test, folds , df_test, df_rest = make_test_and_stratified_folds(feature_cols=feature_names+time_features, import_fn=import_unit_removed_dataset, categorical_encode_cols=time_features)


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
        # Fit scaler on training data only, excluding time variables
        scaler = StandardScaler().fit(X_train[:, 2:])  # Skip 'quarter' and 'Year' columns
        X_train_scaled = X_train.copy()
        X_val_scaled = X_val.copy()
        X_train_scaled[:, 2:] = scaler.transform(X_train[:, 2:])
        X_val_scaled[:, 2:] = scaler.transform(X_val[:, 2:])
        # Fit and evaluate your model here
        gam = GammaGAM(
            s(0, n_splines=k) + s(1, n_splines=k) + s(2, n_splines=k) + s(3, n_splines=k) + s(4, n_splines=k) +
            s(5, n_splines=k) + f(6) + f(7)
        ).gridsearch(X=X_train_scaled, y=y_train)
        y_pred = gam.predict(X_val_scaled)
        # Calculate RMSE
        rmse = ((y_val - y_pred) ** 2).mean() ** 0.5
        scores.append(rmse)
    mean_scores.append((k, sum(scores) / len(scores)))
best_k = min(mean_scores, key=lambda x: x[1])[0]
best_rmse = min(mean_scores, key=lambda x: x[1])[1]
print(f"Best n_splines={best_k} with average RMSE={float(best_rmse):.4f}")

