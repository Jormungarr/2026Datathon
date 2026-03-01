from preprocess.data_utils import import_unit_removed_dataset
from pygam import GammaGAM, s, f
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import numpy as np
from preprocess.data_utils import make_test_and_stratified_folds, import_unit_removed_dataset

# Define feature columns
feature_names = ['passengers', 'nsmiles', 'rl_pax_str', 'tot_pax_str', 'large_ms', 'lf_ms']
time_features = ['quarter', 'Year']
X_test, y_test, folds, X_all, y_all, df_test, df_rest = make_test_and_stratified_folds(
    feature_cols=feature_names + time_features,
    import_fn=import_unit_removed_dataset,
    categorical_encode_cols=time_features
)

# Identify indices for numeric and categorical features
all_features = feature_names + [col for col in X_test.dtype.names or [] if col not in feature_names]
n_numeric = len(feature_names)
n_total = X_test.shape[1]
cat_indices = list(range(n_numeric, n_total))

mean_scores = []
for k in [15, 20, 25]:
    scores = []
    for fold in folds:
        X_train, y_train = fold["train"]
        X_val, y_val = fold["val"]

        # Split numeric and categorical features
        X_train_num = X_train[:, :n_numeric]
        X_val_num = X_val[:, :n_numeric]
        X_train_cat = X_train[:, n_numeric:]
        X_val_cat = X_val[:, n_numeric:]

        # Fit scaler and PCA on numeric features only
        scaler = StandardScaler().fit(X_train_num)
        X_train_num_scaled = scaler.transform(X_train_num)
        X_val_num_scaled = scaler.transform(X_val_num)
        pca = PCA()
        X_train_pca = pca.fit_transform(X_train_num_scaled)
        X_val_pca = pca.transform(X_val_num_scaled)
        X_train_gam = np.hstack([X_train_pca[:, :3], X_train_cat])
        X_val_gam = np.hstack([X_val_pca[:, :3], X_val_cat])

        # Build GAM formula dynamically
        n_cat = X_train_cat.shape[1]
        gam_formula = s(0, n_splines=k) + s(1, n_splines=k) + s(2, n_splines=k)
        for i in range(3, 3 + n_cat):
            gam_formula += f(i)

        gam = GammaGAM(gam_formula).gridsearch(X=X_train_gam, y=y_train)
        y_pred = gam.predict(X_val_gam)
        rmse = ((y_val - y_pred) ** 2).mean() ** 0.5
        scores.append(rmse)
    mean_scores.append((k, sum(scores) / len(scores)))
best_k = min(mean_scores, key=lambda x: x[1])[0]
best_rmse = min(mean_scores, key=lambda x: x[1])[1]
print(f"Best n_splines={best_k} with average RMSE={float(best_rmse):.4f}")

