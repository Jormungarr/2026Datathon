"""
Visualization utilities for general data analysis.
Assumes input data X is a numpy ndarray of shape (n_samples, n_features).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FixedLocator, FuncFormatter
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def plot_scree(
    X,
    feature_names=None,
    standardize_data=True,
    figsize=(8, 5),
    save_path=None,
    show_kaiser=True,
    show_cumulative=True
):
    """
    Generalized scree plot with eigenvalues, explained variance %, and Kaiser criterion.
    Args:
        X (np.ndarray or pd.DataFrame): Data array of shape (n_samples, n_features)
        feature_names (list, optional): List of feature names. Defaults to None.
        standardize_data (bool): Whether to standardize X before PCA. Defaults to True.
        figsize (tuple, optional): Figure size. Defaults to (8, 5).
        save_path (str, optional): If provided, saves the figure to this path.
        show_kaiser (bool): Show Kaiser criterion line (λ=1). Defaults to True.
        show_cumulative (bool): Plot cumulative curve. Defaults to True.
    """
   

    # Prepare data
    if isinstance(X, pd.DataFrame):
        X_vals = X.values
        if feature_names is None:
            feature_names = X.columns.tolist()
    else:
        X_vals = X
        if feature_names is None:
            feature_names = [f"PC{i+1}" for i in range(X.shape[1])]

    if standardize_data:
        X_vals = StandardScaler().fit_transform(X_vals)

    pca = PCA()
    pca.fit(X_vals)
    eigenvalues = pca.explained_variance_
    explained_ratio = pca.explained_variance_ratio_
    n_components = len(eigenvalues)
    x = np.arange(1, n_components + 1)

    plt.figure(figsize=figsize)

    # Bars
    bars = plt.bar(x, eigenvalues, alpha=0.7)

    # Percent labels above bars
    for i in range(n_components):
        plt.text(x[i],
                 eigenvalues[i] + 0.05,
                 f"{explained_ratio[i]*100:.1f}%",
                 ha='center',
                 fontsize=11)

    # Kaiser criterion line
    if show_kaiser:
        plt.axhline(y=1, linestyle='--', color='gray', label="Kaiser criterion (λ = 1)")

    # Optional cumulative curve
    if show_cumulative:
        plt.plot(x, eigenvalues, marker='o', color='black', linestyle='-')

    plt.xlabel("Principal Component")
    plt.ylabel("Eigenvalue")
    plt.title("Eigenvalue Spectrum (Correlation PCA)")
    plt.xticks(x)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', facecolor="white")
    plt.show()


def plot_corner_heatmap(
    X,
    feature_names=None,
    hue=None,
    hue_labels=None,
    palette="Set2",
    figsize=(10, 10),
    bins=30,
    save_path=None,
    legend_title="Group"
):
    """
    Generalized corner plot (PairGrid) with upper triangle correlation heatmap cells.
    Args:
        X (np.ndarray or pd.DataFrame): Data array of shape (n_samples, n_features)
        feature_names (list, optional): List of feature names. Defaults to None.
        hue (array-like, optional): Group labels for coloring. Defaults to None.
        hue_labels (list, optional): Custom labels for hue. Defaults to None.
        palette (str or list, optional): Color palette for hue. Defaults to "Set2".
        figsize (tuple, optional): Figure size. Defaults to (10, 10).
        bins (int, optional): Number of bins for histograms. Defaults to 30.
        save_path (str, optional): If provided, saves the figure to this path.
        legend_title (str, optional): Legend title. Defaults to "Group".
    """
    # Prepare DataFrame
    if isinstance(X, np.ndarray):
        df = pd.DataFrame(X, columns=feature_names if feature_names is not None else [f"f{i}" for i in range(X.shape[1])])
    else:
        df = X.copy()
    if feature_names is not None:
        plot_features = [f for f in feature_names if f in df.columns]
    else:
        plot_features = df.columns.tolist()
    if hue is not None:
        df["_hue"] = hue
        hue_col = "_hue"
    else:
        hue_col = None

    # Correlation matrix for color scale
    X_vals = df[plot_features].to_numpy()
    C = np.corrcoef(X_vals, rowvar=False)
    tri = C[np.triu_indices_from(C, k=1)]
    vmin, vmax = float(tri.min()), float(tri.max())
    pad = 0.02
    vmin = max(-1.0, vmin - pad)
    vmax = min(1.0, vmax + pad)
    cmap = plt.get_cmap("magma")
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Upper triangle cell: colored by correlation
    def corr_cell(x, y, **kws):
        ax = plt.gca()
        if getattr(ax, "_corr_drawn", False):
            return
        ax._corr_drawn = True
        r = np.corrcoef(x, y)[0, 1]
        ax.set_facecolor(cmap(norm(r)))
        for t in list(ax.texts):
            t.remove()
        ax.text(
            0.5, 0.5, f"{r:.2f}",
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=12, fontweight="bold", color="white"
        )
        ax.tick_params(axis="both", which="both",
                       bottom=False, left=False,
                       labelbottom=False, labelleft=False,
                       top=False, right=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    # PairGrid
    g = sns.PairGrid(
        data=df,
        vars=plot_features,
        hue=hue_col,
        palette=palette,
        diag_sharey=False
    )
    g.map_lower(
        sns.scatterplot,
        s=10, alpha=0.75,
        edgecolor="white", linewidth=0.2,
        rasterized=True
    )
    g.map_diag(
        sns.kdeplot,
        fill=True, alpha=0.35,
        common_norm=True,
        bw_adjust=1.0,
        cut=0,
        linewidth=1.0
    )
    g.map_upper(corr_cell)

    # Styling/layout
    n = len(plot_features)
    g.fig.set_size_inches(figsize)
    right_margin = 0.20
    g.fig.subplots_adjust(
        left=0.08, bottom=0.14,
        right=1 - right_margin,
        top=0.98,
        wspace=0.10, hspace=0.10
    )
    for ax in g.axes.flat:
        for sp in ax.spines.values():
            sp.set_linewidth(0.6)
        ax.tick_params(width=0.6, length=3, labelsize=7)

    NT = 6
    def set_fixed_ticks(ax):
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        if not np.isfinite([xmin, xmax, ymin, ymax]).all():
            return
        if xmin == xmax or ymin == ymax:
            return
        xt = np.linspace(xmin, xmax, NT)
        yt = np.linspace(ymin, ymax, NT)
        ax.xaxis.set_major_locator(FixedLocator(xt))
        ax.yaxis.set_major_locator(FixedLocator(yt))
        def pretty2(x, pos=None):
            return f"{x:.2f}".rstrip("0").rstrip(".")
        fmt = FuncFormatter(pretty2)
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)

    for i in range(n):
        for j in range(n):
            ax = g.axes[i, j]
            if i < j:
                ax.grid(False)
                ax.tick_params(axis="both", which="both",
                               bottom=False, left=False,
                               labelbottom=False, labelleft=False,
                               top=False, right=False)
                continue
            set_fixed_ticks(ax)
            ax.grid(True)
            show_left = (j == 0)
            show_bottom = (i == n - 1)
            ax.tick_params(axis="both", which="both", bottom=True, left=True)
            ax.tick_params(axis="y", labelleft=show_left)
            ax.tick_params(axis="x", labelbottom=show_bottom)
    for j in range(n):
        ax = g.axes[n-1, j]
        labs = ax.get_xticklabels()
        for k, t in enumerate(labs):
            if k % 2 == 1:
                t.set_visible(False)

    # Legend
    src_ax = None
    for ax in g.axes.flat:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            src_ax = ax
            break
    if src_ax is not None:
        handles, labels = src_ax.get_legend_handles_labels()
        for ax in g.axes.flat:
            lg = ax.get_legend()
            if lg is not None:
                lg.remove()
        leg = g.fig.legend(
            handles, hue_labels if hue_labels is not None else labels, title=legend_title,
            loc="upper left",
            bbox_to_anchor=(1 - right_margin + 0.01, 0.95),
            bbox_transform=g.fig.transFigure,
            frameon=True
        )
        frame = leg.get_frame()
        frame.set_facecolor("#ffffff")
        frame.set_edgecolor("#cccccc")
        frame.set_linewidth(0.8)
        frame.set_alpha(0.95)
        leg.get_title().set_fontsize(16)
        for txt in leg.get_texts():
            txt.set_fontsize(14)

    # Colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar_ax = g.fig.add_axes([1 - right_margin + 0.035, 0.26, 0.022, 0.52])
    cbar = g.fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Correlation Coefficient (r)", fontsize=11, labelpad=6)
    cbar.ax.tick_params(labelsize=9)

    if save_path:
        g.fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.show()
