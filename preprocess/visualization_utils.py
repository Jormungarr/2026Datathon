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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import os

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
        # Add percentage labels above cumulative curve
        cum_var = np.cumsum(explained_ratio)
        plt.plot(x, cum_var * max(eigenvalues), marker='o', color='blue', linestyle='--', label='Cumulative Variance')
        for i in range(n_components):
            plt.text(x[i], cum_var[i] * max(eigenvalues) + 0.05, f"{cum_var[i]*100:.1f}%", ha='center', fontsize=9, color='blue')

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


def plot_pc_loadings(
    pca,
    feature_names=None,
    pc_index=0,
    ci_low=None,
    ci_high=None,
    figsize=(8, 5),
    title=None,
    save_path=None
):
    """
    Plot loadings for any principal component as a horizontal bar plot.
    Optionally adds confidence intervals.
    Args:
        pca: fitted sklearn PCA object
        feature_names: list of feature names
        pc_index: which PC to plot (0-based)
        ci_low: lower bounds for CIs (array-like, optional)
        ci_high: upper bounds for CIs (array-like, optional)
        figsize: figure size
        title: plot title
        save_path: if provided, saves the figure
    """
    loadings = pca.components_[pc_index]
    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(len(loadings))]
    y = np.arange(len(feature_names))
    colors = np.where(loadings >= 0, "#1f77b4", "#d62728")  # blue/red

    plt.figure(figsize=figsize)
    plt.barh(y, loadings, color=colors, alpha=0.85)

    # Add confidence intervals if provided
    if ci_low is not None and ci_high is not None:
        err_left = loadings - ci_low
        err_right = ci_high - loadings
        plt.errorbar(
            loadings, y,
            xerr=np.vstack([err_left, err_right]),
            fmt="none",
            capsize=4,
            linewidth=1.5,
            color="black",
        )

    plt.axvline(0, linewidth=1, color="black")
    plt.yticks(y, feature_names)
    plt.xlabel("Loading")
    plt.title(title or f"PC{pc_index+1} Loadings")
    plt.tight_layout()
    plt.gca().invert_yaxis()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.show()


def plot_interactive_pca_scatter(
    df,
    feature_names,
    color_columns,
    time_column='time_index',
    pc_x=0,
    pc_y=1,
    standardize=True,
    color_scale_mode='uniform',  # 'uniform', 'exponential', 'normal'
    figsize=(900, 700),
    title="Interactive PCA Scatter Plot",
    save_path=None
):
    """
    Create an interactive PCA scatter plot with toggles for:
    1. Time period (Year + Quarter)
    2. Color-coded value (select from multiple numeric columns)
    3. Color scale distribution mode (uniform/quantile, exponential, normal)
    
    Args:
        df (pd.DataFrame): DataFrame containing features, color columns, and time index
        feature_names (list): List of feature names for PCA
        color_columns (list): List of column names to use for color coding
        time_column (str): Column name for time index (e.g., '2025 Q1')
        pc_x (int): PC index for x-axis (0-based)
        pc_y (int): PC index for y-axis (0-based)
        standardize (bool): Whether to standardize data before PCA
        color_scale_mode (str): 'uniform', 'exponential', or 'normal'
        figsize (tuple): Figure size (width, height)
        title (str): Plot title
        save_path (str): If provided, saves the HTML figure
    
    Returns:
        plotly.graph_objects.Figure
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    
    df = df.copy().dropna(subset=feature_names + color_columns + [time_column])
    
    # Fit PCA
    X = df[feature_names].values
    if standardize:
        X = StandardScaler().fit_transform(X)
    
    pca = PCA()
    pc_scores = pca.fit_transform(X)
    
    # Add PC scores to dataframe
    for i in range(pc_scores.shape[1]):
        df[f'PC{i+1}'] = pc_scores[:, i]
    
    pc_x_col = f'PC{pc_x+1}'
    pc_y_col = f'PC{pc_y+1}'
    
    # Get unique time periods sorted
    time_periods = sorted(df[time_column].unique())
    
    # Create figure
    fig = go.Figure()
    
    # For each time period and color column combination, add a trace
    for time_idx, time_period in enumerate(time_periods):
        df_time = df[df[time_column] == time_period]
        
        for col_idx, color_col in enumerate(color_columns):
            # Compute color scale based on mode
            color_vals = df_time[color_col].values
            scaled_colors = _scale_colors(color_vals, mode=color_scale_mode)
            
            # Determine visibility: only first time period and first color column visible initially
            visible = (time_idx == 0) and (col_idx == 0)
            
            fig.add_trace(go.Scatter(
                x=df_time[pc_x_col],
                y=df_time[pc_y_col],
                mode='markers',
                marker=dict(
                    size=6,
                    color=scaled_colors,
                    colorscale='Viridis',
                    showscale=(col_idx == 0 and time_idx == 0),
                    colorbar=dict(
                        title=color_col,
                        tickvals=_get_colorbar_ticks(color_vals, mode=color_scale_mode),
                        ticktext=_get_colorbar_ticktext(color_vals, mode=color_scale_mode),
                    ),
                    opacity=0.7,
                ),
                text=[
                    f"PC{pc_x+1}: {x:.2f}<br>PC{pc_y+1}: {y:.2f}<br>{color_col}: {v:.2f}"
                    for x, y, v in zip(df_time[pc_x_col], df_time[pc_y_col], df_time[color_col])
                ],
                hoverinfo='text',
                name=f"{time_period} - {color_col}",
                visible=visible
            ))
    
    # Create dropdown menus
    n_time = len(time_periods)
    n_colors = len(color_columns)
    total_traces = n_time * n_colors
    
    # Time period dropdown
    time_buttons = []
    for t_idx, time_period in enumerate(time_periods):
        visibility = [False] * total_traces
        # Show all color columns for this time period (but only first color column visible)
        for c_idx in range(n_colors):
            trace_idx = t_idx * n_colors + c_idx
            visibility[trace_idx] = (c_idx == 0)
        
        time_buttons.append(dict(
            label=str(time_period),
            method='update',
            args=[
                {'visible': visibility},
                {'title': f"{title} - {time_period}"}
            ]
        ))
    
    # Color column dropdown
    color_buttons = []
    for c_idx, color_col in enumerate(color_columns):
        visibility = [False] * total_traces
        # Show this color column for all time periods (but only first time period visible)
        for t_idx in range(n_time):
            trace_idx = t_idx * n_colors + c_idx
            visibility[trace_idx] = (t_idx == 0)
        
        color_buttons.append(dict(
            label=color_col,
            method='update',
            args=[
                {'visible': visibility},
                {'coloraxis.colorbar.title': color_col}
            ]
        ))
    
    # Color scale mode dropdown
    scale_buttons = [
        dict(label='Uniform (Quantile)', method='relayout', args=[{'title': f'{title} - Uniform Scale'}]),
        dict(label='Exponential', method='relayout', args=[{'title': f'{title} - Exponential Scale'}]),
        dict(label='Normal', method='relayout', args=[{'title': f'{title} - Normal Scale'}]),
    ]
    
    # Update layout with dropdowns
    fig.update_layout(
        title=f"{title} - {time_periods[0]}",
        xaxis_title=f"PC{pc_x+1} ({pca.explained_variance_ratio_[pc_x]*100:.1f}%)",
        yaxis_title=f"PC{pc_y+1} ({pca.explained_variance_ratio_[pc_y]*100:.1f}%)",
        width=figsize[0],
        height=figsize[1],
        updatemenus=[
            dict(
                buttons=time_buttons,
                direction='down',
                showactive=True,
                x=0.0,
                xanchor='left',
                y=1.15,
                yanchor='top',
                name='Time Period'
            ),
            dict(
                buttons=color_buttons,
                direction='down',
                showactive=True,
                x=0.25,
                xanchor='left',
                y=1.15,
                yanchor='top',
                name='Color Value'
            ),
        ],
        annotations=[
            dict(text="Time:", x=0.0, xref="paper", y=1.20, yref="paper", showarrow=False, font=dict(size=12)),
            dict(text="Color:", x=0.25, xref="paper", y=1.20, yref="paper", showarrow=False, font=dict(size=12)),
        ]
    )
    
    if save_path:
        # FIX: Ensure proper HTML extension and directory creation
        if not save_path.endswith('.html'):
            html_path = save_path + '.html'
        else:
            html_path = save_path
            
        os.makedirs(os.path.dirname(html_path), exist_ok=True)
        fig.write_html(html_path)
        print(f"✅ Interactive plot saved to: {html_path}")
    
    fig.show()
    return fig


def plot_interactive_pca_scatter_full(
    df,
    feature_names,
    color_columns,
    time_column='time_index',
    year_column='Year',
    pc_x=0,
    pc_y=1,
    standardize=True,
    n_ticks=6,
    figsize=(1000, 750),
    title="Interactive PCA Scatter Plot",
    save_path=None
):
    """
    Create an interactive PCA scatter plot with dropdown toggles for:
      - Time period (All Data, Yearly, Quarterly)
      - Color-coded value (multiple columns)
      - Color scale distribution (uniform, exponential, normal, power, symlog, linear)
    Features:
      - Always shows colorbar for all traces
      - Allows user to set number of colorbar ticks (n_ticks)
      - Supports additional scale distributions: power, symlog, linear
      - Adds toggles for "All Data" and yearly aggregation in addition to quarterly
      - Fixes toggle cutoff/overlap and improves layout
    Args:
        df (pd.DataFrame): DataFrame containing features, color columns, and time index
        feature_names (list): List of feature names for PCA
        color_columns (list): List of column names to use for color coding
        time_column (str): Column name for time index (e.g., '2025 Q1')
        year_column (str): Column name for year (e.g., 'Year')
        pc_x (int): PC index for x-axis (0-based)
        pc_y (int): PC index for y-axis (0-based)
        standardize (bool): Whether to standardize data before PCA
        n_ticks (int): Number of ticks on the color gradient scale
        figsize (tuple): Figure size (width, height)
        title (str): Plot title
        save_path (str): If provided, saves the HTML figure
    Returns:
        plotly.graph_objects.Figure: The interactive PCA scatter plot figure
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import plotly.graph_objects as go

    # Ensure year column exists
    if year_column not in df.columns:
        df = df.copy()
        df[year_column] = df[time_column].astype(str).str.extract(r'(\d{4})')[0]

    df = df.copy().dropna(subset=feature_names + color_columns + [time_column, year_column])

    # Fit PCA
    X = df[feature_names].values
    if standardize:
        X = StandardScaler().fit_transform(X)
    pca = PCA()
    pc_scores = pca.fit_transform(X)
    n_pcs = pc_scores.shape[1]
    for i in range(n_pcs):
        df[f'PC{i+1}'] = pc_scores[:, i]
    pc_x_col = f'PC{pc_x+1}'
    pc_y_col = f'PC{pc_y+1}'

    # Time options: All Data, each year, each quarter
    time_periods_quarterly = sorted(df[time_column].unique())
    time_periods_yearly = sorted(df[year_column].unique())
    time_options = ["All Data"] + [f"Year {y}" for y in time_periods_yearly] + list(time_periods_quarterly)

    scale_modes = ['uniform', 'exponential', 'normal', 'power', 'symlog', 'linear']
    scale_labels = {
        'uniform': 'Uniform (Quantile)',
        'exponential': 'Exponential (Log)',
        'normal': 'Normal (Z-score)',
        'power': 'Power (Sqrt)',
        'symlog': 'Symmetric Log',
        'linear': 'Linear (Min-Max)'
    }

    # Only build traces for the currently selected options for speed
    # Default selections
    default_t, default_c, default_s = 0, 0, 0

    def get_df_time(time_opt):
        if time_opt == "All Data":
            return df.copy()
        elif time_opt.startswith("Year "):
            year_val = time_opt.replace("Year ", "")
            return df[df[year_column].astype(str) == str(year_val)]
        else:
            return df[df[time_column] == time_opt]

    # Initial data
    df_time = get_df_time(time_options[default_t])
    color_col = color_columns[default_c]
    scale_mode = scale_modes[default_s]
    color_vals = df_time[color_col].values
    scaled_colors = _scale_colors(color_vals, mode=scale_mode)
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_time[pc_x_col],
        y=df_time[pc_y_col],
        mode='markers',
        marker=dict(
            size=6,
            color=scaled_colors,
            colorscale='Viridis',
            showscale=True,
            cmin=0,
            cmax=1,
            colorbar=dict(
                title=dict(text=color_col, side='right'),
                tickvals=_get_colorbar_ticks(color_vals, mode=scale_mode, n_ticks=n_ticks),
                ticktext=_get_colorbar_ticktext(color_vals, mode=scale_mode, n_ticks=n_ticks),
                len=0.6,
                y=0.5,
                yanchor='middle',
            ),
            opacity=0.7,
        ),
        text=[
            f"{pc_x_col}: {x:.2f}<br>{pc_y_col}: {y:.2f}<br>{color_col}: {v:.2f}"
            for x, y, v in zip(df_time[pc_x_col], df_time[pc_y_col], df_time[color_col])
        ],
        hoverinfo='text',
        name=f"{time_options[default_t]} | {color_col} | {scale_mode}",
        visible=True
    ))

    # Dropdowns
    def make_update_args(t_idx=None, c_idx=None, s_idx=None):
        # Get new data for the selected options
        t_idx = default_t if t_idx is None else t_idx
        c_idx = default_c if c_idx is None else c_idx
        s_idx = default_s if s_idx is None else s_idx
        df_time = get_df_time(time_options[t_idx])
        color_col = color_columns[c_idx]
        scale_mode = scale_modes[s_idx]
        color_vals = df_time[color_col].values
        scaled_colors = _scale_colors(color_vals, mode=scale_mode)
        return dict(
            x=[df_time[pc_x_col]],
            y=[df_time[pc_y_col]],
            marker=dict(
                size=6,
                color=scaled_colors,
                colorscale='Viridis',
                showscale=True,
                cmin=0,
                cmax=1,
                colorbar=dict(
                    title=dict(text=color_col, side='right'),
                    tickvals=_get_colorbar_ticks(color_vals, mode=scale_mode, n_ticks=n_ticks),
                    ticktext=_get_colorbar_ticktext(color_vals, mode=scale_mode, n_ticks=n_ticks),
                    len=0.6,
                    y=0.5,
                    yanchor='middle',
                ),
                opacity=0.7,
            ),
            text=[
                f"{pc_x_col}: {x:.2f}<br>{pc_y_col}: {y:.2f}<br>{color_col}: {v:.2f}"
                for x, y, v in zip(df_time[pc_x_col], df_time[pc_y_col], df_time[color_col])
            ],
            name=f"{time_options[t_idx]} | {color_col} | {scale_mode}",
        ), f"{title} - {time_options[t_idx]}"

    # Time dropdown
    time_buttons = []
    for t_idx, time_opt in enumerate(time_options):
        update_args, new_title = make_update_args(t_idx=t_idx)
        time_buttons.append(dict(
            label=str(time_opt),
            method='update',
            args=[
                update_args,
                {'title': new_title}
            ]
        ))

    # Color dropdown
    color_buttons = []
    for c_idx, color_col in enumerate(color_columns):
        update_args, _ = make_update_args(c_idx=c_idx)
        color_buttons.append(dict(
            label=color_col,
            method='update',
            args=[update_args, {}]
        ))

    # Scale dropdown
    scale_buttons = []
    for s_idx, scale_mode in enumerate(scale_modes):
        update_args, _ = make_update_args(s_idx=s_idx)
        scale_buttons.append(dict(
            label=scale_labels[scale_mode],
            method='update',
            args=[update_args, {}]
        ))

    # Layout with improved toggle placement
    fig.update_layout(
        title=dict(
            text=f"{title} - {time_options[default_t]}",
            y=0.96,
            x=0.5,
            xanchor='center',
            yanchor='top',
            font=dict(size=16)
        ),
        xaxis_title=f"PC{pc_x+1} ({pca.explained_variance_ratio_[pc_x]*100:.1f}%)",
        yaxis_title=f"PC{pc_y+1} ({pca.explained_variance_ratio_[pc_y]*100:.1f}%)",
        width=figsize[0],
        height=figsize[1],
        margin=dict(t=120, l=80, r=120, b=80),
        updatemenus=[
            dict(
                buttons=time_buttons,
                direction='down',
                showactive=True,
                x=0.0,
                xanchor='left',
                y=1.13,
                yanchor='top',
                bgcolor='rgba(255,255,255,0.95)',
                bordercolor='#ccc',
                font=dict(size=11),
            ),
            dict(
                buttons=color_buttons,
                direction='down',
                showactive=True,
                x=0.18,
                xanchor='left',
                y=1.13,
                yanchor='top',
                bgcolor='rgba(255,255,255,0.95)',
                bordercolor='#ccc',
                font=dict(size=11),
            ),
            dict(
                buttons=scale_buttons,
                direction='down',
                showactive=True,
                x=0.36,
                xanchor='left',
                y=1.13,
                yanchor='top',
                bgcolor='rgba(255,255,255,0.95)',
                bordercolor='#ccc',
                font=dict(size=11),
            ),
        ],
        annotations=[
            dict(text="<b>Time:</b>", x=0.0, xref="paper", y=1.17, yref="paper", showarrow=False, font=dict(size=11)),
            dict(text="<b>Color:</b>", x=0.18, xref="paper", y=1.17, yref="paper", showarrow=False, font=dict(size=11)),
            dict(text="<b>Scale:</b>", x=0.36, xref="paper", y=1.17, yref="paper", showarrow=False, font=dict(size=11)),
        ]
    )

    if save_path:
        # FIX: Ensure proper HTML extension and directory creation
        if not save_path.endswith('.html'):
            html_path = save_path + '.html'
        else:
            html_path = save_path
            
        os.makedirs(os.path.dirname(html_path), exist_ok=True)
        fig.write_html(html_path)
        print(f"✅ Interactive dashboard saved to: {html_path}")
    
    fig.show()
    return fig


def _scale_colors(values, mode='uniform'):
    """
    Scale values for color mapping based on distribution mode.
    
    Args:
        values: array of numeric values
        mode: 'uniform' (quantile), 'exponential', or 'normal'
    
    Returns:
        scaled values between 0 and 1
    """
    values = np.array(values)
    
    if mode == 'uniform':
        # Quantile-based scaling (uniform distribution of colors)
        from scipy.stats import rankdata
        ranks = rankdata(values, method='average')
        scaled = (ranks - 1) / (len(ranks) - 1) if len(ranks) > 1 else np.zeros_like(values)
    
    elif mode == 'exponential':
        # Log-transform for exponentially sparse data
        min_val = values.min()
        shifted = values - min_val + 1  # Shift to positive
        log_vals = np.log(shifted)
        scaled = (log_vals - log_vals.min()) / (log_vals.max() - log_vals.min() + 1e-10)
    
    elif mode == 'normal':
        # Z-score based scaling (assumes normal distribution)
        mean = values.mean()
        std = values.std()
        if std == 0:
            scaled = np.zeros_like(values)
        else:
            z_scores = (values - mean) / std
            # Map z-scores to [0,1] using CDF of standard normal
            from scipy.stats import norm
            scaled = norm.cdf(z_scores)
    
    elif mode == 'power':
        scaled_vals = np.sqrt(np.abs(values))
        scaled = (scaled_vals - scaled_vals.min()) / (scaled_vals.max() - scaled_vals.min() + 1e-10)
    
    elif mode == 'symlog':
        symlog_vals = np.sign(values) * np.log1p(np.abs(values))
        scaled = (symlog_vals - symlog_vals.min()) / (symlog_vals.max() - symlog_vals.min() + 1e-10)
    
    elif mode == 'linear':
        scaled = (values - values.min()) / (values.max() - values.min() + 1e-10)
    
    else:
        scaled = (values - values.min()) / (values.max() - values.min() + 1e-10)
    
    return scaled


def _get_colorbar_ticks(values, mode='uniform', n_ticks=6):
    """
    Get colorbar tick positions based on scaling mode.
    """
    values = np.array(values)
    
    if mode == 'uniform':
        # Quantile ticks
        percentiles = np.linspace(0, 100, n_ticks)
        tick_vals = np.percentile(values, percentiles)
    
    elif mode == 'exponential':
        # Log-spaced ticks
        min_val = values.min()
        max_val = values.max()
        shifted_min = 1
        shifted_max = max_val - min_val + 1
        log_ticks = np.linspace(np.log(shifted_min), np.log(shifted_max), n_ticks)
        tick_vals = np.exp(log_ticks) + min_val - 1
    
    elif mode == 'normal':
        # Z-score based ticks
        mean = values.mean()
        std = values.std()
        z_ticks = np.linspace(-2, 2, n_ticks)
        tick_vals = mean + z_ticks * std
    
    else:
        tick_vals = np.linspace(values.min(), values.max(), n_ticks)
    
    # Scale tick positions to [0, 1] for colorbar
    scaled_ticks = _scale_colors(tick_vals, mode=mode)
    return scaled_ticks.tolist()


def _get_colorbar_ticktext(values, mode='uniform', n_ticks=6):
    """
    Get colorbar tick labels based on scaling mode.
    """
    values = np.array(values)
    
    if mode == 'uniform':
        percentiles = np.linspace(0, 100, n_ticks)
        tick_vals = np.percentile(values, percentiles)
    
    elif mode == 'exponential':
        min_val = values.min()
        max_val = values.max()
        shifted_min = 1
        shifted_max = max_val - min_val + 1
        log_ticks = np.linspace(np.log(shifted_min), np.log(shifted_max), n_ticks)
        tick_vals = np.exp(log_ticks) + min_val - 1
    
    elif mode == 'normal':
        mean = values.mean()
        std = values.std()
        z_ticks = np.linspace(-2, 2, n_ticks)
        tick_vals = mean + z_ticks * std
    
    else:
        tick_vals = np.linspace(values.min(), values.max(), n_ticks)
    
    return [f"{v:.1f}" for v in tick_vals]


def plot_interactive_pc_vs_value(
    df,
    feature_names,
    value_columns,
    time_column='time_index',
    year_column='Year',
    pc_x=0,
    standardize=True,
    n_ticks=6,
    figsize=(1000, 750),
    title="PC vs Value Interactive Plot",
    save_path=None
):
    """
    Interactive plot: PC (fixed, user input) vs value column (toggle), with time, value, and scale toggles.
    Args:
        df (pd.DataFrame): DataFrame containing features, value columns, and time index
        feature_names (list): List of feature names for PCA
        value_columns (list): List of column names to use for y-axis and color coding
        time_column (str): Column name for time index (e.g., '2025 Q1')
        year_column (str): Column name for year (e.g., 'Year')
        pc_x (int): PC index for x-axis (0-based)
        standardize (bool): Whether to standardize data before PCA
        n_ticks (int): Number of ticks on the color gradient scale
        figsize (tuple): Figure size (width, height)
        title (str): Plot title
        save_path (str): If provided, saves the HTML figure
    Returns:
        plotly.graph_objects.Figure: The interactive plot
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import plotly.graph_objects as go

    # Ensure year column exists
    if year_column not in df.columns:
        df = df.copy()
        df[year_column] = df[time_column].astype(str).str.extract(r'(\d{4})')[0]

    df = df.copy().dropna(subset=feature_names + value_columns + [time_column, year_column])

    # Fit PCA
    X = df[feature_names].values
    if standardize:
        X = StandardScaler().fit_transform(X)
    pca = PCA()
    pc_scores = pca.fit_transform(X)
    n_pcs = pc_scores.shape[1]
    for i in range(n_pcs):
        df[f'PC{i+1}'] = pc_scores[:, i]
    pc_x_col = f'PC{pc_x+1}'

    # Time options: All Data, each year, each quarter
    time_periods_quarterly = sorted(df[time_column].unique())
    time_periods_yearly = sorted(df[year_column].unique())
    time_options = ["All Data"] + [f"Year {y}" for y in time_periods_yearly] + list(time_periods_quarterly)

    scale_modes = ['uniform', 'exponential', 'normal', 'power', 'symlog', 'linear']
    scale_labels = {
        'uniform': 'Uniform (Quantile)',
        'exponential': 'Exponential (Log)',
        'normal': 'Normal (Z-score)',
        'power': 'Power (Sqrt)',
        'symlog': 'Symmetric Log',
        'linear': 'Linear (Min-Max)'
    }

    # Only build traces for the currently selected options for speed
    default_t, default_v, default_s = 0, 0, 0

    def get_df_time(time_opt):
        if time_opt == "All Data":
            return df.copy()
        elif time_opt.startswith("Year "):
            year_val = time_opt.replace("Year ", "")
            return df[df[year_column].astype(str) == str(year_val)]
        else:
            return df[df[time_column] == time_opt]

    # Initial data
    df_time = get_df_time(time_options[default_t])
    value_col = value_columns[default_v]
    scale_mode = scale_modes[default_s]
    color_vals = df_time[value_col].values
    scaled_colors = _scale_colors(color_vals, mode=scale_mode)
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_time[pc_x_col],
        y=df_time[value_col],
        mode='markers',
        marker=dict(
            size=6,
            color=scaled_colors,
            colorscale='Viridis',
            showscale=True,
            cmin=0,
            cmax=1,
            colorbar=dict(
                title=dict(text=value_col, side='right'),
                tickvals=_get_colorbar_ticks(color_vals, mode=scale_mode, n_ticks=n_ticks),
                ticktext=_get_colorbar_ticktext(color_vals, mode=scale_mode, n_ticks=n_ticks),
                len=0.6,
                y=0.5,
                yanchor='middle',
            ),
            opacity=0.7,
        ),
        text=[
            f"{pc_x_col}: {x:.2f}<br>{value_col}: {y:.2f}"
            for x, y in zip(df_time[pc_x_col], df_time[value_col])
        ],
        hoverinfo='text',
        name=f"{time_options[default_t]} | {value_col} | {scale_mode}",
        visible=True
    ))

    # Dropdowns
    def make_update_args(t_idx=None, v_idx=None, s_idx=None):
        t_idx = default_t if t_idx is None else t_idx
        v_idx = default_v if v_idx is None else v_idx
        s_idx = default_s if s_idx is None else s_idx
        df_time = get_df_time(time_options[t_idx])
        value_col = value_columns[v_idx]
        scale_mode = scale_modes[s_idx]
        color_vals = df_time[value_col].values
        scaled_colors = _scale_colors(color_vals, mode=scale_mode)
        return dict(
            x=[df_time[pc_x_col]],
            y=[df_time[value_col]],
            marker=dict(
                size=6,
                color=scaled_colors,
                colorscale='Viridis',
                showscale=True,
                cmin=0,
                cmax=1,
                colorbar=dict(
                    title=dict(text=value_col, side='right'),
                    tickvals=_get_colorbar_ticks(color_vals, mode=scale_mode, n_ticks=n_ticks),
                    ticktext=_get_colorbar_ticktext(color_vals, mode=scale_mode, n_ticks=n_ticks),
                    len=0.6,
                    y=0.5,
                    yanchor='middle',
                ),
                opacity=0.7,
            ),
            text=[
                f"{pc_x_col}: {x:.2f}<br>{value_col}: {y:.2f}"
                for x, y in zip(df_time[pc_x_col], df_time[value_col])
            ],
            name=f"{time_options[t_idx]} | {value_col} | {scale_mode}",
        ), f"{title} - {time_options[t_idx]}"

    # Time dropdown
    time_buttons = []
    for t_idx, time_opt in enumerate(time_options):
        update_args, new_title = make_update_args(t_idx=t_idx)
        time_buttons.append(dict(
            label=str(time_opt),
            method='update',
            args=[
                update_args,
                {'title': new_title}
            ]
        ))

    # Value dropdown
    value_buttons = []
    for v_idx, value_col in enumerate(value_columns):
        update_args, _ = make_update_args(v_idx=v_idx)
        value_buttons.append(dict(
            label=value_col,
            method='update',
            args=[update_args, {}]
        ))

    # Scale dropdown
    scale_buttons = []
    for s_idx, scale_mode in enumerate(scale_modes):
        update_args, _ = make_update_args(s_idx=s_idx)
        scale_buttons.append(dict(
            label=scale_labels[scale_mode],
            method='update',
            args=[update_args, {}]
        ))

    # Layout with improved toggle placement
    fig.update_layout(
        title=dict(
            text=f"{title} - {time_options[default_t]}",
            y=0.96,
            x=0.5,
            xanchor='center',
            yanchor='top',
            font=dict(size=16)
        ),
        xaxis_title=f"PC{pc_x+1} ({pca.explained_variance_ratio_[pc_x]*100:.1f}%)",
        yaxis_title=f"{value_columns[default_v]}",
        width=figsize[0],
        height=figsize[1],
        margin=dict(t=120, l=80, r=120, b=80),
        updatemenus=[
            dict(
                buttons=time_buttons,
                direction='down',
                showactive=True,
                x=0.0,
                xanchor='left',
                y=1.13,
                yanchor='top',
                bgcolor='rgba(255,255,255,0.95)',
                bordercolor='#ccc',
                font=dict(size=11),
            ),
            dict(
                buttons=value_buttons,
                direction='down',
                showactive=True,
                x=0.18,
                xanchor='left',
                y=1.13,
                yanchor='top',
                bgcolor='rgba(255,255,255,0.95)',
                bordercolor='#ccc',
                font=dict(size=11),
            ),
            dict(
                buttons=scale_buttons,
                direction='down',
                showactive=True,
                x=0.36,
                xanchor='left',
                y=1.13,
                yanchor='top',
                bgcolor='rgba(255,255,255,0.95)',
                bordercolor='#ccc',
                font=dict(size=11),
            ),
        ],
        annotations=[
            dict(text="<b>Time:</b>", x=0.0, xref="paper", y=1.17, yref="paper", showarrow=False, font=dict(size=11)),
            dict(text="<b>Y Value:</b>", x=0.18, xref="paper", y=1.17, yref="paper", showarrow=False, font=dict(size=11)),
            dict(text="<b>Scale:</b>", x=0.36, xref="paper", y=1.17, yref="paper", showarrow=False, font=dict(size=11)),
        ]
    )

    if save_path:
        # FIX: Ensure proper HTML extension and directory creation
        if not save_path.endswith('.html'):
            html_path = save_path + '.html'
        else:
            html_path = save_path
            
        os.makedirs(os.path.dirname(html_path), exist_ok=True)
        fig.write_html(html_path)
        print(f"✅ Interactive plot saved to: {html_path}")
    
    fig.show()
    return fig


def plot_gam_terms_plotly(gam, term_indices=None, titles=None, save_path=None, plot_title=None):
    """
    Interactive Plotly version of GAM partial dependence plots.
    Much better than matplotlib for exploration and presentation.
    
    Parameters:
    - plot_title: Custom title for the entire plot (e.g., "Training Set Analysis")
    """
    # Filter out intercept terms
    plottable_terms = []
    for i, term in enumerate(gam.terms):
        try:
            gam.generate_X_grid(term=i)
            plottable_terms.append(i)
        except ValueError as e:
            if "intercept" in str(e).lower():
                print(f"Skipping term {i} (intercept term)")
            else:
                print(f"Skipping term {i}: {e}")
    
    if term_indices is None:
        term_indices = plottable_terms
    else:
        term_indices = [i for i in term_indices if i in plottable_terms]
    
    if not term_indices:
        print("No plottable terms found!")
        return None
    
    n_plot = len(term_indices)
    print(f"Plotting {n_plot} terms: {term_indices}")
    
    # Create titles
    if titles is None:
        plot_titles = [f"Term {i}" for i in term_indices]
    else:
        plot_titles = []
        for idx, term_idx in enumerate(term_indices):
            if idx < len(titles):
                plot_titles.append(titles[idx])
            else:
                plot_titles.append(f"Term {term_idx}")
    
    # Create subplots - Plotly handles spacing beautifully
    fig = make_subplots(
        rows=n_plot, cols=1,
        subplot_titles=plot_titles,
        vertical_spacing=0.08,
        shared_xaxes=False
    )
    
    colors = px.colors.qualitative.Set1
    
    for idx, term_i in enumerate(term_indices):
        try:
            XX = gam.generate_X_grid(term=term_i)
            pdep, confi = gam.partial_dependence(term=term_i, X=XX, width=0.95)
            
            # Handle confidence intervals
            confi = np.asarray(confi)
            if confi.ndim == 1:
                lower = pdep - 0.1 * np.abs(pdep)
                upper = pdep + 0.1 * np.abs(pdep)
            else:
                lower = confi[:, 0]
                upper = confi[:, 1]
            
            x_vals = XX[:, term_i]
            
            # Add confidence interval
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([x_vals, x_vals[::-1]]),
                    y=np.concatenate([upper, lower[::-1]]),
                    fill='toself',
                    fillcolor=f'rgba(31, 119, 180, 0.3)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='95% CI',
                    hoverinfo='skip',
                    showlegend=(idx == 0)
                ),
                row=idx+1, col=1
            )
            
            # Add main effect line
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=pdep,
                    mode='lines',
                    line=dict(color='#1f77b4', width=3),
                    name='Partial Effect',
                    hovertemplate=f'<b>{plot_titles[idx]}</b><br>' +
                                  'Value: %{x:.3f}<br>' +
                                  'Effect: %{y:.3f}<br>' +
                                  '<extra></extra>',
                    showlegend=(idx == 0)
                ),
                row=idx+1, col=1
            )
            
            # Add EDOF annotation
            try:
                if hasattr(gam, 'statistics_') and 'edof' in gam.statistics_:
                    edof_values = gam.statistics_["edof"]
                    if np.isscalar(edof_values):
                        edof = edof_values
                    else:
                        edof = edof_values[term_i] if term_i < len(edof_values) else edof_values
                    
                    if n_plot == 1:
                        xref = "x domain"
                        yref = "y domain"
                    else:
                        xref = f"x{idx+1} domain"
                        yref = f"y{idx+1} domain"
                        
                    fig.add_annotation(
                        x=0.95, y=0.95,
                        xref=xref,
                        yref=yref,
                        text=f"EDOF: {edof:.2f}",
                        showarrow=False,
                        font=dict(size=10, color="darkblue"),
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="darkblue",
                        borderwidth=1
                    )
            except Exception as edof_error:
                print(f"Could not add EDOF annotation for term {term_i}: {edof_error}")
            
        except Exception as e:
            if n_plot == 1:
                xref = "x domain"
                yref = "y domain"
            else:
                xref = f"x{idx+1} domain"
                yref = f"y{idx+1} domain"
                
            fig.add_annotation(
                x=0.5, y=0.5,
                xref=xref,
                yref=yref,
                text=f"Error plotting term {term_i}:<br>{str(e)}",
                showarrow=False,
                font=dict(size=12, color="red")
            )
    
    # Update layout with custom title
    main_title = plot_title if plot_title else "GAM Partial Dependence Plots"
    
    fig.update_layout(
        height=300 * n_plot,
        title=dict(
            text=main_title,
            x=0.5,
            font=dict(size=18, family="Arial Black"),
            xanchor='center'
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        font=dict(family="Arial", size=11),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Update axes
    for i in range(n_plot):
        row_num = i + 1
        if n_plot == 1:
            fig.update_xaxes(
                title_text=f"Feature Value",
                gridcolor='lightgray',
                gridwidth=0.5
            )
            fig.update_yaxes(
                title_text="Partial Effect",
                gridcolor='lightgray',
                gridwidth=0.5
            )
        else:
            fig.update_xaxes(
                title_text=f"Feature Value" if i == n_plot-1 else "",
                gridcolor='lightgray',
                gridwidth=0.5,
                row=row_num, col=1
            )
            fig.update_yaxes(
                title_text="Partial Effect",
                gridcolor='lightgray',
                gridwidth=0.5,
                row=row_num, col=1
            )
    
    if save_path:
        if not save_path.endswith('.html'):
            html_path = save_path + '.html'
        else:
            html_path = save_path
            
        os.makedirs(os.path.dirname(html_path), exist_ok=True)
        fig.write_html(html_path)
        print(f"✅ Interactive plot saved to: {html_path}")
    
    fig.show()
    return fig


def plot_gam_summary_dashboard_plotly(gam, X_test, y_test, y_pred, titles=None, save_path=None, plot_title=None):
    """
    Interactive Plotly dashboard for GAM model diagnostics.
    
    Parameters:
    - plot_title: Custom title for the entire dashboard (e.g., "Training Set Model Diagnostics")
    """
    # Filter plottable terms
    plottable_terms = []
    for i, term in enumerate(gam.terms):
        try:
            gam.generate_X_grid(term=i)
            plottable_terms.append(i)
        except ValueError:
            pass
    
    n_terms = len(plottable_terms)
    
    # Create titles
    if titles is None:
        plot_titles = [f"Term {i}" for i in plottable_terms]
    else:
        plot_titles = []
        for idx, term_idx in enumerate(plottable_terms):
            if idx < len(titles):
                plot_titles.append(titles[idx])
            else:
                plot_titles.append(f"Term {term_idx}")
    
    # Calculate grid dimensions
    n_cols = 3
    partial_rows = int(np.ceil(n_terms / n_cols))
    total_rows = partial_rows + 1  # +1 for diagnostics
    
    # Create subplot titles
    subplot_titles = plot_titles + ['Predicted vs Actual', 'Residuals Distribution', 'Q-Q Plot']
    
    # Create subplots with perfect spacing
    fig = make_subplots(
        rows=total_rows, cols=n_cols,
        subplot_titles=subplot_titles,
        specs=[[{"secondary_y": False} for _ in range(n_cols)] for _ in range(total_rows)],
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )
    
    # Plot partial effects
    for plot_idx, term_idx in enumerate(plottable_terms):
        row = (plot_idx // n_cols) + 1
        col = (plot_idx % n_cols) + 1
        
        try:
            XX = gam.generate_X_grid(term=term_idx)
            pdep, confi = gam.partial_dependence(term=term_idx, X=XX, width=0.95)
            confi = np.asarray(confi)
            
            if confi.ndim == 1:
                lower = pdep - 0.1 * np.abs(pdep)
                upper = pdep + 0.1 * np.abs(pdep)
            else:
                lower = confi[:, 0]
                upper = confi[:, 1]
            
            x_vals = XX[:, term_idx]
            
            # Confidence interval
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([x_vals, x_vals[::-1]]),
                    y=np.concatenate([upper, lower[::-1]]),
                    fill='toself',
                    fillcolor='rgba(31, 119, 180, 0.3)',
                    line=dict(color='rgba(255,255,255,0)'),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=row, col=col
            )
            
            # Main effect
            fig.add_trace(
                go.Scatter(
                    x=x_vals, y=pdep,
                    mode='lines',
                    line=dict(color='#1f77b4', width=2.5),
                    showlegend=False,
                    hovertemplate=f'<b>{plot_titles[plot_idx]}</b><br>Value: %{{x:.3f}}<br>Effect: %{{y:.3f}}<extra></extra>'
                ),
                row=row, col=col
            )
            
        except Exception as e:
            fig.add_annotation(
                x=0.5, y=0.5,
                xref=f"x{(row-1)*n_cols + col} domain",
                yref=f"y{(row-1)*n_cols + col} domain",
                text=f"Error: {str(e)[:50]}...",
                showarrow=False,
                font=dict(size=10, color="red")
            )
    
    # Diagnostic plots
    diag_row = partial_rows + 1
    
    # Predicted vs Actual
    fig.add_trace(
        go.Scatter(
            x=y_test, y=y_pred,
            mode='markers',
            marker=dict(size=4, color='steelblue', opacity=0.6),
            showlegend=False,
            hovertemplate='Actual: %{x:.1f}<br>Predicted: %{y:.1f}<extra></extra>'
        ),
        row=diag_row, col=1
    )
    
    # Perfect prediction line
    min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode='lines',
            line=dict(color='red', dash='dash', width=2),
            showlegend=False,
            hoverinfo='skip'
        ),
        row=diag_row, col=1
    )
    
    # Residuals histogram
    residuals = y_test - y_pred
    fig.add_trace(
        go.Histogram(
            x=residuals,
            nbinsx=30,
            marker=dict(color='#2ca02c', opacity=0.7, line=dict(color='black', width=1)),
            showlegend=False,
            hovertemplate='Residual Range: %{x}<br>Count: %{y}<extra></extra>'
        ),
        row=diag_row, col=2
    )
    
    # Add mean line to histogram
    fig.add_vline(
        x=residuals.mean(),
        line=dict(color='red', dash='dash', width=2),
        row=diag_row, col=2
    )
    
    # Q-Q plot
    from scipy import stats
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm", plot=None)
    fig.add_trace(
        go.Scatter(
            x=osm, y=osr,
            mode='markers',
            marker=dict(size=4, color='steelblue'),
            showlegend=False,
            hovertemplate='Theoretical: %{x:.2f}<br>Observed: %{y:.2f}<extra></extra>'
        ),
        row=diag_row, col=3
    )
    
    # Q-Q line
    fig.add_trace(
        go.Scatter(
            x=osm, y=slope * osm + intercept,
            mode='lines',
            line=dict(color='red', width=2),
            showlegend=False,
            hoverinfo='skip'
        ),
        row=diag_row, col=3
    )
    
    # Update layout with custom title
    main_title = plot_title if plot_title else "GAM Model Summary Dashboard"
    
    fig.update_layout(
        height=400 * total_rows,
        title=dict(
            text=main_title,
            x=0.5,
            font=dict(size=20, family="Arial Black"),
            xanchor='center'
        ),
        font=dict(family="Arial", size=10),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Update all axes
    fig.update_xaxes(gridcolor='lightgray', gridwidth=0.5)
    fig.update_yaxes(gridcolor='lightgray', gridwidth=0.5)
    
    if save_path:
        if not save_path.endswith('.html'):
            html_path = save_path + '.html'
        else:
            html_path = save_path
            
        os.makedirs(os.path.dirname(html_path), exist_ok=True)
        fig.write_html(html_path)
        print(f"✅ Interactive dashboard saved to: {html_path}")
    
    fig.show()
    return fig


def plot_gam_feature_importance_plotly(gam, feature_names=None, save_path=None, plot_title=None):
    """
    Interactive Plotly version of GAM feature importance based on EDOF.
    
    Parameters:
    - plot_title: Custom title for the feature importance plot
    """
    try:
        if not hasattr(gam, 'statistics_') or 'edof' not in gam.statistics_:
            print("EDOF statistics not available for this GAM model.")
            return None
            
        edof = gam.statistics_['edof']
        
        # Filter out intercept terms
        plottable_terms = []
        plottable_edof = []
        for i, term in enumerate(gam.terms):
            try:
                gam.generate_X_grid(term=i)
                plottable_terms.append(i)
                plottable_edof.append(edof[i])
            except ValueError as e:
                if "intercept" not in str(e).lower():
                    print(f"Skipping term {i} due to error: {e}")
        
        if not plottable_terms:
            print("No plottable terms found for feature importance!")
            return None
        
        n_terms = len(plottable_terms)
        
        # Create labels - use feature names if provided
        if feature_names is not None and len(feature_names) >= n_terms:
            labels = [feature_names[i] for i in range(n_terms)]
        else:
            labels = [f'Term {i}' for i in plottable_terms]
        
        # Create horizontal bar chart
        fig = go.Figure(data=[
            go.Bar(
                y=labels,
                x=plottable_edof,
                orientation='h',
                marker=dict(
                    color=plottable_edof,
                    colorscale='Viridis',
                    colorbar=dict(title="EDOF Value"),
                    line=dict(color='black', width=1)
                ),
                text=[f'{val:.2f}' for val in plottable_edof],
                textposition='outside',
                textfont=dict(size=12, color='black'),
                hovertemplate='<b>%{y}</b><br>' +
                             'EDOF: %{x:.3f}<br>' +
                             '<extra></extra>'
            )
        ])
        
        # Update layout with custom title
        main_title = plot_title if plot_title else "GAM Feature Importance (Effective Degrees of Freedom)"
        
        fig.update_layout(
            title=dict(
                text=main_title,
                x=0.5,
                font=dict(size=16, family="Arial Black"),
                xanchor='center'
            ),
            xaxis_title="Effective Degrees of Freedom (EDOF)",
            yaxis_title="Features",
            font=dict(family="Arial", size=12),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=max(400, n_terms * 50),
            margin=dict(l=150, r=50, t=80, b=50)
        )
        
        fig.update_xaxes(gridcolor='lightgray', gridwidth=0.5)
        fig.update_yaxes(gridcolor='lightgray', gridwidth=0.5)
        
        if save_path:
            if not save_path.endswith('.html'):
                html_path = save_path + '.html'
            else:
                html_path = save_path
                
            os.makedirs(os.path.dirname(html_path), exist_ok=True)
            fig.write_html(html_path)
            print(f"✅ Interactive feature importance saved to: {html_path}")
        
        fig.show()
        return fig
        
    except Exception as e:
        print(f"❌ Could not plot feature importance: {e}")
        return None


def plot_gam_combined_dashboard_plotly(gam, X_train, y_train, y_train_pred, X_test, y_test, y_test_pred, titles=None, save_path=None, plot_title=None):
    """
    Interactive Plotly dashboard for GAM model diagnostics comparing BOTH training and test sets.
    
    Parameters:
    - gam: Fitted GAM model
    - X_train, y_train, y_train_pred: Training data, actuals, predictions
    - X_test, y_test, y_test_pred: Test data, actuals, predictions
    - titles: Feature names for partial dependence plots
    - save_path: Where to save HTML file
    - plot_title: Custom title for the entire dashboard
    """
    # Filter plottable terms
    plottable_terms = []
    for i, term in enumerate(gam.terms):
        try:
            gam.generate_X_grid(term=i)
            plottable_terms.append(i)
        except ValueError:
            pass
    
    n_terms = len(plottable_terms)
    
    # Create titles for partial dependence plots
    if titles is None:
        plot_titles = [f"Term {i}" for i in plottable_terms]
    else:
        plot_titles = []
        for idx, term_idx in enumerate(plottable_terms):
            if idx < len(titles):
                plot_titles.append(titles[idx])
            else:
                plot_titles.append(f"Term {term_idx}")
    
    # Calculate grid dimensions - partial dependence + 3 diagnostic plots
    n_cols = 3
    partial_rows = int(np.ceil(n_terms / n_cols))
    total_rows = partial_rows + 1  # +1 for diagnostics row
    
    # Create subplot titles
    diagnostic_titles = ['Train vs Test Predictions', 'Residuals Comparison', 'Q-Q Plot Comparison']
    subplot_titles = plot_titles + diagnostic_titles
    
    # Create subplots
    fig = make_subplots(
        rows=total_rows, cols=n_cols,
        subplot_titles=subplot_titles,
        specs=[[{"secondary_y": False} for _ in range(n_cols)] for _ in range(total_rows)],
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )
    
    # Plot partial effects (same as before - model-based)
    for plot_idx, term_idx in enumerate(plottable_terms):
        row = (plot_idx // n_cols) + 1
        col = (plot_idx % n_cols) + 1
        
        try:
            XX = gam.generate_X_grid(term=term_idx)
            pdep, confi = gam.partial_dependence(term=term_idx, X=XX, width=0.95)
            confi = np.asarray(confi)
            
            if confi.ndim == 1:
                lower = pdep - 0.1 * np.abs(pdep)
                upper = pdep + 0.1 * np.abs(pdep)
            else:
                lower = confi[:, 0]
                upper = confi[:, 1]
            
            x_vals = XX[:, term_idx]
            
            # Confidence interval
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([x_vals, x_vals[::-1]]),
                    y=np.concatenate([upper, lower[::-1]]),
                    fill='toself',
                    fillcolor='rgba(31, 119, 180, 0.3)',
                    line=dict(color='rgba(255,255,255,0)'),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=row, col=col
            )
            
            # Main effect line
            fig.add_trace(
                go.Scatter(
                    x=x_vals, y=pdep,
                    mode='lines',
                    line=dict(color='#1f77b4', width=2.5),
                    showlegend=False,
                    hovertemplate=f'<b>{plot_titles[plot_idx]}</b><br>Value: %{{x:.3f}}<br>Effect: %{{y:.3f}}<extra></extra>'
                ),
                row=row, col=col
            )
            
        except Exception as e:
            fig.add_annotation(
                x=0.5, y=0.5,
                xref=f"x{(row-1)*n_cols + col} domain",
                yref=f"y{(row-1)*n_cols + col} domain",
                text=f"Error: {str(e)[:50]}...",
                showarrow=False,
                font=dict(size=10, color="red")
            )
    
    # COMBINED DIAGNOSTIC PLOTS
    diag_row = partial_rows + 1
    
    # 1. COMBINED Predicted vs Actual (Train vs Test)
    # Training points
    fig.add_trace(
        go.Scatter(
            x=y_train, y=y_train_pred,
            mode='markers',
            marker=dict(size=4, color='blue', opacity=0.6),
            name='Training Set',
            hovertemplate='<b>Training</b><br>Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>'
        ),
        row=diag_row, col=1
    )
    
    # Test points
    fig.add_trace(
        go.Scatter(
            x=y_test, y=y_test_pred,
            mode='markers',
            marker=dict(size=4, color='red', opacity=0.6),
            name='Test Set',
            hovertemplate='<b>Test</b><br>Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>'
        ),
        row=diag_row, col=1
    )
    
    # Perfect prediction line
    all_actual = np.concatenate([y_train, y_test])
    all_pred = np.concatenate([y_train_pred, y_test_pred])
    min_val, max_val = min(all_actual.min(), all_pred.min()), max(all_actual.max(), all_pred.max())
    
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode='lines',
            line=dict(color='black', dash='dash', width=2),
            showlegend=False,
            hoverinfo='skip'
        ),
        row=diag_row, col=1
    )
    
    # 2. COMBINED Residuals histogram
    train_residuals = y_train - y_train_pred
    test_residuals = y_test - y_test_pred
    
    # Training residuals
    fig.add_trace(
        go.Histogram(
            x=train_residuals,
            nbinsx=25,
            opacity=0.7,
            marker=dict(color='blue'),
            name='Train Residuals',
            hovertemplate='<b>Training Residuals</b><br>Range: %{x}<br>Count: %{y}<extra></extra>'
        ),
        row=diag_row, col=2
    )
    
    # Test residuals
    fig.add_trace(
        go.Histogram(
            x=test_residuals,
            nbinsx=25,
            opacity=0.7,
            marker=dict(color='red'),
            name='Test Residuals',
            hovertemplate='<b>Test Residuals</b><br>Range: %{x}<br>Count: %{y}<extra></extra>'
        ),
        row=diag_row, col=2
    )
    
    # Mean lines
    fig.add_vline(x=train_residuals.mean(), line=dict(color='blue', dash='dash', width=2), row=diag_row, col=2)
    fig.add_vline(x=test_residuals.mean(), line=dict(color='red', dash='dash', width=2), row=diag_row, col=2)
    
    # 3. COMBINED Q-Q plots
    from scipy import stats
    
    # Training Q-Q
    (osm_train, osr_train), (slope_train, intercept_train, r_train) = stats.probplot(train_residuals, dist="norm", plot=None)
    fig.add_trace(
        go.Scatter(
            x=osm_train, y=osr_train,
            mode='markers',
            marker=dict(size=4, color='blue'),
            name='Train Q-Q',
            hovertemplate='<b>Training Q-Q</b><br>Theoretical: %{x:.2f}<br>Observed: %{y:.2f}<extra></extra>'
        ),
        row=diag_row, col=3
    )
    
    # Test Q-Q
    (osm_test, osr_test), (slope_test, intercept_test, r_test) = stats.probplot(test_residuals, dist="norm", plot=None)
    fig.add_trace(
        go.Scatter(
            x=osm_test, y=osr_test,
            mode='markers',
            marker=dict(size=4, color='red'),
            name='Test Q-Q',
            hovertemplate='<b>Test Q-Q</b><br>Theoretical: %{x:.2f}<br>Observed: %{y:.2f}<extra></extra>'
        ),
        row=diag_row, col=3
    )
    
    # Q-Q reference lines
    fig.add_trace(
        go.Scatter(
            x=osm_train, y=slope_train * osm_train + intercept_train,
            mode='lines',
            line=dict(color='blue', width=2),
            showlegend=False,
            hoverinfo='skip'
        ),
        row=diag_row, col=3
    )
    
    fig.add_trace(
        go.Scatter(
            x=osm_test, y=slope_test * osm_test + intercept_test,
            mode='lines',  
            line=dict(color='red', width=2),
            showlegend=False,
            hoverinfo='skip'
        ),
        row=diag_row, col=3
    )
    
    # Calculate metrics for title
    train_rmse = np.sqrt(np.mean(train_residuals**2))
    test_rmse = np.sqrt(np.mean(test_residuals**2))
    
    # Update layout
    main_title = plot_title if plot_title else f"GAM Model Analysis - Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f}"
    
    fig.update_layout(
        height=400 * total_rows,
        title=dict(
            text=main_title,
            x=0.5,
            font=dict(size=18, family="Arial Black"),
            xanchor='center'
        ),
        font=dict(family="Arial", size=10),
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(
            orientation="h",
            yanchor="bottom", 
            y=1.05,
            xanchor="center",
            x=0.5
        )
    )
    
    # Update all axes
    fig.update_xaxes(gridcolor='lightgray', gridwidth=0.5)
    fig.update_yaxes(gridcolor='lightgray', gridwidth=0.5)
    
    if save_path:
        if not save_path.endswith('.html'):
            html_path = save_path + '.html'
        else:
            html_path = save_path
            
        os.makedirs(os.path.dirname(html_path), exist_ok=True)
        fig.write_html(html_path)
        print(f"✅ Combined dashboard saved to: {html_path}")
    
    fig.show()
    return fig