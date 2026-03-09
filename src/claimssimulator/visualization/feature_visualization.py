"""
Feature visualization: histogram computation and plotting.

This module separates **data computation** from **plotting** so that
callers can obtain histogram bins / statistics without importing
matplotlib.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd


# ── Stat-line specification ───────────────────────────────────────────

# Built-in stat names understood by compute_histogram_data / visualize_features
_STAT_FUNCTIONS: dict[str, callable] = {
    "mean": lambda arr: float(np.nanmean(arr)),
    "median": lambda arr: float(np.nanmedian(arr)),
    "perc5": lambda arr: float(np.nanpercentile(arr, 5)),
    "perc10": lambda arr: float(np.nanpercentile(arr, 10)),
    "perc25": lambda arr: float(np.nanpercentile(arr, 25)),
    "perc75": lambda arr: float(np.nanpercentile(arr, 75)),
    "perc90": lambda arr: float(np.nanpercentile(arr, 90)),
    "perc95": lambda arr: float(np.nanpercentile(arr, 95)),
    "std+": lambda arr: float(np.nanmean(arr) + np.nanstd(arr)),
    "std-": lambda arr: float(np.nanmean(arr) - np.nanstd(arr)),
}

_STAT_STYLES: dict[str, dict] = {
    "mean":   {"color": "red",       "linestyle": "--", "linewidth": 1.5},
    "median": {"color": "green",     "linestyle": "--", "linewidth": 1.5},
    "perc5":  {"color": "orange",    "linestyle": ":",  "linewidth": 1.2},
    "perc10": {"color": "orange",    "linestyle": ":",  "linewidth": 1.2},
    "perc25": {"color": "goldenrod", "linestyle": ":",  "linewidth": 1.2},
    "perc75": {"color": "goldenrod", "linestyle": ":",  "linewidth": 1.2},
    "perc90": {"color": "orange",    "linestyle": ":",  "linewidth": 1.2},
    "perc95": {"color": "orange",    "linestyle": ":",  "linewidth": 1.2},
    "std+":   {"color": "purple",    "linestyle": "-.", "linewidth": 1.0},
    "std-":   {"color": "purple",    "linestyle": "-.", "linewidth": 1.0},
}

_NUMERIC_STYLE: dict = {"color": "gray", "linestyle": "-", "linewidth": 1.0}


# ── Data container ────────────────────────────────────────────────────


@dataclass
class HistogramData:
    """Pre-computed histogram or value-count data for one feature.

    For **numeric** features ``bin_edges`` and ``counts`` are populated
    (standard histogram).  For **categorical** features ``labels`` and
    ``counts`` are populated instead, and ``bin_edges`` is ``None``.

    Attributes
    ----------
    feature_name : str
        Column name.
    bin_edges : np.ndarray | None
        Bin boundaries (length ``n_bins + 1``) for numeric features,
        or ``None`` for categorical features.
    counts : np.ndarray
        Count per bin (numeric) or per category (categorical).
    labels : list[str] | None
        Category labels for categorical features, ``None`` for numeric.
    is_categorical : bool
        ``True`` when the column was treated as categorical.
    vlines : dict[str, float]
        Vertical-line positions (only meaningful for numeric features).
    """

    feature_name: str
    bin_edges: np.ndarray | None
    counts: np.ndarray
    labels: list[str] | None = None
    is_categorical: bool = False
    vlines: dict[str, float] = field(default_factory=dict)


@dataclass
class FeatureAnalysis:
    """Pre-computed histogram and association data for a single feature.

    Attributes
    ----------
    histogram : HistogramData
        Histogram bins, counts, and vertical-line positions.
    correlations : pd.Series
        Association scores with every other numeric column, sorted
        descending by absolute value.  The target feature itself is
        **excluded**.  The kind of score depends on the *association*
        argument passed to :func:`compute_feature_analysis`.
    association : str
        Name of the association measure used (e.g. ``'pearson'``).
    """

    histogram: HistogramData
    correlations: pd.Series
    association: str = "pearson"


# ── Association measures ──────────────────────────────────────────────

# Supported association keywords → human-readable labels for axis titles.
ASSOCIATION_LABELS: dict[str, str] = {
    "pearson": "Correlation (Pearson)",
    "spearman": "Correlation (Spearman)",
    "kendall": "Correlation (Kendall τ)",
    "mutual_info": "Mutual Information",
    "hoeffding": "Hoeffding's D",
}


def _compute_association(
    df: pd.DataFrame,
    feature: str,
    candidates: list[str],
    association: str,
) -> pd.Series:
    """Return a Series of pairwise association scores.

    Parameters
    ----------
    df : pd.DataFrame
    feature : str
        Target column.
    candidates : list[str]
        Other numeric columns to compare against.
    association : str
        One of ``'pearson'``, ``'spearman'``, ``'kendall'``,
        ``'mutual_info'``, or ``'hoeffding'``.

    Returns
    -------
    pd.Series
        Scores indexed by column name, sorted descending by absolute
        value.
    """
    if not candidates:
        return pd.Series(dtype=float, name=association)

    match association:
        case "pearson":
            scores = df[candidates].corrwith(df[feature])
        case "spearman":
            scores = df[candidates].corrwith(df[feature], method="spearman")
        case "kendall":
            scores = df[candidates].corrwith(df[feature], method="kendall")
        case "mutual_info":
            scores = _mutual_info_scores(df, feature, candidates)
        case "hoeffding":
            scores = _hoeffding_scores(df, feature, candidates)
        case _:
            raise ValueError(
                f"Unknown association '{association}'. "
                f"Available: {sorted(ASSOCIATION_LABELS)}"
            )

    scores = scores.rename(association)
    return scores.iloc[scores.abs().values.argsort()[::-1]]


def _mutual_info_scores(
    df: pd.DataFrame,
    feature: str,
    candidates: list[str],
) -> pd.Series:
    """Estimate mutual information between *feature* and each candidate.

    Uses :func:`sklearn.feature_selection.mutual_info_regression` when
    available, otherwise falls back to a simple KDE-free binned estimator.
    """
    try:
        from sklearn.feature_selection import mutual_info_regression
    except ImportError:  # pragma: no cover – sklearn is optional
        raise ImportError(
            "mutual_info requires scikit-learn.  "
            "Install it with:  pip install scikit-learn"
        )

    target = df[feature].values.reshape(-1, 1)
    scores: dict[str, float] = {}
    for col in candidates:
        mi = mutual_info_regression(
            target,
            df[col].values,
            random_state=0,
        )
        scores[col] = float(mi[0])
    return pd.Series(scores)


def _hoeffding_scores(
    df: pd.DataFrame,
    feature: str,
    candidates: list[str],
) -> pd.Series:
    """Compute Hoeffding's D statistic for *feature* vs each candidate.

    Uses a pure-NumPy rank-based implementation so no extra dependencies
    are needed.
    """
    scores: dict[str, float] = {}
    x_vals = df[feature].values
    for col in candidates:
        scores[col] = _hoeffding_d(x_vals, df[col].values)
    return pd.Series(scores)


def _hoeffding_d(x: np.ndarray, y: np.ndarray) -> float:
    """Rank-based Hoeffding's D statistic for two numeric arrays.

    Reference
    ---------
    Hoeffding, W. (1948). "A Non-Parametric Test of Independence".
    *Annals of Mathematical Statistics*, 19(4), 546–557.
    """
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 5:
        return 0.0

    # Ranks (1-based, average ties)
    from scipy.stats import rankdata

    R = rankdata(x, method="average")
    S = rankdata(y, method="average")

    # Bivariate ranks: Q_i = #{j : x_j < x_i and y_j < y_i}
    # Use broadcasting for moderate n; for very large n a sort-based
    # algorithm would be faster, but this is fine for typical feature
    # analysis.
    Q = np.array([
        np.sum((x < x[i]) & (y < y[i]))
        for i in range(n)
    ], dtype=float)

    D1 = np.sum((Q * (Q - 1)))
    D2 = np.sum((R - 1) * (R - 2) * (S - 1) * (S - 2))
    D3 = np.sum((R - 2) * (S - 2) * (Q - 1))

    D = (
        30 * ((n - 2) * (n - 3) * D1 + D2 - 2 * (n - 2) * D3)
        / (n * (n - 1) * (n - 2) * (n - 3) * (n - 4))
    )
    return float(D)


# ── Pure data computation ─────────────────────────────────────────────


def compute_histogram_data(
    df: pd.DataFrame,
    *,
    features: Sequence[str] | None = None,
    n_bins: int = 30,
    vlines: Sequence[str | float | int] | None = None,
) -> list[HistogramData]:
    """Compute histogram bins and summary statistics (no plotting).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the generated features.
    features : list[str] | None
        Columns to include.  ``None`` → **all** columns (numeric *and*
        categorical).
    n_bins : int
        Number of histogram bins for numeric features (default 30).
        Ignored for categorical features.
    vlines : list[str | float | int] | None
        Vertical-line specifications (only applied to numeric features).

        * A **string** referencing a built-in statistic:
          ``'mean'``, ``'median'``, ``'perc5'``, ``'perc10'``,
          ``'perc25'``, ``'perc75'``, ``'perc90'``, ``'perc95'``,
          ``'std+'``, ``'std-'``.
        * A **numeric** literal (int / float) drawn as a fixed line.

        Default is ``None`` (no vertical lines).

    Returns
    -------
    list[HistogramData]
        One entry per feature, in the order they appear in *features*
        (or in the DataFrame column order when *features* is ``None``).
    """
    if features is None:
        features = list(df.columns)

    if vlines is None:
        vlines = []

    numeric_cols = set(df.select_dtypes(include=[np.number]).columns)

    results: list[HistogramData] = []

    for feat in features:
        if feat not in df.columns:
            raise ValueError(f"Feature '{feat}' not found in DataFrame")

        if feat in numeric_cols:
            # ── Numeric feature: standard histogram ──────────────
            values = df[feat].dropna().values
            counts, bin_edges = np.histogram(values, bins=n_bins)

            computed_vlines: dict[str, float] = {}
            for spec in vlines:
                if isinstance(spec, str):
                    key = spec.lower()
                    if key not in _STAT_FUNCTIONS:
                        raise ValueError(
                            f"Unknown vline stat '{spec}'. "
                            f"Available: {sorted(_STAT_FUNCTIONS)}"
                        )
                    computed_vlines[key] = _STAT_FUNCTIONS[key](values)
                elif isinstance(spec, (int, float)):
                    computed_vlines[str(spec)] = float(spec)
                else:
                    raise TypeError(
                        f"vlines elements must be str or numeric, "
                        f"got {type(spec)}"
                    )

            results.append(
                HistogramData(
                    feature_name=feat,
                    bin_edges=bin_edges,
                    counts=counts,
                    vlines=computed_vlines,
                )
            )
        else:
            # ── Categorical feature: value counts ────────────────
            vc = df[feat].value_counts()
            results.append(
                HistogramData(
                    feature_name=feat,
                    bin_edges=None,
                    counts=vc.values,
                    labels=list(vc.index.astype(str)),
                    is_categorical=True,
                )
            )

    return results


# ── Plotting ──────────────────────────────────────────────────────────


def visualize_features(
    df: pd.DataFrame,
    *,
    features: Sequence[str] | None = None,
    n_bins: int = 30,
    vlines: Sequence[str | float | int] | None = None,
    yscale: str = "linear",
    xscale: str = "linear",
    ncols: int = 2,
    figsize_per_ax: tuple[float, float] = (6, 4),
) -> "matplotlib.figure.Figure":  # noqa: F821 – lazy import
    """Plot histograms / bar charts for selected features.

    Numeric features are shown as histograms with optional stat vertical
    lines.  Categorical (non-numeric) features are shown as bar charts
    of value counts.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the generated features.
    features : list[str] | None
        Columns to plot.  ``None`` → **all** columns (numeric *and*
        categorical).
    n_bins : int
        Number of histogram bins for numeric features (default 30).
    vlines : list[str | float | int] | None
        Vertical-line specifications (see :func:`compute_histogram_data`).
        Only applied to numeric features.  Default is ``None``.
    yscale : str
        Matplotlib y-axis scale (default ``'linear'``).
    xscale : str
        Matplotlib x-axis scale (default ``'linear'``).  Only applied to
        numeric features.
    ncols : int
        Number of subplot columns in the grid (default 2).
    figsize_per_ax : tuple[float, float]
        ``(width, height)`` per subplot (default ``(6, 4)``).

    Returns
    -------
    matplotlib.figure.Figure
        The figure object (call ``plt.show()`` yourself if needed).
    """
    import matplotlib.pyplot as plt

    histograms = compute_histogram_data(
        df, features=features, n_bins=n_bins, vlines=vlines,
    )

    n_features = len(histograms)
    if n_features == 0:
        raise ValueError("No features to plot")

    nrows = -(-n_features // ncols)  # ceiling division
    fig_w = figsize_per_ax[0] * ncols
    fig_h = figsize_per_ax[1] * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))

    # Ensure axes is always a flat array
    if n_features == 1:
        axes = np.array([axes])
    axes = np.asarray(axes).ravel()

    for idx, hdata in enumerate(histograms):
        ax = axes[idx]

        if hdata.is_categorical:
            # ── Categorical: vertical bar chart of value counts ──
            ax.bar(
                hdata.labels,
                hdata.counts,
                edgecolor="black",
                alpha=0.7,
            )
            ax.set_title(
                hdata.feature_name, fontsize=11, fontweight="bold",
            )
            ax.set_xlabel(hdata.feature_name)
            ax.set_ylabel("Count")
            ax.set_yscale(yscale)
            ax.tick_params(axis="x", rotation=45)
        else:
            # ── Numeric: histogram from pre-computed bins ────────
            widths = np.diff(hdata.bin_edges)
            ax.bar(
                hdata.bin_edges[:-1],
                hdata.counts,
                width=widths,
                align="edge",
                edgecolor="black",
                alpha=0.7,
            )

            # Vertical lines
            for label, xpos in hdata.vlines.items():
                style = _STAT_STYLES.get(label, _NUMERIC_STYLE).copy()
                ax.axvline(xpos, **style, label=f"{label}: {xpos:.3g}")

            ax.set_title(
                hdata.feature_name, fontsize=11, fontweight="bold",
            )
            ax.set_xlabel(hdata.feature_name)
            ax.set_ylabel("Frequency")
            ax.set_yscale(yscale)
            ax.set_xscale(xscale)

            if hdata.vlines:
                ax.legend(fontsize=8)

    # Hide unused axes
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)

    fig.tight_layout()
    return fig


# ── Single-feature analysis (data) ───────────────────────────────────


def compute_feature_analysis(
    df: pd.DataFrame,
    feature: str,
    *,
    n_bins: int = 30,
    vlines: Sequence[str | float | int] | None = None,
    correlate_with: Sequence[str] | None = None,
    association: str = "pearson",
) -> FeatureAnalysis:
    """Compute histogram + association scores for a single feature (no plotting).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the generated features.
    feature : str
        The target column to analyse.
    n_bins : int
        Number of histogram bins (default 30).
    vlines : list[str | float | int] | None
        Vertical-line specifications (see :func:`compute_histogram_data`).
        Default is ``None``.
    correlate_with : list[str] | None
        Other columns to compute associations against.  ``None`` → all
        other numeric columns.  Non-numeric columns are silently skipped.
    association : str
        Association measure to use.  One of ``'pearson'`` (default),
        ``'spearman'``, ``'kendall'``, ``'mutual_info'``, or
        ``'hoeffding'``.

    Returns
    -------
    FeatureAnalysis
    """
    if association not in ASSOCIATION_LABELS:
        raise ValueError(
            f"Unknown association '{association}'. "
            f"Available: {sorted(ASSOCIATION_LABELS)}"
        )

    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' not found in DataFrame")

    # Associations need a numeric target
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)

    if feature not in numeric_cols:
        raise ValueError(
            f"Feature '{feature}' is not numeric — cannot compute associations"
        )

    # Histogram
    hist = compute_histogram_data(
        df, features=[feature], n_bins=n_bins, vlines=vlines,
    )[0]

    # Association scores
    if correlate_with is not None:
        candidates = [
            c for c in correlate_with
            if c in numeric_cols and c != feature
        ]
    else:
        candidates = [c for c in numeric_cols if c != feature]

    scores = _compute_association(df, feature, candidates, association)

    return FeatureAnalysis(
        histogram=hist, correlations=scores, association=association,
    )


# ── Single-feature analysis (plotting) ───────────────────────────────


def analyze_feature(
    df: pd.DataFrame,
    feature: str,
    *,
    n_bins: int = 30,
    vlines: Sequence[str | float | int] | None = None,
    correlate_with: Sequence[str] | None = None,
    association: str = "pearson",
    yscale: str = "linear",
    xscale: str = "linear",
    figsize: tuple[float, float] = (16, 6),
) -> "matplotlib.figure.Figure":  # noqa: F821 – lazy import
    """Plot a two-panel figure: histogram + association bar chart.

    Left panel
        Histogram of *feature* with optional stat vertical lines.
    Right panel
        Horizontal bar chart of association scores between *feature*
        and every other numeric column (green = positive, red = negative).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the generated features.
    feature : str
        The target column to analyse.
    n_bins : int
        Number of histogram bins (default 30).
    vlines : list[str | float | int] | None
        Vertical-line specifications (see :func:`compute_histogram_data`).
        Default ``None``.
    correlate_with : list[str] | None
        Columns to compute associations against.  ``None`` → all other
        numeric columns.
    association : str
        Association measure: ``'pearson'`` (default), ``'spearman'``,
        ``'kendall'``, ``'mutual_info'``, or ``'hoeffding'``.
    yscale / xscale : str
        Matplotlib axis scales for the histogram (default ``'linear'``).
    figsize : tuple[float, float]
        Overall figure size (default ``(16, 6)``).

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    analysis = compute_feature_analysis(
        df,
        feature,
        n_bins=n_bins,
        vlines=vlines,
        correlate_with=correlate_with,
        association=association,
    )

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # ── Left: histogram ──────────────────────────────────────────
    hdata = analysis.histogram
    widths = np.diff(hdata.bin_edges)
    axes[0].bar(
        hdata.bin_edges[:-1],
        hdata.counts,
        width=widths,
        align="edge",
        edgecolor="black",
        alpha=0.7,
        color="steelblue",
    )

    for label, xpos in hdata.vlines.items():
        style = _STAT_STYLES.get(label, _NUMERIC_STYLE).copy()
        axes[0].axvline(xpos, **style, label=f"{label}: {xpos:.3g}")

    axes[0].set_title(
        f"Distribution of {feature}", fontsize=12, fontweight="bold",
    )
    axes[0].set_xlabel(feature)
    axes[0].set_ylabel("Frequency")
    axes[0].set_yscale(yscale)
    axes[0].set_xscale(xscale)

    if hdata.vlines:
        axes[0].legend(fontsize=9)

    # ── Right: association bar chart ─────────────────────────────
    assoc_label = ASSOCIATION_LABELS.get(association, association)
    corr = analysis.correlations
    if len(corr) > 0:
        colors = ["green" if v > 0 else "red" for v in corr.values]
        axes[1].barh(
            corr.index, corr.values,
            color=colors, alpha=0.7, edgecolor="black",
        )
        axes[1].axvline(0, color="black", linestyle="-", linewidth=0.5)
    axes[1].set_title(
        f"{assoc_label} with {feature}",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].set_xlabel(assoc_label)

    fig.tight_layout()
    return fig
