"""Tests for claimsimulator.visualization.feature_visualization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from claimsimulator.visualization.feature_visualization import (
    ASSOCIATION_LABELS,
    FeatureAnalysis,
    HistogramData,
    compute_feature_analysis,
    compute_histogram_data,
    analyze_feature,
    visualize_features,
    _STAT_FUNCTIONS,
    _hoeffding_d,
)


# ── fixtures ──────────────────────────────────────────────────────────


@pytest.fixture()
def simple_df() -> pd.DataFrame:
    """DataFrame with two simple numeric columns."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "A": rng.normal(10, 2, size=500),
        "B": rng.exponential(3, size=500),
    })


@pytest.fixture()
def mixed_df(simple_df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame that also has a non-numeric column."""
    df = simple_df.copy()
    df["label"] = "x"
    return df


@pytest.fixture()
def categorical_df() -> pd.DataFrame:
    """DataFrame with a categorical column."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "color": rng.choice(["red", "green", "blue"], size=300, p=[0.5, 0.3, 0.2]),
        "size": rng.choice(["S", "M", "L"], size=300),
    })


@pytest.fixture()
def correlated_df() -> pd.DataFrame:
    """DataFrame with known correlations for analysis tests."""
    rng = np.random.default_rng(99)
    n = 1000
    x = rng.normal(0, 1, n)
    return pd.DataFrame({
        "target": x,
        "pos_corr": x + rng.normal(0, 0.3, n),   # strong positive
        "neg_corr": -x + rng.normal(0, 0.3, n),   # strong negative
        "no_corr": rng.normal(5, 2, n),            # ~zero correlation
        "label": "cat",                             # non-numeric
    })


# ── compute_histogram_data ────────────────────────────────────────────


class TestComputeHistogramData:

    def test_returns_list_of_histogram_data(self, simple_df):
        result = compute_histogram_data(simple_df)
        assert isinstance(result, list)
        assert all(isinstance(r, HistogramData) for r in result)

    def test_default_all_columns(self, mixed_df):
        """Default (features=None) now includes ALL columns."""
        result = compute_histogram_data(mixed_df)
        names = [h.feature_name for h in result]
        assert names == ["A", "B", "label"]

    def test_categorical_detected(self, mixed_df):
        result = compute_histogram_data(mixed_df)
        by_name = {h.feature_name: h for h in result}
        assert not by_name["A"].is_categorical
        assert by_name["label"].is_categorical

    def test_explicit_features(self, simple_df):
        result = compute_histogram_data(simple_df, features=["B"])
        assert len(result) == 1
        assert result[0].feature_name == "B"

    def test_nbins(self, simple_df):
        for nbins in (10, 50):
            result = compute_histogram_data(simple_df, n_bins=nbins)
            assert len(result[0].counts) == nbins
            assert len(result[0].bin_edges) == nbins + 1

    def test_counts_sum_to_nrows(self, simple_df):
        result = compute_histogram_data(simple_df, n_bins=20)
        for h in result:
            assert h.counts.sum() == len(simple_df)

    def test_no_vlines_by_default(self, simple_df):
        result = compute_histogram_data(simple_df)
        for h in result:
            assert h.vlines == {}

    def test_stat_vlines(self, simple_df):
        result = compute_histogram_data(
            simple_df, features=["A"], vlines=["mean", "median", "perc5"]
        )
        vl = result[0].vlines
        assert set(vl.keys()) == {"mean", "median", "perc5"}
        # Sanity: mean close to 10
        assert 8 < vl["mean"] < 12

    def test_numeric_vlines(self, simple_df):
        result = compute_histogram_data(
            simple_df, features=["A"], vlines=[0, 5.5]
        )
        assert result[0].vlines == {"0": 0.0, "5.5": 5.5}

    def test_mixed_vlines(self, simple_df):
        result = compute_histogram_data(
            simple_df, features=["A"], vlines=["mean", 0]
        )
        vl = result[0].vlines
        assert "mean" in vl
        assert "0" in vl

    def test_unknown_stat_raises(self, simple_df):
        with pytest.raises(ValueError, match="Unknown vline stat"):
            compute_histogram_data(simple_df, vlines=["bogus"])

    def test_missing_feature_raises(self, simple_df):
        with pytest.raises(ValueError, match="not found"):
            compute_histogram_data(simple_df, features=["MISSING"])

    def test_bad_vline_type_raises(self, simple_df):
        with pytest.raises(TypeError, match="str or numeric"):
            compute_histogram_data(simple_df, vlines=[[1, 2]])

    def test_all_builtin_stats_work(self, simple_df):
        """Every key in _STAT_FUNCTIONS should produce a float."""
        result = compute_histogram_data(
            simple_df,
            features=["A"],
            vlines=list(_STAT_FUNCTIONS.keys()),
        )
        vl = result[0].vlines
        assert set(vl.keys()) == set(_STAT_FUNCTIONS.keys())
        for v in vl.values():
            assert isinstance(v, float)


# ── visualize_features ────────────────────────────────────────────────


class TestVisualizeFeatures:

    def test_returns_figure(self, simple_df):
        import matplotlib.figure

        fig = visualize_features(simple_df, n_bins=15)
        assert isinstance(fig, matplotlib.figure.Figure)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_correct_number_of_visible_axes(self, simple_df):
        import matplotlib.pyplot as plt

        fig = visualize_features(simple_df, ncols=2)
        visible = [ax for ax in fig.axes if ax.get_visible()]
        assert len(visible) == 2  # A and B
        plt.close(fig)

    def test_single_feature(self, simple_df):
        import matplotlib.pyplot as plt

        fig = visualize_features(simple_df, features=["A"])
        visible = [ax for ax in fig.axes if ax.get_visible()]
        assert len(visible) == 1
        plt.close(fig)

    def test_vlines_legend(self, simple_df):
        import matplotlib.pyplot as plt

        fig = visualize_features(
            simple_df, features=["A"], vlines=["mean", "perc95"]
        )
        ax = fig.axes[0]
        legend = ax.get_legend()
        assert legend is not None
        texts = [t.get_text() for t in legend.get_texts()]
        assert any("mean" in t for t in texts)
        plt.close(fig)

    def test_log_scales(self, simple_df):
        import matplotlib.pyplot as plt

        fig = visualize_features(
            simple_df, features=["B"], yscale="log", xscale="log",
        )
        ax = fig.axes[0]
        assert ax.get_yscale() == "log"
        assert ax.get_xscale() == "log"
        plt.close(fig)

    def test_empty_df_raises(self):
        df = pd.DataFrame()
        with pytest.raises(ValueError, match="No features"):
            visualize_features(df)

    def test_categorical_only_df(self):
        """A DF with only categorical columns should still plot."""
        import matplotlib.pyplot as plt

        df = pd.DataFrame({"label": ["a", "b", "c", "a", "b"]})
        fig = visualize_features(df)
        visible = [ax for ax in fig.axes if ax.get_visible()]
        assert len(visible) == 1
        plt.close(fig)

    def test_figsize_per_ax(self, simple_df):
        import matplotlib.pyplot as plt

        fig = visualize_features(
            simple_df, features=["A", "B"], ncols=1, figsize_per_ax=(5, 3),
        )
        w, h = fig.get_size_inches()
        assert w == pytest.approx(5, abs=0.1)
        assert h == pytest.approx(6, abs=0.1)  # 2 rows × 3
        plt.close(fig)

    def test_mixed_numeric_and_categorical(self, mixed_df):
        import matplotlib.pyplot as plt

        fig = visualize_features(mixed_df)
        visible = [ax for ax in fig.axes if ax.get_visible()]
        assert len(visible) == 3  # A, B, label
        plt.close(fig)

    def test_categorical_bar_chart(self, categorical_df):
        import matplotlib.pyplot as plt

        fig = visualize_features(categorical_df, features=["color"])
        ax = fig.axes[0]
        # Should have bar patches for each unique value
        bars = [p for p in ax.patches if hasattr(p, "get_height")]
        assert len(bars) > 0
        assert "color" in ax.get_title()
        plt.close(fig)

    def test_vlines_ignored_for_categorical(self, categorical_df):
        """vlines should not cause errors on categorical features."""
        import matplotlib.pyplot as plt

        fig = visualize_features(
            categorical_df, features=["color"], vlines=["mean"],
        )
        # Just verifying it didn't crash
        assert fig is not None
        plt.close(fig)


# ── compute_feature_analysis ──────────────────────────────────────────


class TestComputeFeatureAnalysis:

    def test_returns_feature_analysis(self, correlated_df):
        result = compute_feature_analysis(correlated_df, "target")
        assert isinstance(result, FeatureAnalysis)

    def test_histogram_populated(self, correlated_df):
        result = compute_feature_analysis(correlated_df, "target", n_bins=20)
        assert result.histogram.feature_name == "target"
        assert len(result.histogram.counts) == 20

    def test_vlines_forwarded(self, correlated_df):
        result = compute_feature_analysis(
            correlated_df, "target", vlines=["mean", "median"],
        )
        assert "mean" in result.histogram.vlines
        assert "median" in result.histogram.vlines

    def test_correlations_exclude_self(self, correlated_df):
        result = compute_feature_analysis(correlated_df, "target")
        assert "target" not in result.correlations.index

    def test_correlations_skip_non_numeric(self, correlated_df):
        result = compute_feature_analysis(correlated_df, "target")
        assert "label" not in result.correlations.index

    def test_correlation_sign(self, correlated_df):
        result = compute_feature_analysis(correlated_df, "target")
        assert result.correlations["pos_corr"] > 0.8
        assert result.correlations["neg_corr"] < -0.8

    def test_correlations_sorted_by_abs(self, correlated_df):
        result = compute_feature_analysis(correlated_df, "target")
        abs_vals = result.correlations.abs().values
        assert all(abs_vals[i] >= abs_vals[i + 1] for i in range(len(abs_vals) - 1))

    def test_correlate_with_subset(self, correlated_df):
        result = compute_feature_analysis(
            correlated_df, "target", correlate_with=["pos_corr"],
        )
        assert list(result.correlations.index) == ["pos_corr"]

    def test_correlate_with_ignores_non_numeric(self, correlated_df):
        result = compute_feature_analysis(
            correlated_df, "target", correlate_with=["pos_corr", "label"],
        )
        assert "label" not in result.correlations.index
        assert "pos_corr" in result.correlations.index

    def test_missing_feature_raises(self, correlated_df):
        with pytest.raises(ValueError, match="not found"):
            compute_feature_analysis(correlated_df, "MISSING")

    def test_non_numeric_feature_raises(self, correlated_df):
        with pytest.raises(ValueError, match="not numeric"):
            compute_feature_analysis(correlated_df, "label")

    def test_no_other_numeric_columns(self):
        df = pd.DataFrame({"x": [1, 2, 3], "label": ["a", "b", "c"]})
        result = compute_feature_analysis(df, "x")
        assert len(result.correlations) == 0

    # ── association parameter ─────────────────────────────────────

    def test_default_association_is_pearson(self, correlated_df):
        result = compute_feature_analysis(correlated_df, "target")
        assert result.association == "pearson"

    def test_spearman(self, correlated_df):
        result = compute_feature_analysis(
            correlated_df, "target", association="spearman",
        )
        assert result.association == "spearman"
        assert result.correlations["pos_corr"] > 0.8
        assert result.correlations["neg_corr"] < -0.8

    def test_kendall(self, correlated_df):
        result = compute_feature_analysis(
            correlated_df, "target", association="kendall",
        )
        assert result.association == "kendall"
        assert result.correlations["pos_corr"] > 0.5
        assert result.correlations["neg_corr"] < -0.5

    def test_mutual_info(self, correlated_df):
        result = compute_feature_analysis(
            correlated_df, "target", association="mutual_info",
        )
        assert result.association == "mutual_info"
        # MI is always ≥ 0; correlated features should have higher MI
        assert result.correlations["pos_corr"] > result.correlations["no_corr"]
        assert result.correlations["neg_corr"] > result.correlations["no_corr"]

    def test_hoeffding(self, correlated_df):
        result = compute_feature_analysis(
            correlated_df, "target", association="hoeffding",
        )
        assert result.association == "hoeffding"
        # Hoeffding's D should be highest for strongly dependent features
        assert result.correlations["pos_corr"] > result.correlations["no_corr"]

    def test_unknown_association_raises(self, correlated_df):
        with pytest.raises(ValueError, match="Unknown association"):
            compute_feature_analysis(
                correlated_df, "target", association="bogus",
            )

    def test_association_sorted_by_abs(self, correlated_df):
        for assoc in ASSOCIATION_LABELS:
            result = compute_feature_analysis(
                correlated_df, "target", association=assoc,
            )
            abs_vals = result.correlations.abs().values
            assert all(
                abs_vals[i] >= abs_vals[i + 1]
                for i in range(len(abs_vals) - 1)
            ), f"Not sorted for association={assoc}"

    def test_correlate_with_subset_spearman(self, correlated_df):
        result = compute_feature_analysis(
            correlated_df,
            "target",
            correlate_with=["pos_corr"],
            association="spearman",
        )
        assert list(result.correlations.index) == ["pos_corr"]


# ── analyze_feature ───────────────────────────────────────────────────


class TestAnalyzeFeature:

    def test_returns_figure(self, correlated_df):
        import matplotlib.figure
        import matplotlib.pyplot as plt

        fig = analyze_feature(correlated_df, "target")
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_two_axes(self, correlated_df):
        import matplotlib.pyplot as plt

        fig = analyze_feature(correlated_df, "target")
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_histogram_title(self, correlated_df):
        import matplotlib.pyplot as plt

        fig = analyze_feature(correlated_df, "target")
        assert "target" in fig.axes[0].get_title()
        plt.close(fig)

    def test_correlation_title(self, correlated_df):
        import matplotlib.pyplot as plt

        fig = analyze_feature(correlated_df, "target")
        title = fig.axes[1].get_title()
        assert "target" in title
        assert "Pearson" in title
        plt.close(fig)

    def test_vlines_legend(self, correlated_df):
        import matplotlib.pyplot as plt

        fig = analyze_feature(
            correlated_df, "target", vlines=["mean", "perc95"],
        )
        legend = fig.axes[0].get_legend()
        assert legend is not None
        plt.close(fig)

    def test_figsize(self, correlated_df):
        import matplotlib.pyplot as plt

        fig = analyze_feature(correlated_df, "target", figsize=(10, 4))
        w, h = fig.get_size_inches()
        assert w == pytest.approx(10, abs=0.1)
        assert h == pytest.approx(4, abs=0.1)
        plt.close(fig)

    def test_correlate_with_subset(self, correlated_df):
        import matplotlib.pyplot as plt

        fig = analyze_feature(
            correlated_df, "target", correlate_with=["pos_corr"],
        )
        # The correlation axis should only show one bar
        ax = fig.axes[1]
        # barh creates one Rectangle patch per bar
        bars = [p for p in ax.patches if hasattr(p, "get_width")]
        assert len(bars) == 1
        plt.close(fig)

    @pytest.mark.parametrize("assoc", list(ASSOCIATION_LABELS))
    def test_association_parameter(self, correlated_df, assoc):
        import matplotlib.pyplot as plt

        fig = analyze_feature(
            correlated_df, "target", association=assoc,
        )
        ax = fig.axes[1]
        expected_label = ASSOCIATION_LABELS[assoc]
        assert expected_label in ax.get_xlabel()
        assert expected_label in ax.get_title()
        plt.close(fig)

    def test_unknown_association_raises(self, correlated_df):
        with pytest.raises(ValueError, match="Unknown association"):
            analyze_feature(
                correlated_df, "target", association="bogus",
            )


# ── Categorical histogram data ────────────────────────────────────────


class TestCategoricalHistogramData:

    def test_categorical_labels_and_counts(self, categorical_df):
        result = compute_histogram_data(categorical_df, features=["color"])
        h = result[0]
        assert h.is_categorical
        assert h.bin_edges is None
        assert h.labels is not None
        assert len(h.labels) == len(h.counts)
        assert sum(h.counts) == len(categorical_df)

    def test_categorical_vlines_empty(self, categorical_df):
        """vlines are silently skipped for categorical features."""
        result = compute_histogram_data(
            categorical_df, features=["color"], vlines=["mean"],
        )
        assert result[0].vlines == {}

    def test_all_categorical(self, categorical_df):
        result = compute_histogram_data(categorical_df)
        assert len(result) == 2
        assert all(h.is_categorical for h in result)

    def test_mixed_types(self):
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "cat": ["a", "b", "a"],
        })
        result = compute_histogram_data(df)
        by_name = {h.feature_name: h for h in result}
        assert not by_name["x"].is_categorical
        assert by_name["cat"].is_categorical


# ── Hoeffding's D ─────────────────────────────────────────────────────


class TestHoeffdingD:

    def test_identical_returns_positive(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=200)
        d = _hoeffding_d(x, x)
        assert d > 0

    def test_independent_near_zero(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=500)
        y = rng.normal(size=500)
        d = _hoeffding_d(x, y)
        assert abs(d) < 0.05

    def test_short_array_returns_zero(self):
        assert _hoeffding_d(np.array([1, 2, 3]), np.array([4, 5, 6])) == 0.0

    def test_handles_nans(self):
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0])
        y = np.array([1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0])
        d = _hoeffding_d(x, y)
        assert isinstance(d, float)
