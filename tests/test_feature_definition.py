"""Tests for feature definition module"""

import numpy as np
import pandas as pd
import pytest
from claimssimulator.features.feature_definition import FeatureDefinition
from claimssimulator.features.feature_spec import (
    Feature,
    Normal,
    Gamma,
    GammaMeanStd,
    Beta,
    BetaMeanConcentration,
    LogNormal,
    LogNormalMeanStd,
    CorrelatedNormals,
    Uniform,
    Categorical,
)


def test_feature_definition_basic():
    """Test basic feature definition and generation"""
    specs = [
        Feature('age', Normal(loc=35, scale=15)),
        Feature('annual_mileage', Gamma(shape=2, scale=8000)),
    ]

    featuregen = FeatureDefinition(specs)
    df = featuregen.generate(n_samples=100, random_seed=42)

    assert df.shape[0] == 100
    assert df.shape[1] == 2
    assert 'age' in df.columns
    assert 'annual_mileage' in df.columns


def test_feature_definition_with_categorical():
    """Test feature definition with categorical features"""
    specs = [
        Feature('vehicle_type', Categorical(
            probabilities=[0.5, 0.3, 0.2],
            labels=['sedan', 'suv', 'truck'],
        )),
    ]

    featuregen = FeatureDefinition(specs)
    df = featuregen.generate(n_samples=100, random_seed=42)

    assert df.shape[0] == 100
    assert set(df['vehicle_type']).issubset({'sedan', 'suv', 'truck'})


def test_feature_definition_reproducibility():
    """Test that same seed produces same results"""
    specs = [
        Feature('age', Normal(loc=35, scale=15)),
    ]

    df1 = FeatureDefinition(specs).generate(n_samples=100, random_seed=42)
    df2 = FeatureDefinition(specs).generate(n_samples=100, random_seed=42)

    pd.testing.assert_frame_equal(df1, df2)


def test_feature_definition_get_names():
    """Test getting feature names"""
    specs = [
        Feature('age', Normal(loc=35, scale=15)),
        Feature('mileage', Gamma(shape=2, scale=8000)),
        Feature('vehicle_type', Categorical(probabilities=[0.5, 0.5])),
    ]

    featuregen = FeatureDefinition(specs)
    names = featuregen.get_feature_names()

    assert names == ['age', 'mileage', 'vehicle_type']


def test_duplicate_names_rejected():
    """Test that duplicate feature names raise an error"""
    specs = [
        Feature('age', Normal(loc=35, scale=15)),
        Feature('age', Uniform(low=18, high=65)),
    ]
    with pytest.raises(ValueError, match="Duplicate feature name"):
        FeatureDefinition(specs)


# ── GammaMeanStd tests ───────────────────────────────────────────────


def test_gamma_mean_std_basic():
    """Test GammaMeanStd generates samples with expected mean and std"""
    target_mean, target_std = 10.0, 3.0
    specs = [Feature('x', GammaMeanStd(mean=target_mean, std=target_std))]

    df = FeatureDefinition(specs).generate(n_samples=50_000, random_seed=42)

    assert abs(df['x'].mean() - target_mean) < 0.2
    assert abs(df['x'].std() - target_std) < 0.2
    assert (df['x'] > 0).all()


def test_gamma_mean_std_equivalence():
    """GammaMeanStd(mean=k*θ, std=√(k)*θ) should match Gamma(shape=k, scale=θ)"""
    k, theta = 4.0, 2.5
    mean = k * theta               # 10
    std = (k ** 0.5) * theta        # 5

    specs_native = [Feature('x', Gamma(shape=k, scale=theta))]
    specs_ms = [Feature('x', GammaMeanStd(mean=mean, std=std))]

    df_native = FeatureDefinition(specs_native).generate(n_samples=50_000, random_seed=99)
    df_ms = FeatureDefinition(specs_ms).generate(n_samples=50_000, random_seed=99)

    # Same underlying rng seed + same derived parameters → identical arrays
    np.testing.assert_array_almost_equal(df_native['x'].values, df_ms['x'].values)


def test_gamma_mean_std_invalid_mean():
    with pytest.raises(ValueError, match="mean > 0"):
        GammaMeanStd(mean=0, std=1.0)

    with pytest.raises(ValueError, match="mean > 0"):
        GammaMeanStd(mean=-5, std=1.0)


def test_gamma_mean_std_invalid_std():
    with pytest.raises(ValueError, match="std > 0"):
        GammaMeanStd(mean=5.0, std=0)

    with pytest.raises(ValueError, match="std > 0"):
        GammaMeanStd(mean=5.0, std=-1)


# ── LogNormalMeanStd tests ───────────────────────────────────────────


def test_lognormal_mean_std_basic():
    """Test LogNormalMeanStd generates samples with expected output mean and std"""
    target_mean, target_std = 100.0, 30.0
    specs = [Feature('x', LogNormalMeanStd(mean=target_mean, std=target_std))]

    df = FeatureDefinition(specs).generate(n_samples=100_000, random_seed=42)

    assert abs(df['x'].mean() - target_mean) < 1.5
    assert abs(df['x'].std() - target_std) < 1.5
    assert (df['x'] > 0).all()


def test_lognormal_mean_std_equivalence():
    """LogNormalMeanStd should produce the same samples as LogNormal with derived params"""
    import math
    target_mean, target_std = 50.0, 20.0

    sigma_u = math.sqrt(math.log(1 + (target_std / target_mean) ** 2))
    mu_u = math.log(target_mean) - sigma_u ** 2 / 2

    specs_native = [Feature('x', LogNormal(mean=mu_u, sigma=sigma_u))]
    specs_ms = [Feature('x', LogNormalMeanStd(mean=target_mean, std=target_std))]

    df_native = FeatureDefinition(specs_native).generate(n_samples=50_000, random_seed=7)
    df_ms = FeatureDefinition(specs_ms).generate(n_samples=50_000, random_seed=7)

    np.testing.assert_array_almost_equal(df_native['x'].values, df_ms['x'].values)


def test_lognormal_mean_std_invalid_mean():
    with pytest.raises(ValueError, match="mean > 0"):
        LogNormalMeanStd(mean=0, std=1.0)

    with pytest.raises(ValueError, match="mean > 0"):
        LogNormalMeanStd(mean=-10, std=1.0)


def test_lognormal_mean_std_invalid_std():
    with pytest.raises(ValueError, match="std > 0"):
        LogNormalMeanStd(mean=10.0, std=0)

    with pytest.raises(ValueError, match="std > 0"):
        LogNormalMeanStd(mean=10.0, std=-5)


# ── Beta tests ───────────────────────────────────────────────────────


def test_beta_basic():
    """Test Beta distribution generates samples in [0, 1] with expected moments"""
    a, b = 2.0, 5.0
    expected_mean = a / (a + b)
    specs = [Feature('x', Beta(a=a, b=b))]

    df = FeatureDefinition(specs).generate(n_samples=50_000, random_seed=42)

    assert (df['x'] >= 0).all() and (df['x'] <= 1).all()
    assert abs(df['x'].mean() - expected_mean) < 0.01


def test_beta_invalid_a():
    with pytest.raises(ValueError, match="a > 0"):
        Beta(a=0, b=1)

    with pytest.raises(ValueError, match="a > 0"):
        Beta(a=-1, b=1)


def test_beta_invalid_b():
    with pytest.raises(ValueError, match="b > 0"):
        Beta(a=1, b=0)

    with pytest.raises(ValueError, match="b > 0"):
        Beta(a=1, b=-2)


# ── BetaMeanConcentration tests ──────────────────────────────────────


def test_beta_mean_concentration_basic():
    """Test BetaMeanConcentration produces correct output mean"""
    target_mean = 0.3
    kappa = 20.0
    specs = [Feature('x', BetaMeanConcentration(mean=target_mean, concentration=kappa))]

    df = FeatureDefinition(specs).generate(n_samples=50_000, random_seed=42)

    assert (df['x'] >= 0).all() and (df['x'] <= 1).all()
    assert abs(df['x'].mean() - target_mean) < 0.01


def test_beta_mean_concentration_equivalence():
    """BetaMeanConcentration(mean, κ) should equal Beta(a=mean*κ, b=(1-mean)*κ)"""
    mean, kappa = 0.4, 15.0
    a = mean * kappa       # 6.0
    b = (1 - mean) * kappa  # 9.0

    specs_native = [Feature('x', Beta(a=a, b=b))]
    specs_mc = [Feature('x', BetaMeanConcentration(mean=mean, concentration=kappa))]

    df_native = FeatureDefinition(specs_native).generate(n_samples=50_000, random_seed=7)
    df_mc = FeatureDefinition(specs_mc).generate(n_samples=50_000, random_seed=7)

    np.testing.assert_array_almost_equal(df_native['x'].values, df_mc['x'].values)


def test_beta_mean_concentration_high_kappa():
    """High concentration should produce a tight distribution"""
    specs = [Feature('x', BetaMeanConcentration(mean=0.5, concentration=1000))]
    df = FeatureDefinition(specs).generate(n_samples=10_000, random_seed=42)

    assert df['x'].std() < 0.02  # very tight around 0.5


def test_beta_mean_concentration_invalid_mean():
    with pytest.raises(ValueError, match="0 < mean < 1"):
        BetaMeanConcentration(mean=0, concentration=10)

    with pytest.raises(ValueError, match="0 < mean < 1"):
        BetaMeanConcentration(mean=1, concentration=10)

    with pytest.raises(ValueError, match="0 < mean < 1"):
        BetaMeanConcentration(mean=-0.5, concentration=10)

    with pytest.raises(ValueError, match="0 < mean < 1"):
        BetaMeanConcentration(mean=1.5, concentration=10)


def test_beta_mean_concentration_invalid_concentration():
    with pytest.raises(ValueError, match="concentration > 0"):
        BetaMeanConcentration(mean=0.5, concentration=0)

    with pytest.raises(ValueError, match="concentration > 0"):
        BetaMeanConcentration(mean=0.5, concentration=-5)


# ── CorrelatedNormals tests ──────────────────────────────────────────


def test_correlated_normals_basic():
    """Test that CorrelatedNormals produces columns with correct means and stds"""
    specs = [
        CorrelatedNormals(
            names=('x1', 'x2'),
            means=(10.0, 20.0),
            stds=(2.0, 5.0),
            correlation=((1.0, 0.0),
                         (0.0, 1.0)),
        ),
    ]

    df = FeatureDefinition(specs).generate(n_samples=50_000, random_seed=42)

    assert 'x1' in df.columns and 'x2' in df.columns
    assert df.shape == (50_000, 2)
    assert abs(df['x1'].mean() - 10.0) < 0.1
    assert abs(df['x2'].mean() - 20.0) < 0.2
    assert abs(df['x1'].std() - 2.0) < 0.1
    assert abs(df['x2'].std() - 5.0) < 0.2


def test_correlated_normals_correlation():
    """Test that the empirical correlation matches the specified one"""
    rho = 0.8
    specs = [
        CorrelatedNormals(
            names=('a', 'b'),
            means=(0.0, 0.0),
            stds=(1.0, 1.0),
            correlation=((1.0, rho),
                         (rho, 1.0)),
        ),
    ]

    df = FeatureDefinition(specs).generate(n_samples=100_000, random_seed=42)
    empirical_corr = df['a'].corr(df['b'])
    assert abs(empirical_corr - rho) < 0.02


def test_correlated_normals_negative_correlation():
    """Test negative correlation"""
    rho = -0.6
    specs = [
        CorrelatedNormals(
            names=('x', 'y'),
            means=(5.0, 5.0),
            stds=(1.0, 1.0),
            correlation=((1.0, rho),
                         (rho, 1.0)),
        ),
    ]

    df = FeatureDefinition(specs).generate(n_samples=100_000, random_seed=42)
    empirical_corr = df['x'].corr(df['y'])
    assert abs(empirical_corr - rho) < 0.02


def test_correlated_normals_three_variables():
    """Test with 3 correlated variables"""
    specs = [
        CorrelatedNormals(
            names=('a', 'b', 'c'),
            means=(0.0, 10.0, -5.0),
            stds=(1.0, 3.0, 2.0),
            correlation=(
                (1.0, 0.5, 0.3),
                (0.5, 1.0, -0.2),
                (0.3, -0.2, 1.0),
            ),
        ),
    ]

    df = FeatureDefinition(specs).generate(n_samples=100_000, random_seed=7)

    assert df.shape == (100_000, 3)
    assert abs(df['a'].mean() - 0.0) < 0.05
    assert abs(df['b'].mean() - 10.0) < 0.05
    assert abs(df['c'].mean() - (-5.0)) < 0.05
    assert abs(df['a'].corr(df['b']) - 0.5) < 0.02
    assert abs(df['a'].corr(df['c']) - 0.3) < 0.02
    assert abs(df['b'].corr(df['c']) - (-0.2)) < 0.02


def test_correlated_normals_mixed_with_other_features():
    """Test CorrelatedNormals alongside regular Feature specs"""
    specs = [
        CorrelatedNormals(
            names=('x1', 'x2'),
            means=(0.0, 0.0),
            stds=(1.0, 1.0),
            correlation=((1.0, 0.5),
                         (0.5, 1.0)),
        ),
        Feature('z', Normal(loc=100, scale=10)),
    ]

    gen = FeatureDefinition(specs)
    df = gen.generate(n_samples=1_000, random_seed=42)

    assert list(df.columns) == ['x1', 'x2', 'z']
    assert gen.get_feature_names() == ['x1', 'x2', 'z']


def test_correlated_normals_reproducibility():
    """Same seed → identical output"""
    spec = CorrelatedNormals(
        names=('a', 'b'),
        means=(1.0, 2.0),
        stds=(0.5, 1.0),
        correlation=((1.0, 0.7), (0.7, 1.0)),
    )
    df1 = FeatureDefinition([spec]).generate(n_samples=500, random_seed=99)
    df2 = FeatureDefinition([spec]).generate(n_samples=500, random_seed=99)
    pd.testing.assert_frame_equal(df1, df2)


def test_correlated_normals_get_feature_names():
    specs = [
        CorrelatedNormals(
            names=('a', 'b', 'c'),
            means=(0.0, 0.0, 0.0),
            stds=(1.0, 1.0, 1.0),
            correlation=(
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
        ),
        Feature('d', Normal()),
    ]

    assert FeatureDefinition(specs).get_feature_names() == ['a', 'b', 'c', 'd']


# ── CorrelatedNormals validation tests ───────────────────────────────


def test_correlated_normals_mismatched_means_length():
    with pytest.raises(ValueError, match="means"):
        CorrelatedNormals(
            names=('a', 'b'),
            means=(1.0,),
            stds=(1.0, 1.0),
            correlation=((1.0, 0.0), (0.0, 1.0)),
        )


def test_correlated_normals_mismatched_stds_length():
    with pytest.raises(ValueError, match="stds"):
        CorrelatedNormals(
            names=('a', 'b'),
            means=(1.0, 2.0),
            stds=(1.0,),
            correlation=((1.0, 0.0), (0.0, 1.0)),
        )


def test_correlated_normals_nonpositive_std():
    with pytest.raises(ValueError, match="stds.*> 0"):
        CorrelatedNormals(
            names=('a', 'b'),
            means=(0.0, 0.0),
            stds=(1.0, 0.0),
            correlation=((1.0, 0.0), (0.0, 1.0)),
        )


def test_correlated_normals_wrong_matrix_shape():
    with pytest.raises(ValueError, match="rows"):
        CorrelatedNormals(
            names=('a', 'b'),
            means=(0.0, 0.0),
            stds=(1.0, 1.0),
            correlation=((1.0, 0.0),),  # only 1 row
        )


def test_correlated_normals_non_unit_diagonal():
    with pytest.raises(ValueError, match="Diagonal.*1.0"):
        CorrelatedNormals(
            names=('a', 'b'),
            means=(0.0, 0.0),
            stds=(1.0, 1.0),
            correlation=((0.9, 0.0), (0.0, 1.0)),
        )


def test_correlated_normals_asymmetric_matrix():
    with pytest.raises(ValueError, match="symmetric"):
        CorrelatedNormals(
            names=('a', 'b'),
            means=(0.0, 0.0),
            stds=(1.0, 1.0),
            correlation=((1.0, 0.5), (0.3, 1.0)),
        )


def test_correlated_normals_out_of_range_correlation():
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        CorrelatedNormals(
            names=('a', 'b'),
            means=(0.0, 0.0),
            stds=(1.0, 1.0),
            correlation=((1.0, 1.5), (1.5, 1.0)),
        )


def test_correlated_normals_duplicate_names():
    with pytest.raises(ValueError, match="duplicate"):
        CorrelatedNormals(
            names=('a', 'a'),
            means=(0.0, 0.0),
            stds=(1.0, 1.0),
            correlation=((1.0, 0.0), (0.0, 1.0)),
        )


def test_correlated_normals_duplicate_with_other_feature():
    """A name inside CorrelatedNormals clashes with another Feature"""
    specs = [
        Feature('x1', Normal()),
        CorrelatedNormals(
            names=('x1', 'x2'),
            means=(0.0, 0.0),
            stds=(1.0, 1.0),
            correlation=((1.0, 0.0), (0.0, 1.0)),
        ),
    ]
    with pytest.raises(ValueError, match="Duplicate feature name"):
        FeatureDefinition(specs)
