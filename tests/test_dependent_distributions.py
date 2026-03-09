"""Tests for dependent distributions — distribution parameters that reference
other feature columns by name (string references).
"""

import math

import numpy as np
import pandas as pd
import pytest

from claimsimulator.features.feature_definition import FeatureDefinition
from claimsimulator.features.feature_spec import (
    Feature,
    DerivedFeature,
    Normal,
    LogNormal,
    LogNormalMeanStd,
    Gamma,
    GammaMeanStd,
    Beta,
    BetaMeanConcentration,
    Uniform,
    Exponential,
    Poisson,
    NegativeBinomial,
    Transform,
    DependentTransform,
    get_distribution_dependencies,
)


# ── get_distribution_dependencies helper ─────────────────────────────


class TestGetDistributionDependencies:
    """Tests for the `get_distribution_dependencies` utility."""

    def test_no_string_params(self):
        assert get_distribution_dependencies(Normal(loc=0, scale=1)) == set()
        assert get_distribution_dependencies(Gamma(shape=2, scale=3)) == set()

    def test_single_string_param(self):
        deps = get_distribution_dependencies(Normal(loc='mu_col', scale=1.0))
        assert deps == {'mu_col'}

    def test_multiple_string_params(self):
        deps = get_distribution_dependencies(Normal(loc='mu', scale='sigma'))
        assert deps == {'mu', 'sigma'}

    def test_mixed_param_types(self):
        deps = get_distribution_dependencies(Gamma(shape='k_col', scale=2.0))
        assert deps == {'k_col'}

    def test_all_distribution_types(self):
        """Spot-check that every distribution type is handled."""
        assert get_distribution_dependencies(LogNormal(mean='m', sigma=0.5)) == {'m'}
        assert get_distribution_dependencies(Uniform(low='lo', high='hi')) == {'lo', 'hi'}
        assert get_distribution_dependencies(Exponential(scale='s')) == {'s'}
        assert get_distribution_dependencies(Poisson(lam='rate')) == {'rate'}
        assert get_distribution_dependencies(NegativeBinomial(n='nn', p=0.5)) == {'nn'}
        assert get_distribution_dependencies(Beta(a='alpha', b=2.0)) == {'alpha'}


# ── Dependency ordering validation ───────────────────────────────────


class TestDependencyValidation:
    """Ensure that string-param dependencies are checked at init time."""

    def test_missing_dependency_raises(self):
        """Referencing a column that hasn't been defined yet must fail."""
        specs = [
            Feature('x', Normal(loc='undefined_column', scale=1.0)),
        ]
        with pytest.raises(ValueError, match="have not been defined yet"):
            FeatureDefinition(specs)

    def test_correct_ordering_accepted(self):
        """Dependency defined before the feature that uses it."""
        specs = [
            Feature('mu', Normal(loc=5.0, scale=1.0)),
            Feature('x', Normal(loc='mu', scale=1.0)),
        ]
        gen = FeatureDefinition(specs)
        df = gen.generate(n_samples=100, random_seed=42)
        assert 'x' in df.columns

    def test_reverse_ordering_rejected(self):
        specs = [
            Feature('x', Normal(loc='mu', scale=1.0)),
            Feature('mu', Normal(loc=5.0, scale=1.0)),
        ]
        with pytest.raises(ValueError, match="have not been defined yet"):
            FeatureDefinition(specs)


# ── Normal with dependent parameters ────────────────────────────────


class TestDependentNormal:

    def test_loc_from_column(self):
        """Normal(loc='mu_col') should sample each row around that row's mu."""
        specs = [
            Feature('mu_col', Uniform(low=10.0, high=10.0)),  # constant 10
            Feature('x', Normal(loc='mu_col', scale=0.01)),
        ]
        df = FeatureDefinition(specs).generate(n_samples=5_000, random_seed=42)
        assert abs(df['x'].mean() - 10.0) < 0.1

    def test_scale_from_column(self):
        """Normal(scale='sigma_col') should use per-row standard deviation."""
        specs = [
            Feature('sigma_col', Uniform(low=0.01, high=0.01)),  # tiny spread
            Feature('x', Normal(loc=100.0, scale='sigma_col')),
        ]
        df = FeatureDefinition(specs).generate(n_samples=5_000, random_seed=42)
        assert df['x'].std() < 0.1

    def test_both_params_from_columns(self):
        specs = [
            Feature('mu', Uniform(low=50.0, high=50.0)),
            Feature('sigma', Uniform(low=0.01, high=0.01)),
            Feature('x', Normal(loc='mu', scale='sigma')),
        ]
        df = FeatureDefinition(specs).generate(n_samples=5_000, random_seed=42)
        assert abs(df['x'].mean() - 50.0) < 0.1
        assert df['x'].std() < 0.1


# ── LogNormal with dependent parameters ─────────────────────────────


class TestDependentLogNormal:

    def test_mean_from_column(self):
        """LogNormal(mean='mu_col') should use per-row underlying mean."""
        specs = [
            Feature('mu_col', Uniform(low=3.0, high=3.0)),  # constant
            Feature('x', LogNormal(mean='mu_col', sigma=0.1)),
        ]
        df = FeatureDefinition(specs).generate(n_samples=10_000, random_seed=42)
        # E[X] = exp(mu + sigma^2/2) = exp(3 + 0.005) ≈ 20.19
        expected = math.exp(3.0 + 0.01 / 2)
        assert abs(df['x'].mean() - expected) < 1.0

    def test_sigma_from_column(self):
        specs = [
            Feature('sig', Uniform(low=0.05, high=0.05)),
            Feature('x', LogNormal(mean=3.0, sigma='sig')),
        ]
        df = FeatureDefinition(specs).generate(n_samples=10_000, random_seed=42)
        expected_mean = math.exp(3.0 + 0.05**2 / 2)
        assert abs(df['x'].mean() - expected_mean) < 1.0


# ── LogNormalMeanStd with dependent parameters ──────────────────────


class TestDependentLogNormalMeanStd:

    def test_mean_from_column(self):
        specs = [
            Feature('target_mean', Uniform(low=100.0, high=100.0)),
            Feature('x', LogNormalMeanStd(mean='target_mean', std=5.0)),
        ]
        df = FeatureDefinition(specs).generate(n_samples=50_000, random_seed=42)
        assert abs(df['x'].mean() - 100.0) < 2.0


# ── Gamma with dependent parameters ─────────────────────────────────


class TestDependentGamma:

    def test_scale_from_column(self):
        specs = [
            Feature('theta', Uniform(low=5.0, high=5.0)),
            Feature('x', Gamma(shape=2.0, scale='theta')),
        ]
        df = FeatureDefinition(specs).generate(n_samples=50_000, random_seed=42)
        # E[X] = shape * scale = 2 * 5 = 10
        assert abs(df['x'].mean() - 10.0) < 0.2

    def test_shape_from_column(self):
        specs = [
            Feature('k', Uniform(low=3.0, high=3.0)),
            Feature('x', Gamma(shape='k', scale=2.0)),
        ]
        df = FeatureDefinition(specs).generate(n_samples=50_000, random_seed=42)
        # E[X] = 3 * 2 = 6
        assert abs(df['x'].mean() - 6.0) < 0.2


# ── GammaMeanStd with dependent parameters ──────────────────────────


class TestDependentGammaMeanStd:

    def test_mean_from_column(self):
        specs = [
            Feature('target', Uniform(low=20.0, high=20.0)),
            Feature('x', GammaMeanStd(mean='target', std=5.0)),
        ]
        df = FeatureDefinition(specs).generate(n_samples=50_000, random_seed=42)
        assert abs(df['x'].mean() - 20.0) < 0.5


# ── Uniform with dependent parameters ───────────────────────────────


class TestDependentUniform:

    def test_high_from_column(self):
        """E.g. salary uniform between 30k and an age-dependent maximum."""
        specs = [
            Feature('max_val', Uniform(low=100.0, high=100.0)),
            Feature('x', Uniform(low=0.0, high='max_val')),
        ]
        df = FeatureDefinition(specs).generate(n_samples=50_000, random_seed=42)
        assert abs(df['x'].mean() - 50.0) < 1.0
        assert df['x'].max() <= 100.0 + 1e-10


# ── Exponential with dependent parameters ────────────────────────────


class TestDependentExponential:

    def test_scale_from_column(self):
        specs = [
            Feature('s', Uniform(low=5.0, high=5.0)),
            Feature('x', Exponential(scale='s')),
        ]
        df = FeatureDefinition(specs).generate(n_samples=50_000, random_seed=42)
        assert abs(df['x'].mean() - 5.0) < 0.2


# ── Poisson with dependent parameters ───────────────────────────────


class TestDependentPoisson:

    def test_lam_from_column(self):
        specs = [
            Feature('rate', Uniform(low=3.0, high=3.0)),
            Feature('x', Poisson(lam='rate')),
        ]
        df = FeatureDefinition(specs).generate(n_samples=50_000, random_seed=42)
        assert abs(df['x'].mean() - 3.0) < 0.1


# ── Beta with dependent parameters ──────────────────────────────────


class TestDependentBeta:

    def test_a_from_column(self):
        specs = [
            Feature('alpha', Uniform(low=5.0, high=5.0)),
            Feature('x', Beta(a='alpha', b=5.0)),
        ]
        df = FeatureDefinition(specs).generate(n_samples=50_000, random_seed=42)
        # E[X] = a / (a + b) = 5/10 = 0.5
        assert abs(df['x'].mean() - 0.5) < 0.01


# ── BetaMeanConcentration with dependent parameters ─────────────────


class TestDependentBetaMeanConcentration:

    def test_concentration_from_column(self):
        specs = [
            Feature('kappa', Uniform(low=100.0, high=100.0)),
            Feature('x', BetaMeanConcentration(mean=0.3, concentration='kappa')),
        ]
        df = FeatureDefinition(specs).generate(n_samples=50_000, random_seed=42)
        assert abs(df['x'].mean() - 0.3) < 0.01


# ── NegativeBinomial with dependent parameters ──────────────────────


class TestDependentNegativeBinomial:

    def test_n_from_column(self):
        specs = [
            Feature('nn', Uniform(low=10, high=10)),
            Feature('x', NegativeBinomial(n='nn', p=0.5)),
        ]
        df = FeatureDefinition(specs).generate(n_samples=50_000, random_seed=42)
        # E[X] = n*(1-p)/p = 10*0.5/0.5 = 10
        assert abs(df['x'].mean() - 10.0) < 0.5


# ── Combining dependent distribution + DependentTransform ───────────


class TestDependentDistributionWithTransform:

    def test_distribution_dep_and_transform(self):
        """A feature whose distribution AND transform both depend on others."""
        specs = [
            Feature('age', Uniform(low=25.0, high=65.0)),
            Feature('experience', Gamma(shape=2.0, scale='age'),
                    DependentTransform(
                        lambda x, age: np.clip(x, 0, age - 18),
                        dependencies=('age',),
                    )),
        ]
        df = FeatureDefinition(specs).generate(n_samples=5_000, random_seed=42)
        # Experience should not exceed age - 18
        assert (df['experience'] <= df['age'] - 18 + 1e-10).all()

    def test_distribution_dep_and_simple_transform(self):
        """Dependent distribution + simple Transform."""
        specs = [
            Feature('base', Uniform(low=5.0, high=5.0)),
            Feature('x', Normal(loc='base', scale=0.01),
                    Transform(lambda x: np.round(x))),
        ]
        df = FeatureDefinition(specs).generate(n_samples=1_000, random_seed=42)
        assert abs(df['x'].mean() - 5.0) < 0.5


# ── Varying dependent parameters (non-constant columns) ─────────────


class TestVaryingDependentParams:

    def test_heterogeneous_scale(self):
        """Rows with larger scale should have more spread."""
        specs = [
            Feature('spread', Uniform(low=0.1, high=10.0)),
            Feature('x', Normal(loc=0.0, scale='spread')),
        ]
        df = FeatureDefinition(specs).generate(n_samples=50_000, random_seed=42)

        # Split into low-spread and high-spread halves
        median_spread = df['spread'].median()
        low = df[df['spread'] < median_spread]['x']
        high = df[df['spread'] >= median_spread]['x']
        assert high.std() > low.std()

    def test_poisson_varying_rate(self):
        """Rows with higher rate should produce more counts."""
        specs = [
            Feature('rate', Uniform(low=0.5, high=10.0)),
            Feature('count', Poisson(lam='rate')),
        ]
        df = FeatureDefinition(specs).generate(n_samples=50_000, random_seed=42)

        median_rate = df['rate'].median()
        low_mean = df[df['rate'] < median_rate]['count'].mean()
        high_mean = df[df['rate'] >= median_rate]['count'].mean()
        assert high_mean > low_mean * 1.5  # substantial difference


# ── Realistic use case: insurance-style dependent features ───────────


class TestRealisticInsuranceExample:
    """End-to-end test inspired by the user's request."""

    def test_lognormal_with_dependent_mean(self):
        """
        csim.Feature('income', csim.LogNormal(mean='log_income_mu', sigma=0.3))
        where log_income_mu depends on age via a DerivedFeature.
        """
        specs = [
            Feature('age', Uniform(low=25.0, high=65.0)),
            DerivedFeature(
                'log_income_mu',
                function=lambda age: 9.5 + 0.02 * age,
                dependencies=('age',),
            ),
            Feature('income', LogNormal(mean='log_income_mu', sigma=0.3)),
        ]
        df = FeatureDefinition(specs).generate(n_samples=10_000, random_seed=42)
        assert 'income' in df.columns
        # Older people should have higher average income
        young = df[df['age'] < 40]['income'].mean()
        old = df[df['age'] >= 50]['income'].mean()
        assert old > young

    def test_gamma_claim_severity_depends_on_risk(self):
        """Claim severity drawn from Gamma whose scale depends on risk score."""
        specs = [
            Feature('risk_score', Uniform(low=1.0, high=5.0)),
            Feature('claim_severity', Gamma(shape=2.0, scale='risk_score')),
        ]
        df = FeatureDefinition(specs).generate(n_samples=50_000, random_seed=42)
        # Higher risk → higher severity on average
        low = df[df['risk_score'] < 2.5]['claim_severity'].mean()
        high = df[df['risk_score'] >= 3.5]['claim_severity'].mean()
        assert high > low


# ── Reproducibility ─────────────────────────────────────────────────


class TestDependentReproducibility:

    def test_same_seed_same_results(self):
        specs = [
            Feature('mu', Uniform(low=1.0, high=10.0)),
            Feature('x', Normal(loc='mu', scale=1.0)),
        ]
        df1 = FeatureDefinition(specs).generate(n_samples=500, random_seed=99)
        df2 = FeatureDefinition(specs).generate(n_samples=500, random_seed=99)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seed_different_results(self):
        specs = [
            Feature('mu', Uniform(low=1.0, high=10.0)),
            Feature('x', Normal(loc='mu', scale=1.0)),
        ]
        df1 = FeatureDefinition(specs).generate(n_samples=500, random_seed=1)
        df2 = FeatureDefinition(specs).generate(n_samples=500, random_seed=2)
        assert not df1['x'].equals(df2['x'])
