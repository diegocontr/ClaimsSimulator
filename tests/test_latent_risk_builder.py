import pytest
import numpy as np
from claimssimulator.features.feature_definition import FeatureDefinition
from claimssimulator.features.feature_spec import (
    Feature, FormulaFeature, Uniform,
)


def test_formula_feature_basic():
    """Test basic formula-based feature"""
    specs = [
        Feature('age', Uniform(low=25, high=55)),
        FormulaFeature(
            'age_risk',
            formula="{beta} * age",
            parameters={'beta': 0.02},
        ),
    ]

    gen = FeatureDefinition(specs)
    df = gen.generate(n_samples=100, random_seed=42)

    assert 'age_risk' in df.columns
    np.testing.assert_array_almost_equal(
        df['age_risk'].values, 0.02 * df['age'].values
    )


def test_formula_feature_with_mean():
    """Test formula feature with mean renormalization"""
    specs = [
        Feature('age', Uniform(low=25, high=55)),
        FormulaFeature(
            'age_risk',
            formula="{beta} * age",
            parameters={'beta': 0.02},
            mean=0.5,
        ),
    ]

    gen = FeatureDefinition(specs)
    df = gen.generate(n_samples=1000, random_seed=42)

    assert 'age_risk' in df.columns
    assert abs(df['age_risk'].mean() - 0.5) < 0.01


def test_formula_feature_multiple_terms():
    """Test formula with multiple features and parameters"""
    specs = [
        Feature('age', Uniform(low=25, high=55)),
        Feature('experience', Uniform(low=5, high=35)),
        Feature('accidents', Uniform(low=0, high=3)),
        FormulaFeature(
            'profile_risk',
            formula="{beta_age} * age + {beta_exp} * experience + {beta_acc} * accidents",
            parameters={
                'beta_age': 0.02,
                'beta_exp': -0.01,
                'beta_acc': 0.5,
            },
        ),
    ]

    gen = FeatureDefinition(specs)
    df = gen.generate(n_samples=100, random_seed=42)

    expected = 0.02 * df['age'] + (-0.01) * df['experience'] + 0.5 * df['accidents']
    np.testing.assert_array_almost_equal(
        df['profile_risk'].values, expected.values
    )


def test_formula_feature_dependency_order():
    """Test that formula features detect dependency issues"""
    specs = [
        FormulaFeature(
            'risk',
            formula="{beta} * age",
            parameters={'beta': 0.02},
        ),
        Feature('age', Uniform(low=25, high=55)),
    ]

    with pytest.raises(ValueError, match="have not been defined yet"):
        FeatureDefinition(specs)


def test_formula_feature_chained():
    """Test that a formula feature can depend on another formula feature"""
    specs = [
        Feature('age', Uniform(low=25, high=55)),
        FormulaFeature(
            'raw_risk',
            formula="{beta} * age",
            parameters={'beta': 0.02},
        ),
        FormulaFeature(
            'scaled_risk',
            formula="raw_risk * {scale}",
            parameters={'scale': 10.0},
        ),
    ]

    gen = FeatureDefinition(specs)
    df = gen.generate(n_samples=100, random_seed=42)

    np.testing.assert_array_almost_equal(
        df['scaled_risk'].values, df['raw_risk'].values * 10.0
    )
