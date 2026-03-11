import pytest
import numpy as np
from claimssimulator.features.feature_definition import FeatureDefinition
from claimssimulator.features.feature_spec import (
    Feature, FormulaFeature, Uniform, Categorical,
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


def test_formula_feature_categorical_parameter():
    """Test categorical parameter via column[param] syntax"""
    specs = [
        Feature('age', Uniform(low=25, high=55)),
        Feature('vehicle_type', Categorical(
            probabilities=[0.5, 0.3, 0.2],
            labels=['Sedan', 'SUV', 'Truck'],
        )),
        FormulaFeature(
            'risk',
            formula='{beta_age} * age + vehicle_type[b_vehtype]',
            parameters={
                'beta_age': 0.02,
                'b_vehtype': {'Sedan': 0.0, 'SUV': 0.3, 'Truck': 0.5},
            },
        ),
    ]

    gen = FeatureDefinition(specs)
    df = gen.generate(n_samples=500, random_seed=42)

    # Manually compute expected values
    cat_vals = np.where(
        df['vehicle_type'] == 'SUV', 0.3,
        np.where(df['vehicle_type'] == 'Truck', 0.5, 0.0)
    )
    expected = 0.02 * df['age'].values + cat_vals
    np.testing.assert_array_almost_equal(df['risk'].values, expected)


def test_formula_feature_categorical_with_mean():
    """Test categorical parameter with mean rescaling"""
    specs = [
        Feature('vehicle_type', Categorical(
            probabilities=[0.5, 0.3, 0.2],
            labels=['Sedan', 'SUV', 'Truck'],
        )),
        FormulaFeature(
            'risk',
            formula='vehicle_type[b_vehtype]',
            parameters={
                'b_vehtype': {'Sedan': 1.0, 'SUV': 2.0, 'Truck': 3.0},
            },
            mean=5.0,
        ),
    ]

    gen = FeatureDefinition(specs)
    df = gen.generate(n_samples=1000, random_seed=42)

    assert abs(df['risk'].mean() - 5.0) < 0.1


def test_formula_feature_categorical_missing_label_defaults_to_zero():
    """Labels not listed in the dict default to 0"""
    specs = [
        Feature('vehicle_type', Categorical(
            probabilities=[0.5, 0.3, 0.2],
            labels=['Sedan', 'SUV', 'Truck'],
        )),
        FormulaFeature(
            'risk',
            formula='vehicle_type[b_vehtype]',
            parameters={
                'b_vehtype': {'SUV': 0.3},  # Sedan and Truck → 0
            },
        ),
    ]

    gen = FeatureDefinition(specs)
    df = gen.generate(n_samples=500, random_seed=42)

    expected = np.where(df['vehicle_type'] == 'SUV', 0.3, 0.0)
    np.testing.assert_array_almost_equal(df['risk'].values, expected)


def test_formula_feature_categorical_dependency_detected():
    """The categorical column must be defined before the formula feature"""
    specs = [
        FormulaFeature(
            'risk',
            formula='vehicle_type[b_vehtype]',
            parameters={
                'b_vehtype': {'SUV': 0.3, 'Truck': 0.5},
            },
        ),
        Feature('vehicle_type', Categorical(
            probabilities=[0.5, 0.3, 0.2],
            labels=['Sedan', 'SUV', 'Truck'],
        )),
    ]

    with pytest.raises(ValueError, match="have not been defined yet"):
        FeatureDefinition(specs)


def test_formula_feature_categorical_bad_param_type():
    """Using column[param] with a non-dict param should raise TypeError"""
    specs = [
        Feature('vehicle_type', Categorical(
            probabilities=[0.5, 0.5],
            labels=['A', 'B'],
        )),
        FormulaFeature(
            'risk',
            formula='vehicle_type[b_vehtype]',
            parameters={'b_vehtype': 0.5},  # should be a dict
        ),
    ]

    gen = FeatureDefinition(specs)
    with pytest.raises(TypeError, match="must be a dict"):
        gen.generate(n_samples=10, random_seed=42)


def test_formula_feature_multiple_categoricals():
    """Multiple categorical terms in the same formula"""
    specs = [
        Feature('color', Categorical(
            probabilities=[0.5, 0.5],
            labels=['red', 'blue'],
        )),
        Feature('size', Categorical(
            probabilities=[0.5, 0.5],
            labels=['small', 'large'],
        )),
        FormulaFeature(
            'risk',
            formula='color[b_color] + size[b_size]',
            parameters={
                'b_color': {'red': 1.0, 'blue': 2.0},
                'b_size': {'small': 10.0, 'large': 20.0},
            },
        ),
    ]

    gen = FeatureDefinition(specs)
    df = gen.generate(n_samples=500, random_seed=42)

    color_val = np.where(df['color'] == 'red', 1.0, 2.0)
    size_val = np.where(df['size'] == 'small', 10.0, 20.0)
    np.testing.assert_array_almost_equal(df['risk'].values, color_val + size_val)