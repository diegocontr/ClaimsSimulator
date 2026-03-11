"""Tests for feature transformations"""

import numpy as np
import pytest
from claimssimulator.features.feature_definition import FeatureDefinition
from claimssimulator.features.feature_spec import (
    Feature,
    DerivedFeature,
    Normal,
    LogNormal,
    Gamma,
    Uniform,
    Categorical,
    Transform,
    DependentTransform,
)


def test_simple_transformation():
    """Test simple transformation without dependencies"""
    specs = [
        Feature('age', LogNormal(mean=3.5, sigma=0.2),
                Transform(lambda x: x + 18)),
    ]

    featuregen = FeatureDefinition(specs)
    df = featuregen.generate(n_samples=1000, random_seed=42)

    # Age should be shifted by 18 (lognormal is always positive)
    assert df['age'].min() > 18
    assert df.shape == (1000, 1)


def test_dependent_transformation():
    """Test transformation with dependencies"""
    specs = [
        Feature('age', Normal(loc=35, scale=15),
                Transform(lambda x: np.abs(x) + 18)),
        Feature('experience', Gamma(shape=2, scale=5),
                DependentTransform(
                    lambda x, age: np.clip(x, 0, age - 18),
                    dependencies=('age',),
                )),
    ]

    featuregen = FeatureDefinition(specs)
    df = featuregen.generate(n_samples=1000, random_seed=42)

    # Experience should never exceed age - 18
    assert (df['experience'] <= df['age'] - 18).all()
    assert (df['experience'] >= 0).all()
    assert df.shape == (1000, 2)


def test_multiple_dependencies():
    """Test transformation with multiple dependencies"""
    specs = [
        Feature('base', Uniform(low=10, high=20)),
        Feature('multiplier', Uniform(low=1, high=2)),
        Feature('combined', Uniform(low=0, high=1),
                DependentTransform(
                    lambda x, b, m: x + b * m,
                    dependencies=('base', 'multiplier'),
                )),
    ]

    featuregen = FeatureDefinition(specs)
    df = featuregen.generate(n_samples=100, random_seed=42)

    # Combined should be roughly x + base * multiplier
    # Since combined also has its own random component (0-1),
    # we just check shape and that it exists
    assert df.shape == (100, 3)
    assert 'combined' in df.columns


def test_dependency_order():
    """Test that dependencies must be defined in correct order"""
    # Correct order: a, then b (depends on a), then c (depends on a and b)
    specs = [
        Feature('a', Uniform(low=0, high=1)),
        Feature('b', Uniform(low=0, high=1),
                DependentTransform(lambda x, a: x + a, dependencies=('a',))),
        Feature('c', Uniform(low=0, high=1),
                DependentTransform(lambda x, a, b: x + a + b, dependencies=('a', 'b'))),
    ]

    featuregen = FeatureDefinition(specs)
    df = featuregen.generate(n_samples=100, random_seed=42)

    assert df.shape == (100, 3)
    assert set(df.columns) == {'a', 'b', 'c'}


def test_wrong_dependency_order():
    """Test that wrong order raises error at construction time"""
    specs = [
        Feature('c', Uniform(low=0, high=1),
                DependentTransform(lambda x, a, b: x + a + b, dependencies=('a', 'b'))),
        Feature('b', Uniform(low=0, high=1),
                DependentTransform(lambda x, a: x + a, dependencies=('a',))),
        Feature('a', Uniform(low=0, high=1)),
    ]

    with pytest.raises(ValueError, match="have not been defined yet"):
        FeatureDefinition(specs)


def test_circular_dependency():
    """Test that forward dependencies raise error"""
    specs = [
        Feature('a', Uniform(low=0, high=1),
                DependentTransform(lambda x, b: x + b, dependencies=('b',))),
        Feature('b', Uniform(low=0, high=1)),
    ]

    with pytest.raises(ValueError, match="have not been defined yet"):
        FeatureDefinition(specs)


def test_missing_dependency():
    """Test that missing dependencies are detected"""
    specs = [
        Feature('a', Uniform(low=0, high=1),
                DependentTransform(lambda x, b: x + b, dependencies=('b',))),
    ]

    with pytest.raises(ValueError, match="have not been defined yet"):
        FeatureDefinition(specs)


def test_mixed_features():
    """Test mixing basic, transformed, and dependent features"""
    specs = [
        Feature('basic', Normal(loc=0, scale=1)),
        Feature('transformed', Normal(loc=0, scale=1),
                Transform(lambda x: x ** 2)),
        Feature('dependent', Normal(loc=0, scale=1),
                DependentTransform(lambda x, basic: x + basic, dependencies=('basic',))),
    ]

    featuregen = FeatureDefinition(specs)
    df = featuregen.generate(n_samples=1000, random_seed=42)

    assert df.shape == (1000, 3)
    assert set(df.columns) == {'basic', 'transformed', 'dependent'}
    # Transformed should be non-negative (squared)
    assert (df['transformed'] >= 0).all()


def test_complex_example():
    """Test the complex example from the notebook"""
    specs = [
        Feature('age', LogNormal(mean=3.5, sigma=0.2),
                Transform(lambda x: x + 18)),
        Feature('gender', Categorical(
            probabilities=[0.5, 0.5],
            labels=['Male', 'Female'],
        )),
        Feature('driving_experience_years', Gamma(shape=2, scale=5),
                DependentTransform(
                    lambda x, age: np.clip(x, 0, age - 18),
                    dependencies=('age',),
                )),
    ]

    featuregen = FeatureDefinition(specs)
    df = featuregen.generate(n_samples=5000, random_seed=42)

    # Verify constraints
    assert (df['age'] >= 18).all()
    assert (df['driving_experience_years'] >= 0).all()
    assert (df['driving_experience_years'] <= df['age'] - 18).all()
    assert set(df['gender'].unique()).issubset({'Male', 'Female'})
    assert df.shape == (5000, 3)


def test_derived_feature():
    """Test pure derived features (no distribution, just function of other features)"""
    specs = [
        Feature('age', LogNormal(mean=3.5, sigma=0.2),
                Transform(lambda x: x + 18)),
        Feature('driving_experience_years', Gamma(shape=2, scale=5),
                DependentTransform(
                    lambda x, age: np.clip(x, 0, age - 18),
                    dependencies=('age',),
                )),
        DerivedFeature('age_at_first_license',
                       function=lambda age, exp: age - exp,
                       dependencies=('age', 'driving_experience_years')),
    ]

    featuregen = FeatureDefinition(specs)
    df = featuregen.generate(n_samples=1000, random_seed=42)

    assert df.shape == (1000, 3)
    # Verify the derived feature is correctly calculated
    assert np.allclose(df['age_at_first_license'], df['age'] - df['driving_experience_years'])
    # Age at first license should be at least 18
    assert (df['age_at_first_license'] >= 18).all()


def test_multiple_derived_features():
    """Test multiple derived features depending on each other"""
    specs = [
        Feature('base', Uniform(low=10, high=20)),
        Feature('multiplier', Uniform(low=2, high=3)),
        DerivedFeature('product',
                       function=lambda b, m: b * m,
                       dependencies=('base', 'multiplier')),
        DerivedFeature('sum',
                       function=lambda b, p: b + p,
                       dependencies=('base', 'product')),
    ]

    featuregen = FeatureDefinition(specs)
    df = featuregen.generate(n_samples=100, random_seed=42)

    assert df.shape == (100, 4)
    # Verify calculations
    assert np.allclose(df['product'], df['base'] * df['multiplier'])
    assert np.allclose(df['sum'], df['base'] + df['product'])


def test_derived_feature_wrong_order():
    """Test that derived features also require correct dependency order"""
    specs = [
        DerivedFeature('derived',
                       function=lambda b: b * 2,
                       dependencies=('base',)),
        Feature('base', Uniform(low=10, high=20)),
    ]

    with pytest.raises(ValueError, match="have not been defined yet"):
        FeatureDefinition(specs)
