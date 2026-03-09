"""
Feature specification and generation for synthetic insurance data.
"""

from .feature_spec import (
    # Distributions
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
    Categorical,
    Distribution,
    # Transforms
    Transform,
    DependentTransform,
    # Feature specs
    Feature,
    DerivedFeature,
    FormulaFeature,
    CorrelatedNormals,
    FeatureSpec,
    # Helpers
    Param,
    get_distribution_dependencies,
)
from .feature_definition import FeatureDefinition

__all__ = [
    # Distribution specs
    "Normal",
    "LogNormal",
    "LogNormalMeanStd",
    "Gamma",
    "GammaMeanStd",
    "Beta",
    "BetaMeanConcentration",
    "Uniform",
    "Exponential",
    "Poisson",
    "NegativeBinomial",
    "Categorical",
    "Distribution",
    # Transform specs
    "Transform",
    "DependentTransform",
    # Feature specs
    "Feature",
    "DerivedFeature",
    "FormulaFeature",
    "CorrelatedNormals",
    "FeatureSpec",
    # Helpers
    "Param",
    "get_distribution_dependencies",
    # Core classes
    "FeatureDefinition",
]
