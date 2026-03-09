"""
Claims Simulator - Synthetic insurance claims data generation
"""

__version__ = "0.2.0"
__author__ = "Diego"

from .features import (
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
    # Transforms
    Transform,
    DependentTransform,
    # Feature specs
    Feature,
    DerivedFeature,
    FormulaFeature,
    CorrelatedNormals,
    # Core classes
    FeatureDefinition,
)
from .simulation import ClaimsSimulator
from .visualization import (
    ASSOCIATION_LABELS,
    compute_feature_analysis,
    compute_histogram_data,
    analyze_feature,
    visualize_features,
)
from .metrics import (
    gini,
    poisson_deviance_ratio,
    gamma_deviance_ratio,
    calibration_quality_ratio,
    mae,
    rmse,
    mape,
    r2
)

__all__ = [
    # Distribution specs
    'Normal',
    'LogNormal',
    'LogNormalMeanStd',
    'Gamma',
    'GammaMeanStd',
    'Beta',
    'BetaMeanConcentration',
    'Uniform',
    'Exponential',
    'Poisson',
    'NegativeBinomial',
    'Categorical',
    # Transform specs
    'Transform',
    'DependentTransform',
    # Feature specs
    'Feature',
    'DerivedFeature',
    'FormulaFeature',
    'CorrelatedNormals',
    # Core classes
    'FeatureDefinition',
    'ClaimsSimulator',
    # Visualization
    'ASSOCIATION_LABELS',
    'compute_feature_analysis',
    'compute_histogram_data',
    'analyze_feature',
    'visualize_features',
    # Metrics
    'gini',
    'poisson_deviance_ratio',
    'gamma_deviance_ratio',
    'calibration_quality_ratio',
    'mae',
    'rmse',
    'mape',
    'r2'
]