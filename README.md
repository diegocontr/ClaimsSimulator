# Claims Simulator

Synthetic insurance data generation for:
- feature simulation with explicit dependencies,
- formula-based latent risk construction,
- piecewise claim-event simulation (frequency),
- optional claim severity generation,
- feature diagnostics and model evaluation metrics.

Current package version in code: `0.2.0` (`src/claimssimulator/__init__.py`).

## What Is In This Version

- Dataclass-based feature specifications (no tuple/dict DSL required).
- Dependent distribution parameters (for example `Normal(loc="mu")`).
- `FormulaFeature` with scalar placeholders and categorical term syntax.
- `CorrelatedNormals` for multivariate normal feature blocks.
- Claims simulation with:
  - `Poisson`, `NegativeBinomialMixture`, `NegativeBinomialMeanVar`, or custom generators,
  - `claim` and `contract_end` renewal modes,
  - optional time-varying parameters via callables,
  - optional severity sampling (`claim_cost`).
- Visualization helpers for histogram and feature-association analysis.
- Built-in metrics for frequency/severity model evaluation.

## Installation

From the repository root:

```bash
pip install -e .
```

If you only want runtime dependencies from the pinned list:

```bash
pip install -r requirements.txt
```

Requirements:
- Python `>=3.12`

## Quick Start

```python
import numpy as np
from claimssimulator import (
    FeatureDefinition,
    Feature,
    DerivedFeature,
    FormulaFeature,
    Transform,
    DependentTransform,
    Normal,
    Gamma,
    Uniform,
    Categorical,
)

specs = [
    Feature("age", Normal(loc=35, scale=12), Transform(lambda x: np.clip(x, 18, 95))),
    Feature("experience", Gamma(shape=2.0, scale=6.0), DependentTransform(
        lambda x, age: np.clip(x, 0, age - 18), dependencies=("age",)
    )),
    Feature("vehicle_type", Categorical(
        probabilities=[0.55, 0.30, 0.15], labels=["sedan", "suv", "truck"]
    )),
    DerivedFeature(
        "age_at_first_license",
        function=lambda age, exp: age - exp,
        dependencies=("age", "experience"),
    ),
    FormulaFeature(
        "risk",
        formula="{b_age} * age + {b_exp} * experience + vehicle_type[b_vehicle]",
        parameters={
            "b_age": 0.015,
            "b_exp": -0.02,
            "b_vehicle": {"sedan": 0.00, "suv": 0.20, "truck": 0.40},
        },
        mean=0.5,
    ),
]

features = FeatureDefinition(specs).generate(n_samples=5000, random_seed=42)
print(features.head())
```

## Claims Simulation

`ClaimsSimulator` converts one row per contract into piecewise rows with exposure and claim indicators.

```python
from claimssimulator import ClaimsSimulator

risk_df = features[["risk"]].copy()
risk_df["duration"] = 3.0

sim = ClaimsSimulator(
    generator="Poisson",
    param_columns={"rate": "risk"},
    time_to_simulate="duration",
    max_exposure=1.0,
    renewal_mode="claim",
    claim_counter="n_previous_claims",
    random_seed=42,
)

claims = sim.simulate(risk_df)
print(claims[["contract_id", "start_time", "end_time", "exposure", "claim"]].head())
```

Core output columns:
- `exposure` (or your custom `exposure_column`)
- `claim` (or your custom `claim_column`)
- `start_time`, `end_time`
- `contract_id`
- optional `claim_counter` column

Optional severity model:

```python
sim_cost = ClaimsSimulator(
    generator="Poisson",
    param_columns={"rate": "risk"},
    time_to_simulate="duration",
    severity_column="expected_claim_cost",
    severity_cv=0.4,
    claim_cost_column="claim_cost",
    random_seed=42,
)
```

Optional time-varying parameter example:

```python
sim_tv = ClaimsSimulator(
    generator="Poisson",
    param_columns={"rate": lambda t: 0.1 + 0.02 * t},
    time_to_simulate=2.0,
    random_seed=42,
)
```

## Visualization Utilities

```python
from claimssimulator import compute_feature_analysis, visualize_features

analysis = compute_feature_analysis(features, feature="risk", association="spearman")
print(analysis.correlations.head())

fig = visualize_features(features, features=["age", "experience", "risk"], vlines=["mean", "perc95"])
```

Supported association keywords:
- `pearson`
- `spearman`
- `kendall`
- `mutual_info`
- `hoeffding`

## Metrics

Available top-level metrics:
- `gini`
- `poisson_deviance_ratio`
- `gamma_deviance_ratio`
- `calibration_quality_ratio`
- `mae`, `rmse`, `mape`, `r2`

Example:

```python
from claimssimulator import gini, poisson_deviance_ratio

score_gini = gini(t=y_true, p=y_pred, w=weights)
score_pdr = poisson_deviance_ratio(t=y_true, p=y_pred, w=weights)
```

## Project Structure

```text
ClaimsSimulator/
|-- src/claimssimulator/
|   |-- __init__.py
|   |-- metrics.py
|   |-- features/
|   |   |-- feature_spec.py
|   |   `-- feature_definition.py
|   |-- simulation/
|   |   `-- claims_simulator.py
|   `-- visualization/
|       `-- feature_visualization.py
|-- tests/
|-- notebooks/
|-- pyproject.toml
`-- README.md
```


## Notebooks

The `notebooks/` folder contains worked examples for:
- feature definition and latent risk construction,
- exposure-aware synthetic datasets,
- model training workflows.

