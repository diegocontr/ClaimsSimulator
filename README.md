# Claims Simulator

A synthetic data generation system for creating realistic insurance claims datasets with features, latent risk variables, and simulated claim events.

## Project Structure

```
├── src/                           # Source code
│   ├── __init__.py               # Package initialization
│   ├── distributions.py          # Distribution classes and factory
│   ├── feature_definition.py     # Feature specification and generation
│   ├── latent_risk_builder.py    # Latent risk component builder
│   ├── claims_simulator.py       # Claims and severity simulation (future)
│   └── ...
├── notebooks/                    # Jupyter notebooks for exploration
│   └── 01_feature_definition_and_latent_risk.ipynb
├── tests/                        # Unit tests
│   ├── test_distributions.py
│   ├── test_feature_definition.py
│   ├── test_latent_risk_builder.py
│   └── ...
├── data/                         # Generated datasets
├── pyproject.toml               # Project configuration
└── README.md                    # This file
```

## Features

### 1. **Distribution System** (`distributions.py`)

Flexible distribution definitions supporting:
- **Normal (Gaussian)**: `('normal', {'loc': mean, 'scale': std})`
- **Log-Normal**: `('lognormal', {'mean': mean, 'sigma': sigma})`
- **Gamma**: `('gamma', {'shape': k, 'scale': theta})`
- **Uniform**: `('uniform', {'low': a, 'high': b})`
- **Negative Binomial**: `('negative_binomial', {'n': n, 'p': p})`
- **Poisson**: `('poisson', {'lam': lambda})`
- **Exponential**: `('exponential', {'scale': scale})`
- **Categorical**: `('categorical', {'probabilities': [...], 'labels': [...]})`

### 2. **Feature Definition** (`feature_definition.py`)

Define and generate synthetic features using a specification dictionary with optional transformations:

```python
from claimsimulator.feature_definition import FeatureDefinition
import numpy as np

feature_spec = {
    # Basic feature
    'income': ('normal', {'loc': 50000, 'scale': 15000}),
    
    # Simple transformation - shift to ensure age >= 18
    'age': ('lognormal', {'mean': 3.5, 'sigma': 0.2}, lambda x: x + 18),
    
    # Dependent transformation - clip experience to age - 18
    'driving_experience_years': ('gamma', {'shape': 2, 'scale': 5}, 
                                 (lambda x, age: np.clip(x, 0, age - 18), ('age',))),
    
    # Derived feature - pure function of other features (no distribution)
    'age_at_first_license': (lambda age, exp: age - exp, ('age', 'driving_experience_years')),
    
    # Categorical feature
    'vehicle_type': ('categorical', {
        'probabilities': [0.5, 0.3, 0.2],
        'labels': ['Sedan', 'SUV', 'Truck']
    }),
}

feature_gen = FeatureDefinition(feature_spec)
features_df = feature_gen.generate(n_samples=5000, random_seed=42)
```

**Feature Specification Formats:**
- **Basic**: `'name': ('distribution', {args})`
- **Transformed**: `'name': ('distribution', {args}, transform_func)`
- **Dependent**: `'name': ('distribution', {args}, (transform_func, ('dep1', 'dep2')))`
- **Derived**: `'name': (transform_func, ('dep1', 'dep2'))` - pure function, no distribution

### 3. **Latent Risk Builder** (`latent_risk_builder.py`)

Compose latent risk variables from multiple feature-based components:

```python
from claimsimulator.latent_risk_builder import LatentRiskBuilder

# Define risk components
def age_risk(df):
    age = df['age'].values
    risk = np.where(age < 25, 1.8, np.where(age < 60, 1.0, 1.2))
    return risk

def experience_risk(df):
    exp = df['driving_experience_years'].values
    return np.exp(-0.08 * exp)

# Build composite risk
risk_builder = LatentRiskBuilder(normalize_weights=True)
risk_builder.add_component('age_risk', ['age'], age_risk, weight=0.5)
risk_builder.add_component('exp_risk', ['experience'], experience_risk, weight=0.5)

risk_df = risk_builder.build(features_df)
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

See the notebook `notebooks/01_feature_definition_and_latent_risk.ipynb` for a comprehensive example.

### Example: Generate Features and Calculate Risk

```python
import numpy as np
import pandas as pd
from claimsimulator.feature_definition import FeatureDefinition
from claimsimulator.latent_risk_builder import LatentRiskBuilder

# 1. Define features
feature_spec = {
    'age': ('normal', {'loc': 35, 'scale': 15}),
    'experience': ('gamma', {'shape': 2, 'scale': 5}),
}

# 2. Generate features
feature_gen = FeatureDefinition(feature_spec)
features_df = feature_gen.generate(n_samples=1000, random_seed=42)

# 3. Define risk components
def age_component(df):
    age = df['age'].values if isinstance(df, pd.DataFrame) else df.values
    return np.where(age < 25, 1.5, np.where(age < 60, 1.0, 1.2))

def experience_component(df):
    exp = df['experience'].values if isinstance(df, pd.DataFrame) else df.values
    return np.exp(-0.1 * exp)

# 4. Build latent risk
builder = LatentRiskBuilder(normalize_weights=True)
builder.add_component('age_risk', ['age'], age_component, weight=0.6)
builder.add_component('exp_risk', ['experience'], experience_component, weight=0.4)

risk_df = builder.build(features_df)

# 5. Results
print(risk_df[['age', 'experience', 'age_risk', 'exp_risk', 'latent_risk']].head())
```

## API Reference

### FeatureDefinition

```python
FeatureDefinition(features_spec: Dict[str, Tuple[str, dict]])
    .generate(n_samples: int, random_seed: int = None) -> pd.DataFrame
    .get_feature_names() -> List[str]
```

### LatentRiskBuilder

```python
LatentRiskBuilder(normalize_weights: bool = True)
    .add_component(name: str, function: str, parameters: Dict[str, float],
                   weight: float = 1.0, mean: float = None) -> LatentRiskBuilder
    .build(features_df: pd.DataFrame) -> pd.DataFrame
    .get_component_names() -> List[str]
```

**Parameters:**
- `function`: String formula with placeholders like `"{beta} * age + {gamma} * experience"`
- `parameters`: Dictionary mapping parameter names to values, e.g., `{'beta': 0.02, 'gamma': -0.01}`
- `mean`: Optional target mean for renormalization of the component
- `weight`: Weight for the component (informational, not used for aggregation)

**Note:** `build()` creates only individual risk components. Each component represents a different type of risk that can be used independently in your models.

### ClaimsSimulator

```python
ClaimsSimulator(generator: Union[str, Callable] = 'Poisson',
                param_columns: Optional[Dict[str, str]] = None,
                claim_column: str = 'claim',
                time_to_simulate: Union[str, float] = 'contract_duration_years',
                max_exposure: float = 1.0,
                exposure_column: str = 'exposure',
                claim_counter: Optional[str] = None,
                renewal_mode: Literal['claim', 'contract_end'] = 'claim',
                random_seed: Optional[int] = None)
    .simulate(risk_df: pd.DataFrame) -> pd.DataFrame
```

**Constructor Parameters:**
- `generator`: Distribution for generating claims. Can be:
  - `'Poisson'`: Exponential inter-arrival times (default)
  - `'NegativeBinomialMixture'`: Gamma-Poisson mixture (unobserved heterogeneity)
  - `'NegativeBinomialMeanVar'`: Proper NB where E[claims/exposure] = rate
  - `Callable`: Custom function with signature `(params: Dict[str, float]) -> float` that returns time to next claim
- `param_columns`: Dict mapping parameter names to column names:
  - Poisson: `{'rate': 'column_name'}`
  - NegativeBinomialMixture: `{'rate': 'col1', 'dispersion': 'col2'}`
  - NegativeBinomialMeanVar: `{'rate': 'col1', 'overdispersion': 'col2'}`
  - Custom callable: depends on your function's requirements (must be provided)
- `claim_column`: Name for output claim indicator column (default 'claim')
- `time_to_simulate`: Column name or fixed value for contract duration (default 'contract_duration_years')
- `max_exposure`: Maximum exposure time per row (default 1.0 year)
- `exposure_column`: Name for output exposure column (default 'exposure')
- `claim_counter`: If provided, adds a column counting claims before current row (default None)
- `renewal_mode`: Controls how contracts split into images/rows (default 'claim'):
  - `'claim'`: Contract renews immediately after each claim. After a claim at time t, starts new row from time t.
  - `'contract_end'`: Contract renews at fixed max_exposure intervals. Multiple claims within an interval create multiple rows, but intervals always complete to max_exposure boundaries (or total_duration).
- `random_seed`: Random seed for reproducibility
- `exposure_column`: Name for output exposure column (default 'exposure')
- `claim_counter`: If provided, adds a column counting claims before current row (default None)
- `random_seed`: Random seed for reproducibility

**Example Usage:**
```python
# Poisson process with claim counter
sim = ClaimsSimulator(
    generator='Poisson',
    param_columns={'rate': 'risk'},
    time_to_simulate='duration',
    max_exposure=1.0,
    claim_counter='past_claims'  # Track number of previous claims
)
claims_df = sim.simulate(risk_df)

# NegativeBinomialMeanVar (overdispersion with E[claims/exposure] = rate)
sim_nb = ClaimsSimulator(
    generator='NegativeBinomialMeanVar',
    param_columns={'rate': 'risk', 'overdispersion': 'overdisp'},
    time_to_simulate='duration',
    max_exposure=1.0
)
claims_df_nb = sim_nb.simulate(risk_df)

# Contract renewal at fixed intervals (contract_end mode)
sim_contract_end = ClaimsSimulator(
    generator='Poisson',
    param_columns={'rate': 'risk'},
    time_to_simulate='duration',
    max_exposure=1.0,
    renewal_mode='contract_end',  # Renews at max_exposure boundaries
    claim_counter='past_claims'
)
claims_df_contract_end = sim_contract_end.simulate(risk_df)

# Custom generator function (e.g., Weibull distribution)
def weibull_time_generator(params):
    """Returns time to next claim from Weibull distribution"""
    import numpy as np
    shape = params['shape']
    scale = params['scale']
    return np.random.weibull(shape) * scale

risk_df['weibull_shape'] = 2.0
risk_df['weibull_scale'] = 3.0

sim_custom = ClaimsSimulator(
    generator=weibull_time_generator,
    param_columns={'shape': 'weibull_shape', 'scale': 'weibull_scale'},
    time_to_simulate='duration',
    max_exposure=1.0
)
claims_df_custom = sim_custom.simulate(risk_df)
```

**Output Format (for Piecewise Exponential Models):**
- Each contract generates multiple rows (images) based on claim events
- `exposure`: Time exposed to risk in this interval
- `claim`: 1 if claim occurred at end of interval, 0 otherwise
- `start_time`, `end_time`: Time boundaries of the interval
- `contract_id`: Identifier to track original contracts
- If claim occurs, a new row starts immediately after

## Testing

Run the test suite:

```bash
pytest tests/
```

Run specific test file:

```bash
pytest tests/test_distributions.py -v
pytest tests/test_feature_definition.py -v
pytest tests/test_latent_risk_builder.py -v
```

## Architecture

The project follows a modular design:

1. **Distributions Layer**: Handles all probability distributions
2. **Feature Layer**: Generates synthetic features from specifications
3. **Risk Layer**: Computes latent risk from features
4. **Simulation Layer**: (Coming soon) Simulates claim events based on risk

## Future Work

- Claims event simulation based on latent risk
- Claim severity modeling
- Claims simulator integration
- Advanced risk modeling techniques
- Documentation and examples expansion

## Installation & Requirements

- Python 3.12+
- NumPy >= 2.2.6
- Pandas >= 2.2.3
- SciPy >= 1.15.3
- Scikit-learn >= 1.6.1 (optional, for future modeling)

See `pyproject.toml` for complete dependency list.

