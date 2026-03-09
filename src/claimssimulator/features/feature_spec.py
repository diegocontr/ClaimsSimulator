"""
Dataclass-based feature specification for synthetic data generation.

Instead of specifying features as raw tuples/dicts, use typed dataclasses
that provide IDE autocomplete, type checking, and self-documenting code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Callable

# Type alias: a distribution parameter may be a numeric literal **or** a
# string referencing a previously generated feature column.
Param = float | str


# ── Distribution specifications ───────────────────────────────────────


@dataclass(frozen=True)
class Normal:
    """Normal (Gaussian) distribution.

    Parameters may be numeric literals **or** strings that reference a
    previously generated feature column (element-wise sampling).
    """

    loc: Param = 0.0
    scale: Param = 1.0


@dataclass(frozen=True)
class LogNormal:
    """Log-normal distribution (parameterised by the underlying normal).

    Parameters
    ----------
    mean : Param
        Mean of the underlying normal distribution (μ).
        May be a float or a string referencing another column.
    sigma : Param
        Standard deviation of the underlying normal distribution (σ).
        May be a float or a string referencing another column.

    Notes
    -----
    The *output* (exponentiated) values have:

    * E[X] = exp(mean + sigma² / 2)
    * Var[X] = [exp(sigma²) - 1] · exp(2·mean + sigma²)

    See also :class:`LogNormalMeanStd` for specifying the output mean/std
    directly.
    """

    mean: Param = 0.0
    sigma: Param = 1.0


@dataclass(frozen=True)
class LogNormalMeanStd:
    """Log-normal distribution parameterised by the **output** mean and std.

    Parameters
    ----------
    mean : Param
        Desired mean of the output distribution.  Must be > 0 (when scalar).
    std : Param
        Desired standard deviation of the output distribution.  Must be > 0
        (when scalar).

    The underlying normal parameters are derived as:

    * σ² = ln(1 + (std / mean)²)
    * μ  = ln(mean) − σ² / 2
    """

    mean: Param
    std: Param

    def __post_init__(self) -> None:
        if isinstance(self.mean, (int, float)) and self.mean <= 0:
            raise ValueError(
                f"LogNormalMeanStd requires mean > 0, got {self.mean}"
            )
        if isinstance(self.std, (int, float)) and self.std <= 0:
            raise ValueError(
                f"LogNormalMeanStd requires std > 0, got {self.std}"
            )

    @property
    def _sigma_underlying(self) -> float:
        """σ of the underlying normal."""
        return math.sqrt(math.log(1 + (self.std / self.mean) ** 2))

    @property
    def _mean_underlying(self) -> float:
        """μ of the underlying normal."""
        return math.log(self.mean) - self._sigma_underlying ** 2 / 2


@dataclass(frozen=True)
class Gamma:
    """Gamma distribution (shape / scale parameterisation).

    Parameters
    ----------
    shape : Param
        Shape parameter (k).  Must be > 0 (when scalar).
    scale : Param
        Scale parameter (θ).  Must be > 0 (when scalar).

    Notes
    -----
    E[X] = shape · scale, Var[X] = shape · scale².

    See also :class:`GammaMeanStd` for specifying mean and std directly.
    """

    shape: Param = 1.0
    scale: Param = 1.0


@dataclass(frozen=True)
class GammaMeanStd:
    """Gamma distribution parameterised by **output** mean and std.

    Parameters
    ----------
    mean : Param
        Desired mean of the distribution.  Must be > 0 (when scalar).
    std : Param
        Desired standard deviation.  Must be > 0 (when scalar).

    The native parameters are derived as:

    * shape = (mean / std)²
    * scale = std² / mean
    """

    mean: Param
    std: Param

    def __post_init__(self) -> None:
        if isinstance(self.mean, (int, float)) and self.mean <= 0:
            raise ValueError(
                f"GammaMeanStd requires mean > 0, got {self.mean}"
            )
        if isinstance(self.std, (int, float)) and self.std <= 0:
            raise ValueError(
                f"GammaMeanStd requires std > 0, got {self.std}"
            )

    @property
    def _shape(self) -> float:
        return (self.mean / self.std) ** 2

    @property
    def _scale(self) -> float:
        return self.std ** 2 / self.mean


@dataclass(frozen=True)
class Beta:
    """Beta distribution (standard parameterisation).

    Parameters
    ----------
    a : float
        Shape parameter α.  Must be > 0.
    b : float
        Shape parameter β.  Must be > 0.

    Notes
    -----
    The distribution is supported on [0, 1] with:

    * E[X] = α / (α + β)
    * Var[X] = αβ / ((α + β)² (α + β + 1))

    See also :class:`BetaMeanConcentration` for specifying the mean and
    concentration directly.
    """

    a: Param = 1.0
    b: Param = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.a, (int, float)) and self.a <= 0:
            raise ValueError(f"Beta requires a > 0, got {self.a}")
        if isinstance(self.b, (int, float)) and self.b <= 0:
            raise ValueError(f"Beta requires b > 0, got {self.b}")


@dataclass(frozen=True)
class BetaMeanConcentration:
    """Beta distribution parameterised by **mean** and **concentration**.

    Parameters
    ----------
    mean : float
        Desired mean of the distribution.  Must be in (0, 1).
    concentration : float
        Concentration (κ = α + β).  Must be > 0.
        Higher values produce a tighter distribution around the mean;
        lower values produce a wider spread.

    The native parameters are derived as:

    * α = mean · concentration
    * β = (1 − mean) · concentration

    Examples
    --------
    >>> BetaMeanConcentration(mean=0.3, concentration=10)
    # equivalent to Beta(a=3, b=7)
    """

    mean: Param
    concentration: Param

    def __post_init__(self) -> None:
        if isinstance(self.mean, (int, float)) and not (0 < self.mean < 1):
            raise ValueError(
                f"BetaMeanConcentration requires 0 < mean < 1, got {self.mean}"
            )
        if isinstance(self.concentration, (int, float)) and self.concentration <= 0:
            raise ValueError(
                f"BetaMeanConcentration requires concentration > 0, "
                f"got {self.concentration}"
            )

    @property
    def _a(self) -> float:
        return self.mean * self.concentration

    @property
    def _b(self) -> float:
        return (1 - self.mean) * self.concentration


@dataclass(frozen=True)
class Uniform:
    """Uniform distribution."""

    low: Param = 0.0
    high: Param = 1.0


@dataclass(frozen=True)
class Exponential:
    """Exponential distribution."""

    scale: Param = 1.0


@dataclass(frozen=True)
class Poisson:
    """Poisson distribution."""

    lam: Param = 1.0


@dataclass(frozen=True)
class NegativeBinomial:
    """Negative binomial distribution."""

    n: Param = 1
    p: Param = 0.5


@dataclass(frozen=True)
class Categorical:
    """Categorical distribution.

    Parameters
    ----------
    probabilities : list[float]
        Probabilities for each category (must sum to 1).
    labels : list[str] | None
        Optional labels for each category. If None, integer indices are used.
    """

    probabilities: list[float]
    labels: list[str] | None = None


Distribution = (
    Normal
    | LogNormal
    | LogNormalMeanStd
    | Gamma
    | GammaMeanStd
    | Beta
    | BetaMeanConcentration
    | Uniform
    | Exponential
    | Poisson
    | NegativeBinomial
    | Categorical
)


# ── Transform specifications ─────────────────────────────────────────


@dataclass(frozen=True)
class Transform:
    """A simple element-wise transformation applied to sampled values.

    Parameters
    ----------
    function : Callable
        A function ``f(x) -> x'`` where *x* is the array of sampled values.

    Examples
    --------
    >>> Transform(lambda x: x + 18)          # shift
    >>> Transform(lambda x: np.clip(x, 1, 10))  # clip
    """

    function: Callable


@dataclass(frozen=True)
class DependentTransform:
    """A transformation that depends on previously generated features.

    Parameters
    ----------
    function : Callable
        A function ``f(x, dep1, dep2, ...) -> x'`` where *x* is the array
        of sampled values and *dep1*, *dep2*, … are arrays from the
        dependency features.
    dependencies : tuple[str, ...]
        Names of features that this transform depends on. These features
        must appear **before** the current feature in the feature list.

    Examples
    --------
    >>> DependentTransform(
    ...     lambda x, age: np.clip(x, 0, age - 18),
    ...     dependencies=('age',),
    ... )
    """

    function: Callable
    dependencies: tuple[str, ...]


# ── Feature specifications ────────────────────────────────────────────


@dataclass(frozen=True)
class Feature:
    """A feature sampled from a distribution, optionally transformed.

    Parameters
    ----------
    name : str
        Column name in the generated DataFrame.
    distribution : Distribution
        One of the distribution dataclasses (Normal, LogNormal, …).
    transform : Transform | DependentTransform | None
        Optional transformation applied after sampling.

    Examples
    --------
    >>> Feature('age', LogNormal(mean=3.5, sigma=0.2),
    ...         Transform(lambda x: x + 18))
    >>> Feature('vehicle_age', Uniform(low=0, high=20))
    >>> Feature('experience', Gamma(shape=2, scale=5),
    ...         DependentTransform(
    ...             lambda x, age: np.clip(x, 0, age - 18),
    ...             dependencies=('age',)))
    """

    name: str
    distribution: Distribution
    transform: Transform | DependentTransform | None = None


@dataclass(frozen=True)
class DerivedFeature:
    """A feature computed purely from other features (no random sampling).

    Parameters
    ----------
    name : str
        Column name in the generated DataFrame.
    function : Callable
        A function ``f(dep1, dep2, ...) -> values``.
    dependencies : tuple[str, ...]
        Names of features this derived feature depends on.

    Examples
    --------
    >>> DerivedFeature('age_at_first_license',
    ...               function=lambda age, exp: age - exp,
    ...               dependencies=('age', 'driving_experience_years'))
    """

    name: str
    function: Callable
    dependencies: tuple[str, ...]


@dataclass(frozen=True)
class FormulaFeature:
    """A feature defined by a string formula over other features.

    The formula uses ``{param_name}`` placeholders for parameters, and
    bare identifiers for references to other features.  The formula is
    evaluated with :meth:`pandas.DataFrame.eval`.

    Categorical features can be included via ``column[param_name]``
    notation, where *param_name* maps to a ``dict[str, float]`` in
    ``parameters`` giving per-label coefficients.  Labels not listed
    default to 0.

    Parameters
    ----------
    name : str
        Column name in the generated DataFrame.
    formula : str
        Expression string.  Parameter placeholders use curly braces, e.g.
        ``"{beta_age} * age + {beta_exp} * experience"``.
        Categorical terms use ``column[param_name]``, e.g.
        ``"vehicle_type[b_vehtype]"``.
    parameters : dict[str, float | dict[str, float]]
        Mapping from placeholder names to numeric values **or** to
        ``{label: coefficient}`` dicts for categorical terms.
    mean : float | None
        If given, the result is linearly rescaled so that its sample mean
        equals this value (useful for risk scores).

    Examples
    --------
    >>> FormulaFeature(
    ...     'profile_risk',
    ...     formula="{beta_age} * age + vehicle_type[b_vehtype]",
    ...     parameters={
    ...         'beta_age': 0.02,
    ...         'b_vehtype': {'Sedan': 0.0, 'SUV': 0.3, 'Truck': 0.5},
    ...     },
    ...     mean=0.25,
    ... )
    """

    name: str
    formula: str
    parameters: dict[str, float | dict[str, float]]
    mean: float | None = None


@dataclass(frozen=True)
class CorrelatedNormals:
    """A block of correlated normal variables generated jointly.

    Parameters
    ----------
    names : tuple[str, ...]
        Column names for each of the *N* variables.
    means : tuple[float, ...]
        Mean of each variable (length *N*).
    stds : tuple[float, ...]
        Standard deviation of each variable (length *N*).  All values
        must be > 0.
    correlation : tuple[tuple[float, ...], ...]
        *N × N* symmetric correlation matrix with ones on the diagonal.
        Each off-diagonal entry must be in [−1, 1].  The matrix must be
        positive semi-definite.

    Notes
    -----
    Internally the correlation matrix and ``stds`` are combined into a
    full covariance matrix:

        Σ = diag(stds) @ R @ diag(stds)

    which is passed to ``numpy.random.Generator.multivariate_normal``.

    Examples
    --------
    >>> CorrelatedNormals(
    ...     names=('x1', 'x2'),
    ...     means=(0.0, 5.0),
    ...     stds=(1.0, 2.0),
    ...     correlation=((1.0, 0.8),
    ...                  (0.8, 1.0)),
    ... )
    """

    names: tuple[str, ...]
    means: tuple[float, ...]
    stds: tuple[float, ...]
    correlation: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        n = len(self.names)

        if len(self.means) != n:
            raise ValueError(
                f"Length of means ({len(self.means)}) must equal "
                f"number of names ({n})"
            )
        if len(self.stds) != n:
            raise ValueError(
                f"Length of stds ({len(self.stds)}) must equal "
                f"number of names ({n})"
            )
        for i, s in enumerate(self.stds):
            if s <= 0:
                raise ValueError(
                    f"All stds must be > 0; stds[{i}] = {s}"
                )

        # Validate correlation matrix shape
        if len(self.correlation) != n:
            raise ValueError(
                f"Correlation matrix must be {n}×{n}, "
                f"got {len(self.correlation)} rows"
            )
        for i, row in enumerate(self.correlation):
            if len(row) != n:
                raise ValueError(
                    f"Correlation matrix row {i} has length {len(row)}, "
                    f"expected {n}"
                )

        # Diagonal must be 1, off-diagonal in [-1, 1], and symmetric
        for i in range(n):
            if not math.isclose(self.correlation[i][i], 1.0):
                raise ValueError(
                    f"Diagonal of correlation matrix must be 1.0; "
                    f"correlation[{i}][{i}] = {self.correlation[i][i]}"
                )
            for j in range(i + 1, n):
                rij = self.correlation[i][j]
                rji = self.correlation[j][i]
                if not math.isclose(rij, rji):
                    raise ValueError(
                        f"Correlation matrix must be symmetric; "
                        f"correlation[{i}][{j}]={rij} != "
                        f"correlation[{j}][{i}]={rji}"
                    )
                if not (-1.0 <= rij <= 1.0):
                    raise ValueError(
                        f"Correlation values must be in [-1, 1]; "
                        f"correlation[{i}][{j}] = {rij}"
                    )

        # Duplicate names
        if len(set(self.names)) != n:
            raise ValueError(
                f"CorrelatedNormals contains duplicate names: {self.names}"
            )


FeatureSpec = Feature | DerivedFeature | FormulaFeature | CorrelatedNormals


def get_distribution_dependencies(dist: Distribution) -> set[str]:
    """Return the set of feature-column names referenced by *dist* parameters.

    Any distribution field whose value is a ``str`` is treated as a
    reference to a previously generated column.
    """
    deps: set[str] = set()
    for f in fields(dist):
        val = getattr(dist, f.name)
        if isinstance(val, str):
            deps.add(val)
    return deps
