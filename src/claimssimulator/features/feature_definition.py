"""
Feature definitions for synthetic data generation.

Accepts a list of Feature / DerivedFeature dataclasses (defined in
``claimssimulator.feature_spec``) and generates a pandas DataFrame of
synthetic data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .feature_spec import (
    Feature,
    DerivedFeature,
    FormulaFeature,
    CorrelatedNormals,
    FeatureSpec,
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
    Transform,
    DependentTransform,
    get_distribution_dependencies,
)


class FeatureDefinition:
    """Generate a synthetic feature DataFrame from a list of feature specs.

    Parameters
    ----------
    specs : list[FeatureSpec]
        An ordered list of :class:`Feature` and :class:`DerivedFeature`
        instances.  Dependencies must appear *before* the features that
        reference them.

    Examples
    --------
    >>> from claimssimulator.feature_spec import *
    >>> gen = FeatureDefinition([
    ...     Feature('age', LogNormal(mean=3.5, sigma=0.2),
    ...             Transform(lambda x: x + 18)),
    ...     Feature('vehicle_age', Uniform(low=0, high=20)),
    ... ])
    >>> df = gen.generate(n_samples=1000, random_seed=42)
    """

    def __init__(self, specs: list[FeatureSpec]) -> None:
        self.specs = specs
        self._validate()

    # ── validation ────────────────────────────────────────────────

    def _validate(self) -> None:
        # Collect all names (CorrelatedNormals contributes multiple)
        all_names: list[str] = []
        for s in self.specs:
            match s:
                case CorrelatedNormals(names=names):
                    all_names.extend(names)
                case _:
                    all_names.append(s.name)

        # Duplicate names
        seen: set[str] = set()
        for n in all_names:
            if n in seen:
                raise ValueError(f"Duplicate feature name: '{n}'")
            seen.add(n)

        # Dependency ordering
        defined: set[str] = set()
        for spec in self.specs:
            deps = self._get_dependencies(spec)
            missing = deps - defined
            if missing:
                label = (
                    spec.names if isinstance(spec, CorrelatedNormals)
                    else spec.name
                )
                raise ValueError(
                    f"Feature '{label}' depends on {missing} which "
                    f"have not been defined yet. Reorder your feature list."
                )
            match spec:
                case CorrelatedNormals(names=names):
                    defined.update(names)
                case _:
                    defined.add(spec.name)

    @staticmethod
    def _get_dependencies(spec: FeatureSpec) -> set[str]:
        """Return the set of feature names *spec* depends on."""
        deps: set[str] = set()
        match spec:
            case DerivedFeature(dependencies=d):
                deps.update(d)
            case Feature(distribution=dist, transform=transform):
                # Dependencies from distribution parameters (str refs)
                deps.update(get_distribution_dependencies(dist))
                # Dependencies from transform
                match transform:
                    case DependentTransform(dependencies=d):
                        deps.update(d)
            case FormulaFeature(formula=formula, parameters=params):
                import re
                resolved = formula
                # Remove scalar parameter placeholders
                for pname, pvalue in params.items():
                    if isinstance(pvalue, (int, float)):
                        resolved = resolved.replace(f'{{{pname}}}', '0')
                # Extract categorical column names from col[param] syntax
                # and remove those tokens so the regex below doesn't
                # pick up the parameter name as a dependency.
                cat_pattern = re.compile(
                    r'([a-zA-Z_][a-zA-Z0-9_]*)\[([a-zA-Z_][a-zA-Z0-9_]*)\]'
                )
                for col_match in cat_pattern.finditer(resolved):
                    deps.add(col_match.group(1))
                resolved = cat_pattern.sub('0', resolved)
                # Remaining bare identifiers are feature references
                idents = set(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', resolved))
                deps.update(idents)
        return deps

    # ── sampling ──────────────────────────────────────────────────

    @staticmethod
    def _resolve_param(
        value: float | str,
        data: dict[str, np.ndarray],
    ) -> float | np.ndarray:
        """If *value* is a string, look it up in *data*; otherwise return as-is."""
        if isinstance(value, str):
            return data[value]
        return value

    @staticmethod
    def _sample(
        dist: Normal | LogNormal | LogNormalMeanStd | Gamma | GammaMeanStd | Beta | BetaMeanConcentration | Uniform | Exponential | Poisson | NegativeBinomial | Categorical,
        n: int,
        rng: np.random.Generator,
        data: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Draw *n* samples from *dist* using the provided *rng*.

        When a distribution parameter is a string, it is resolved from
        *data* (a dict of already-generated columns) so that the parameter
        varies per row.  NumPy broadcasting handles the element-wise
        sampling automatically.
        """
        _r = FeatureDefinition._resolve_param
        _d = data or {}

        match dist:
            case Normal(loc=loc, scale=scale):
                return rng.normal(_r(loc, _d), _r(scale, _d), n)
            case LogNormal(mean=mean, sigma=sigma):
                return rng.lognormal(_r(mean, _d), _r(sigma, _d), n)
            case LogNormalMeanStd(mean=mean, std=std):
                m = _r(mean, _d)
                s = _r(std, _d)
                sigma_u = np.sqrt(np.log(1 + (s / m) ** 2))
                mu_u = np.log(m) - sigma_u ** 2 / 2
                return rng.lognormal(mu_u, sigma_u, n)
            case Gamma(shape=shape, scale=scale):
                return rng.gamma(_r(shape, _d), _r(scale, _d), n)
            case GammaMeanStd(mean=mean, std=std):
                m = _r(mean, _d)
                s = _r(std, _d)
                sh = (m / s) ** 2
                sc = s ** 2 / m
                return rng.gamma(sh, sc, n)
            case Beta(a=a, b=b):
                return rng.beta(_r(a, _d), _r(b, _d), n)
            case BetaMeanConcentration(mean=mean, concentration=conc):
                m = _r(mean, _d)
                k = _r(conc, _d)
                return rng.beta(m * k, (1 - m) * k, n)
            case Uniform(low=low, high=high):
                return rng.uniform(_r(low, _d), _r(high, _d), n)
            case Exponential(scale=scale):
                return rng.exponential(_r(scale, _d), n)
            case Poisson(lam=lam):
                return rng.poisson(_r(lam, _d), n)
            case NegativeBinomial(n=nb_n, p=p):
                return rng.negative_binomial(_r(nb_n, _d), _r(p, _d), n)
            case Categorical(probabilities=probs, labels=labels):
                indices = rng.choice(len(probs), size=n, p=probs)
                if labels is not None:
                    return np.array(labels)[indices]
                return indices
            case _:
                raise TypeError(f"Unknown distribution type: {type(dist)}")

    # ── generation ────────────────────────────────────────────────

    def generate(
        self,
        n_samples: int,
        random_seed: int | None = None,
    ) -> pd.DataFrame:
        """Generate features based on the specification.

        Parameters
        ----------
        n_samples : int
            Number of rows to generate.
        random_seed : int | None
            Seed for the random number generator (reproducible output).

        Returns
        -------
        pd.DataFrame
        """
        rng = np.random.default_rng(random_seed)
        data: dict[str, np.ndarray] = {}

        for spec in self.specs:
            match spec:
                case Feature(name=name, distribution=dist, transform=transform):
                    values = self._sample(dist, n_samples, rng, data)
                    match transform:
                        case Transform(function=fn):
                            values = fn(values)
                        case DependentTransform(function=fn, dependencies=deps):
                            dep_values = [data[d] for d in deps]
                            values = fn(values, *dep_values)
                        case None:
                            pass
                    data[name] = values

                case CorrelatedNormals(
                    names=names, means=means, stds=stds,
                    correlation=correlation,
                ):
                    mean_arr = np.array(means, dtype=float)
                    std_arr = np.array(stds, dtype=float)
                    corr_arr = np.array(correlation, dtype=float)
                    cov = np.outer(std_arr, std_arr) * corr_arr
                    samples = rng.multivariate_normal(mean_arr, cov, n_samples)
                    for i, name in enumerate(names):
                        data[name] = samples[:, i]

                case DerivedFeature(name=name, function=fn, dependencies=deps):
                    dep_values = [data[d] for d in deps]
                    data[name] = fn(*dep_values)

                case FormulaFeature(
                    name=name, formula=formula, parameters=params, mean=target_mean
                ):
                    import re
                    resolved = formula

                    # 1. Expand categorical terms: col[param] → numeric array
                    #    using the dict in params[param].
                    cat_pattern = re.compile(
                        r'([a-zA-Z_][a-zA-Z0-9_]*)\[([a-zA-Z_][a-zA-Z0-9_]*)\]'
                    )
                    cat_arrays: dict[str, np.ndarray] = {}
                    for m in cat_pattern.finditer(resolved):
                        col_name, param_name = m.group(1), m.group(2)
                        label_map = params[param_name]
                        if not isinstance(label_map, dict):
                            raise TypeError(
                                f"Parameter '{param_name}' referenced via "
                                f"categorical syntax {col_name}[{param_name}] "
                                f"must be a dict, got {type(label_map).__name__}"
                            )
                        col_data = data[col_name]
                        placeholder = f'__cat_{col_name}_{param_name}__'
                        arr = np.zeros(len(col_data), dtype=float)
                        for label, coeff in label_map.items():
                            arr[col_data == label] = coeff
                        cat_arrays[placeholder] = arr
                        resolved = resolved.replace(m.group(0), placeholder)

                    # 2. Replace scalar parameter placeholders
                    for pname, pvalue in params.items():
                        if isinstance(pvalue, (int, float)):
                            resolved = resolved.replace(f'{{{pname}}}', str(pvalue))

                    # 3. Build temp DataFrame and evaluate
                    tmp_df = pd.DataFrame({**data, **cat_arrays})
                    values = tmp_df.eval(resolved).values
                    if target_mean is not None:
                        current_mean = values.mean()
                        if current_mean != 0:
                            values = values * (target_mean / current_mean)
                    data[name] = values

        return pd.DataFrame(data)

    # ── helpers ───────────────────────────────────────────────────

    def get_feature_names(self) -> list[str]:
        """Return the list of feature names in definition order."""
        names: list[str] = []
        for s in self.specs:
            match s:
                case CorrelatedNormals(names=block_names):
                    names.extend(block_names)
                case _:
                    names.append(s.name)
        return names
