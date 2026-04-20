"""
Multi-Cause Claims Simulator - Generate claim events from multiple independent claim sources.

Each claim source (cause) has its own generator, param_columns, and configuration.
Exposure is cut whenever *any* cause generates a claim, matching the competing-risks
/ piecewise-exponential framework used by ClaimsSimulator.
"""

from typing import Union, Optional, Literal, Dict, Callable, List
import pandas as pd
import numpy as np

from .claims_simulator import (
    _poisson_time_to_claim,
    _negative_binomial_mixture_time_to_claim,
    _negative_binomial_meanvar_time_to_claim,
    _gamma_severity,
)


class ClaimSource:
    """
    Configuration for a single claim source (cause).

    Parameters
    ----------
    name : str
        Unique identifier for this source.  Used to name the output claim and
        cost columns, e.g. ``'fire'`` produces a ``'claim_fire'`` column.
    generator : str or callable
        Same semantics as ``ClaimsSimulator.generator``.
    param_columns : dict, optional
        Same semantics as ``ClaimsSimulator.param_columns``.
    claim_column : str, optional
        Override the output claim-indicator column name.  Defaults to
        ``'claim_<name>'``.
    severity_column : str, optional
        Column in the input DataFrame containing the mean claim cost for this
        source.  If ``None`` no cost column is written.
    severity_cv : float
        Coefficient of variation for claim costs (default 0.3).
    claim_cost_column : str, optional
        Override the output cost column name.  Defaults to
        ``'claim_cost_<name>'``.
    severity_generator : callable, optional
        ``f(mean, std, rng) -> float`` – defaults to Gamma.
    """

    def __init__(
        self,
        name: str,
        generator: Union[
            Literal["Poisson", "NegativeBinomialMixture", "NegativeBinomialMeanVar"],
            Callable[[Dict[str, float]], float],
        ] = "Poisson",
        param_columns: Optional[Dict[str, Union[str, Callable[[float], float]]]] = None,
        claim_column: Optional[str] = None,
        severity_column: Optional[str] = None,
        severity_cv: float = 0.3,
        claim_cost_column: Optional[str] = None,
        severity_generator: Optional[Callable[[float, float, np.random.Generator], float]] = None,
    ):
        self.name = name
        self.generator = generator
        self.claim_column = claim_column or f"claim_{name}"
        self.severity_column = severity_column
        self.severity_cv = severity_cv
        self.claim_cost_column = claim_cost_column or f"claim_cost_{name}"
        self.severity_generator = severity_generator or _gamma_severity

        # Resolve param_columns defaults (mirrors ClaimsSimulator logic)
        if param_columns is None:
            if generator == "Poisson":
                self.param_columns: Dict[str, Union[str, Callable]] = {"rate": "risk"}
            elif generator == "NegativeBinomialMixture":
                self.param_columns = {"rate": "risk", "dispersion": "dispersion"}
            elif generator == "NegativeBinomialMeanVar":
                self.param_columns = {"rate": "risk", "overdispersion": "overdispersion"}
            elif callable(generator):
                raise ValueError(
                    f"param_columns must be provided for custom generator in source '{name}'"
                )
            else:
                raise ValueError(f"Unknown generator: {generator}")
        else:
            self.param_columns = param_columns

        # Validate required params for built-in generators
        if generator == "Poisson" and "rate" not in self.param_columns:
            raise ValueError(f"Source '{name}': param_columns must contain 'rate' for Poisson")
        elif generator == "NegativeBinomialMixture" and (
            "rate" not in self.param_columns or "dispersion" not in self.param_columns
        ):
            raise ValueError(
                f"Source '{name}': param_columns must contain 'rate' and 'dispersion'"
            )
        elif generator == "NegativeBinomialMeanVar" and (
            "rate" not in self.param_columns or "overdispersion" not in self.param_columns
        ):
            raise ValueError(
                f"Source '{name}': param_columns must contain 'rate' and 'overdispersion'"
            )

        # Resolve time-to-claim function
        if generator == "Poisson":
            self._time_to_claim_func = _poisson_time_to_claim
        elif generator == "NegativeBinomialMixture":
            self._time_to_claim_func = _negative_binomial_mixture_time_to_claim
        elif generator == "NegativeBinomialMeanVar":
            self._time_to_claim_func = _negative_binomial_meanvar_time_to_claim
        elif callable(generator):
            self._time_to_claim_func = generator
        else:
            raise ValueError(f"Unknown generator: {generator}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_params_at_time(self, row: pd.Series, time: float) -> Dict[str, float]:
        """Return parameter dict for *row* evaluated at global *time*."""
        params: Dict[str, float] = {}
        for param_name, param_source in self.param_columns.items():
            if callable(param_source):
                params[param_name] = param_source(time)
            else:
                params[param_name] = row[param_source]
        return params

    def time_to_next_claim(self, row: pd.Series, time: float) -> float:
        """Draw a time-to-next-claim for *row* at global *time*."""
        params = self.get_params_at_time(row, time)
        return self._time_to_claim_func(params)

    def sample_cost(self, row: pd.Series, rng: np.random.Generator) -> float:
        """Sample a claim cost for this source (returns 0.0 if no severity_column)."""
        if self.severity_column is None:
            return 0.0
        mean_cost = row[self.severity_column]
        std_cost = mean_cost * self.severity_cv
        return self.severity_generator(mean_cost, std_cost, rng)

    def validate_columns(self, df: pd.DataFrame) -> None:
        """Raise if any required column is missing from *df*."""
        for param_name, param_source in self.param_columns.items():
            if isinstance(param_source, str) and param_source not in df.columns:
                raise ValueError(
                    f"Source '{self.name}': column '{param_source}' (for parameter "
                    f"'{param_name}') not found in DataFrame"
                )
        if self.severity_column is not None and self.severity_column not in df.columns:
            raise ValueError(
                f"Source '{self.name}': severity_column '{self.severity_column}' "
                "not found in DataFrame"
            )


class MultiCauseClaimsSimulator:
    """
    Simulate claims from multiple independent causes with competing-risks semantics.

    Each cause is modelled as an independent point process (``ClaimSource``).  At
    every simulated interval the first cause to fire determines the next event:

    * The row's exposure is cut at the earliest claim time.
    * The winning cause's claim indicator is set to 1; all other causes are 0.
    * A new row starts immediately after (``renewal_mode='claim'``) or the
      remainder of the current max-exposure window is filled (``'contract_end'``).

    Parameters
    ----------
    sources : list of ClaimSource
        At least one cause must be provided.  Source names must be unique.
    time_to_simulate : str or float
        Column name containing contract duration, or a fixed value.
    max_exposure : float
        Maximum exposure per row (default 1.0).
    exposure_column : str
        Name for the output exposure column (default ``'exposure'``).
    claim_counter : str, optional
        If provided, adds a column counting *total* claims before the current row.
    renewal_mode : {'claim', 'contract_end'}
        Same semantics as ``ClaimsSimulator.renewal_mode``.
    start_time_column : str, optional
        Column with initial time offset per contract.
    random_seed : int, optional
        Seed for reproducibility.

    Examples
    --------
    >>> sources = [
    ...     ClaimSource('fire',  generator='Poisson',
    ...                 param_columns={'rate': 'fire_rate'}),
    ...     ClaimSource('theft', generator='Poisson',
    ...                 param_columns={'rate': 'theft_rate'}),
    ... ]
    >>> sim = MultiCauseClaimsSimulator(sources=sources,
    ...                                  time_to_simulate='duration')
    >>> result = sim.simulate(df)
    """

    def __init__(
        self,
        sources: List[ClaimSource],
        time_to_simulate: Union[str, float] = "contract_duration_years",
        max_exposure: float = 1.0,
        exposure_column: str = "exposure",
        claim_counter: Optional[str] = None,
        renewal_mode: Literal["claim", "contract_end"] = "claim",
        start_time_column: Optional[str] = None,
        random_seed: Optional[int] = None,
    ):
        if not sources:
            raise ValueError("At least one ClaimSource must be provided.")
        names = [s.name for s in sources]
        if len(names) != len(set(names)):
            raise ValueError("ClaimSource names must be unique.")

        self.sources = sources
        self.time_to_simulate = time_to_simulate
        self.max_exposure = max_exposure
        self.exposure_column = exposure_column
        self.claim_counter = claim_counter
        self.renewal_mode = renewal_mode
        self.start_time_column = start_time_column
        self.random_seed = random_seed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def simulate(self, risk_df: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate claims data for all contracts in *risk_df*.

        Parameters
        ----------
        risk_df : pd.DataFrame
            Must contain all columns referenced by the configured sources.

        Returns
        -------
        pd.DataFrame
            One or more rows per contract.  Contains one claim-indicator column
            per source (e.g. ``claim_fire``, ``claim_theft``), plus the shared
            ``exposure`` column.  If any source has a ``severity_column``,
            the corresponding cost column is also included.
        """
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        self._severity_rng = np.random.default_rng(self.random_seed)

        # Validate columns
        for source in self.sources:
            source.validate_columns(risk_df)
        if self.start_time_column is not None and self.start_time_column not in risk_df.columns:
            raise ValueError(
                f"start_time_column '{self.start_time_column}' not found in DataFrame"
            )

        # Resolve contract durations
        if isinstance(self.time_to_simulate, str):
            if self.time_to_simulate not in risk_df.columns:
                raise ValueError(
                    f"time_to_simulate column '{self.time_to_simulate}' not found in DataFrame"
                )
            contract_durations = risk_df[self.time_to_simulate].values
        else:
            contract_durations = np.full(len(risk_df), float(self.time_to_simulate))

        result_rows: List[dict] = []

        for contract_id, (idx, row) in enumerate(risk_df.iterrows()):
            total_duration = float(contract_durations[contract_id])
            start_time = float(row[self.start_time_column]) if self.start_time_column else 0.0

            contract_rows = self._simulate_contract(row, total_duration, start_time)
            for r in contract_rows:
                r["contract_id"] = contract_id
            result_rows.extend(contract_rows)

        return pd.DataFrame(result_rows)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _simulate_contract(
        self, row: pd.Series, total_duration: float, start_time: float
    ) -> List[dict]:
        rows: List[dict] = []
        time_elapsed = 0.0
        claims_so_far = 0  # total claims across all sources before current row

        if self.renewal_mode == "claim":
            while time_elapsed < total_duration:
                t_global = start_time + time_elapsed
                time_elapsed, claims_so_far, interval_rows = self._process_interval_claim_mode(
                    row, time_elapsed, total_duration, t_global, claims_so_far
                )
                rows.extend(interval_rows)
        else:  # contract_end
            while time_elapsed < total_duration:
                t_global = start_time + time_elapsed
                time_elapsed, claims_so_far, interval_rows = self._process_interval_contract_end_mode(
                    row, time_elapsed, total_duration, t_global, claims_so_far
                )
                rows.extend(interval_rows)

        return rows

    def _draw_times(self, row: pd.Series, t_global: float) -> Dict[str, float]:
        """Draw a time-to-next-claim for every source at global time *t_global*."""
        return {s.name: s.time_to_next_claim(row, t_global) for s in self.sources}

    def _build_row(
        self,
        row: pd.Series,
        exposure_time: float,
        winning_source: Optional[str],
        time_elapsed: float,
        claims_so_far: int,
    ) -> dict:
        """
        Build a single output row dict.

        Parameters
        ----------
        winning_source : str or None
            Name of the source that fired (``None`` means no claim in this row).
        """
        row_dict = row.to_dict()
        row_dict[self.exposure_column] = exposure_time
        row_dict["start_time"] = time_elapsed
        row_dict["end_time"] = time_elapsed + exposure_time
        if self.claim_counter is not None:
            row_dict[self.claim_counter] = claims_so_far

        for source in self.sources:
            fired = winning_source == source.name
            row_dict[source.claim_column] = int(fired)
            if source.severity_column is not None:
                row_dict[source.claim_cost_column] = (
                    source.sample_cost(row, self._severity_rng) if fired else 0.0
                )

        return row_dict

    def _process_interval_claim_mode(
        self,
        row: pd.Series,
        time_elapsed: float,
        total_duration: float,
        t_global: float,
        claims_so_far: int,
    ):
        remaining = total_duration - time_elapsed
        exposure_time = min(self.max_exposure, remaining)

        ttc = self._draw_times(row, t_global)
        earliest_source = min(ttc, key=ttc.get)
        earliest_time = ttc[earliest_source]

        if earliest_time <= exposure_time:
            row_dict = self._build_row(row, earliest_time, earliest_source, time_elapsed, claims_so_far)
            time_elapsed += earliest_time
            claims_so_far += 1
        else:
            row_dict = self._build_row(row, exposure_time, None, time_elapsed, claims_so_far)
            time_elapsed += exposure_time

        return time_elapsed, claims_so_far, [row_dict]

    def _process_interval_contract_end_mode(
        self,
        row: pd.Series,
        time_elapsed: float,
        total_duration: float,
        t_global: float,
        claims_so_far: int,
    ):
        rows: List[dict] = []
        remaining = total_duration - time_elapsed
        interval_end = time_elapsed + min(self.max_exposure, remaining)

        interval_start_elapsed = time_elapsed  # remember where this interval started

        while time_elapsed < interval_end:
            time_left = interval_end - time_elapsed
            # Global time at the current position within the interval
            current_t_global = t_global + (time_elapsed - interval_start_elapsed)

            ttc = self._draw_times(row, current_t_global)
            earliest_source = min(ttc, key=ttc.get)
            earliest_time = ttc[earliest_source]

            if earliest_time <= time_left:
                row_dict = self._build_row(row, earliest_time, earliest_source, time_elapsed, claims_so_far)
                rows.append(row_dict)
                time_elapsed += earliest_time
                claims_so_far += 1
            else:
                row_dict = self._build_row(row, time_left, None, time_elapsed, claims_so_far)
                rows.append(row_dict)
                time_elapsed += time_left

        return time_elapsed, claims_so_far, rows
