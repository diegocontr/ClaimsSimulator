"""
Claims Simulator - Generate claim events for piecewise exponential models
"""

from typing import Union, Optional, Literal, Dict, Callable
import pandas as pd
import numpy as np


def _poisson_time_to_claim(params: Dict[str, float]) -> float:
    """
    Generate time to next claim using Poisson process (exponential inter-arrival).
    
    Parameters
    ----------
    params : dict
        Must contain 'rate' parameter
        
    Returns
    -------
    float
        Time until next claim
    """
    rate = params['rate']
    return np.random.exponential(1.0 / rate) if rate > 0 else np.inf


def _negative_binomial_mixture_time_to_claim(params: Dict[str, float]) -> float:
    """
    Generate time to next claim using Gamma-Poisson mixture (unobserved heterogeneity).
    
    Parameters
    ----------
    params : dict
        Must contain 'rate' and 'dispersion' parameters
        
    Returns
    -------
    float
        Time until next claim
    """
    rate = params['rate']
    dispersion = params['dispersion']
    
    if dispersion > 0 and rate > 0:
        shape = 1.0 / dispersion
        scale = rate * dispersion
        effective_rate = np.random.gamma(shape, scale)
        return np.random.exponential(1.0 / effective_rate) if effective_rate > 0 else np.inf
    else:
        return np.inf


def _negative_binomial_meanvar_time_to_claim(params: Dict[str, float]) -> float:
    """
    Generate time to next claim using proper NB parameterization.
    Ensures E[claims/exposure] = rate.
    
    Parameters
    ----------
    params : dict
        Must contain 'rate' and 'overdispersion' parameters
        
    Returns
    -------
    float
        Time until next claim
    """
    rate = params['rate']
    overdispersion = params['overdispersion']
    
    if overdispersion > 0 and rate > 0:
        # Convert to NB parameters
        # E[N] = rate * exposure
        # Var[N] = rate * exposure * (1 + overdispersion)
        # For NB(r, p): E = r(1-p)/p, Var = r(1-p)/p^2
        # Set mean = rate, var = rate * (1 + overdispersion)
        # This gives: r = rate / overdispersion, p = 1 / (1 + overdispersion)
        r = 1.0 / overdispersion  # shape parameter
        # Generate effective rate from gamma with shape r, scale rate * overdispersion
        # This gives E[effective_rate] = r * (rate * overdispersion) = rate
        # And Var[effective_rate] = r * (rate * overdispersion)^2 = rate^2 * overdispersion
        effective_rate = np.random.gamma(r, rate * overdispersion)
        return np.random.exponential(1.0 / effective_rate) if effective_rate > 0 else np.inf
    else:
        return np.inf


def _gamma_severity(mean: float, std: float, rng: np.random.Generator) -> float:
    """Sample a single claim cost from a Gamma distribution.

    Parameters
    ----------
    mean : float
        Expected claim cost (must be > 0).
    std : float
        Standard deviation of claim cost (must be > 0).
    rng : numpy.random.Generator
        Random number generator instance.

    Returns
    -------
    float
        Sampled claim cost.
    """
    if mean <= 0 or std <= 0:
        return 0.0
    shape = (mean / std) ** 2
    scale = std ** 2 / mean
    return float(rng.gamma(shape, scale))


class ClaimsSimulator:
    """
    Simulate claims data for training piecewise exponential proportional hazards models.
    
    Each contract is split into multiple rows (images) based on claim events:
    - If no claim occurs: one row with exposure = min(max_exposure, remaining_time), claim = 0
    - If claim occurs at time t: row with exposure = t, claim = 1, then continue with new row
    
    This format is suitable for training survival models with time-varying covariates.
    """
    
    def __init__(self,
                 generator: Union[Literal['Poisson', 'NegativeBinomialMixture', 'NegativeBinomialMeanVar'], Callable[[Dict[str, float]], float]] = 'Poisson',
                 param_columns: Optional[Dict[str, Union[str, Callable[[float], float]]]] = None,
                 claim_column: str = 'claim',
                 time_to_simulate: Union[str, float] = 'contract_duration_years',
                 max_exposure: float = 1.0,
                 exposure_column: str = 'exposure',
                 claim_counter: Optional[str] = None,
                 renewal_mode: Literal['claim', 'contract_end'] = 'claim',
                 start_time_column: Optional[str] = None,
                 severity_column: Optional[str] = None,
                 severity_generator: Optional[Callable[[float, float, np.random.Generator], float]] = None,
                 severity_cv: float = 0.3,
                 claim_cost_column: str = 'claim_cost',
                 random_seed: Optional[int] = None):
        """
        Initialize the claims simulator with configuration.
        
        Parameters
        ----------
        generator : str or callable
            Distribution to use for generating claims. Can be:
            - 'Poisson': Exponential inter-arrival times with rate from 'rate' column
            - 'NegativeBinomialMixture': Gamma-Poisson mixture (unobserved heterogeneity)
              Requires: 'rate' (mean rate) and 'dispersion' (shape parameter)
            - 'NegativeBinomialMeanVar': Proper NB parameterization where E[claims/exposure] = rate
              Requires: 'rate' (mean) and 'overdispersion' where Var = Mean * (1 + overdispersion)
            - Callable: Custom function with signature (params: Dict[str, float]) -> float
              that returns time to next claim
        param_columns : dict, optional
            Dictionary mapping parameter names to either:
            - Column names (str): Static parameter values extracted from DataFrame columns
            - Callable functions: Time-varying parameters evaluated as function(t) where t is
              the global time since contract start (or since start_time if start_time_column provided)
            Required parameters depend on generator:
            - Poisson: {'rate': 'column_name'} or {'rate': lambda t: ...}
            - NegativeBinomialMixture: {'rate': ..., 'dispersion': ...}
            - NegativeBinomialMeanVar: {'rate': ..., 'overdispersion': ...}
            - Custom callable: depends on your function's requirements
            If not provided, defaults to {'rate': 'risk'} for Poisson
            Time-varying parameters are evaluated at each interval boundary (piecewise constant per interval)
        claim_column : str
            Name for the output claim indicator column (default 'claim')
        time_to_simulate : str or float
            Column name containing contract duration, or fixed value for all contracts
            (default 'contract_duration_years')
        max_exposure : float
            Maximum exposure time per row (default 1.0 year)
        exposure_column : str
            Name for the output exposure column (default 'exposure')
        claim_counter : str, optional
            If provided, adds a column with this name that counts the number of claims
            that occurred before the current image/row in the contract (default None)
        renewal_mode : {'claim', 'contract_end'}
            Controls how contracts are split into images/rows:
            - 'claim': (default) Contract renews immediately after each claim.
              When a claim occurs at time t, creates one row with exposure=t and claim=1,
              then starts a new row from time t.
            - 'contract_end': Contract renews at max_exposure intervals regardless of claims.
              When a claim occurs at time t within an interval [0, max_exposure], 
              creates one row with exposure=t and claim=1, then another row with 
              exposure=(max_exposure - t) and claim=0 to complete the interval.
        start_time_column : str, optional
            If provided, specifies the column containing the initial time offset for each contract.
            When set, global time t = start_time + time_elapsed. This allows modeling contracts
            that don't start at t=0 (e.g., contract continuations or staggered starts).
            If None, contracts start at t=0 (default None)
        severity_column : str, optional
            Column name in the input DataFrame that contains the **mean** claim cost
            for each contract.  When provided the simulator generates a ``claim_cost``
            for every claim row and sets it to 0.0 for non-claim rows.
            If ``None`` (default), no cost column is produced.
        severity_generator : callable, optional
            A function ``f(mean, std, rng) -> float`` that samples a single claim
            cost.  *mean* comes from ``severity_column``, *std* is computed as
            ``mean × severity_cv``, and *rng* is a ``numpy.random.Generator``.
            Defaults to a Gamma distribution (``_gamma_severity``).
        severity_cv : float
            Coefficient of variation for claim costs.  The actual standard
            deviation passed to ``severity_generator`` is computed as
            ``mean_cost × severity_cv`` (default 0.3, i.e. 30 % of the mean).
        claim_cost_column : str
            Name for the output claim-cost column (default ``'claim_cost'``).
        random_seed : int, optional
            Random seed for reproducibility
        """
        self.generator = generator
        self.claim_column = claim_column
        self.time_to_simulate = time_to_simulate
        self.max_exposure = max_exposure
        self.exposure_column = exposure_column
        self.claim_counter = claim_counter
        self.renewal_mode = renewal_mode
        self.start_time_column = start_time_column
        self.severity_column = severity_column
        self.severity_generator = severity_generator or _gamma_severity
        self.severity_cv = severity_cv
        self.claim_cost_column = claim_cost_column
        self.random_seed = random_seed
        
        # Set default param_columns if not provided
        if param_columns is None:
            if generator == 'Poisson':
                self.param_columns = {'rate': 'risk'}
            elif generator == 'NegativeBinomialMixture':
                self.param_columns = {'rate': 'risk', 'dispersion': 'dispersion'}
            elif generator == 'NegativeBinomialMeanVar':
                self.param_columns = {'rate': 'risk', 'overdispersion': 'overdispersion'}
            elif callable(generator):
                # For custom generator, user must provide param_columns
                raise ValueError("param_columns must be provided when using a custom generator function")
            else:
                raise ValueError(f"Unknown generator: {generator}")
        else:
            self.param_columns = param_columns
        
        # Validate inputs for built-in generators
        if generator == 'Poisson':
            if 'rate' not in self.param_columns:
                raise ValueError("param_columns must contain 'rate' for Poisson generator")
        elif generator == 'NegativeBinomialMixture':
            if 'rate' not in self.param_columns or 'dispersion' not in self.param_columns:
                raise ValueError("param_columns must contain 'rate' and 'dispersion' for NegativeBinomialMixture")
        elif generator == 'NegativeBinomialMeanVar':
            if 'rate' not in self.param_columns or 'overdispersion' not in self.param_columns:
                raise ValueError("param_columns must contain 'rate' and 'overdispersion' for NegativeBinomialMeanVar")
        # For custom callable, we skip validation as requirements are user-defined
        
        # Map generator to the appropriate function
        if generator == 'Poisson':
            self._time_to_claim_func = _poisson_time_to_claim
        elif generator == 'NegativeBinomialMixture':
            self._time_to_claim_func = _negative_binomial_mixture_time_to_claim
        elif generator == 'NegativeBinomialMeanVar':
            self._time_to_claim_func = _negative_binomial_meanvar_time_to_claim
        elif callable(generator):
            self._time_to_claim_func = generator
        else:
            raise ValueError(f"Unknown generator: {generator}")
    
    def simulate(self, risk_df: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate claims data with multiple rows per contract based on claim events.
        
        Parameters
        ----------
        risk_df : pd.DataFrame
            DataFrame with risk scores and contract information.
            Must contain columns specified in param_columns.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with one or more rows per contract, including exposure and claim columns
            
        Notes
        -----
        The output format is designed for piecewise exponential PH models:
        - Each contract can have multiple rows (images)
        - exposure: time exposed to risk in this interval
        - claim: 1 if claim occurred at end of interval, 0 otherwise
        - If claim occurs, a new row starts immediately after
        """
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
        # Create a Generator for severity sampling (separate from legacy np.random)
        self._severity_rng = np.random.default_rng(self.random_seed)
        
        # Validate inputs - check that required columns exist
        for param_name, param_source in self.param_columns.items():
            # Only validate column names (static parameters), not callables
            if isinstance(param_source, str) and param_source not in risk_df.columns:
                raise ValueError(f"Column '{param_source}' (for parameter '{param_name}') not found in DataFrame")
        
        # Validate severity_column if provided
        if self.severity_column is not None:
            if self.severity_column not in risk_df.columns:
                raise ValueError(f"severity_column '{self.severity_column}' not found in DataFrame")
        
        # Validate start_time_column if provided
        if self.start_time_column is not None:
            if self.start_time_column not in risk_df.columns:
                raise ValueError(f"start_time_column '{self.start_time_column}' not found in DataFrame")
        
        # Get contract durations
        if isinstance(self.time_to_simulate, str):
            if self.time_to_simulate not in risk_df.columns:
                raise ValueError(f"time_to_simulate column '{self.time_to_simulate}' not found in DataFrame")
            contract_durations = risk_df[self.time_to_simulate].values
        else:
            contract_durations = np.full(len(risk_df), self.time_to_simulate)
        
        # Process each contract
        result_rows = []
        
        for contract_id, (idx, row) in enumerate(risk_df.iterrows()):
            total_duration = contract_durations[contract_id] if isinstance(self.time_to_simulate, str) else self.time_to_simulate
            
            # Get start time for this contract
            start_time = row[self.start_time_column] if self.start_time_column is not None else 0.0
            
            # Simulate claims for this contract
            contract_rows = self._simulate_contract(
                row=row,
                total_duration=total_duration,
                max_exposure=self.max_exposure,
                generator=self.generator,
                start_time=start_time,
                renewal_mode=self.renewal_mode
            )
            
            # Add contract_id and exposure/claim columns
            for contract_row in contract_rows:
                contract_row['contract_id'] = contract_id
                result_rows.append(contract_row)
        
        # Create result DataFrame
        claim_df = pd.DataFrame(result_rows)
        
        # Rename columns
        if 'exposure_time' in claim_df.columns:
            claim_df = claim_df.rename(columns={'exposure_time': self.exposure_column})
        if 'claim_occurred' in claim_df.columns:
            claim_df = claim_df.rename(columns={'claim_occurred': self.claim_column})
        
        return claim_df
    
    def _get_params_at_time(self, row: pd.Series, time: float) -> Dict[str, float]:
        """
        Extract parameters for the current time point.
        
        Parameters
        ----------
        row : pd.Series
            The contract row containing static parameter values
        time : float
            The global time since contract start (or since start_time if start_time_column provided)
            
        Returns
        -------
        dict
            Dictionary of parameter values evaluated at the specified time
        """
        params = {}
        for param_name, param_source in self.param_columns.items():
            if callable(param_source):
                # Time-varying parameter: evaluate function at current time
                params[param_name] = param_source(time)
            else:
                # Static parameter: extract from DataFrame column
                params[param_name] = row[param_source]
        return params
    
    def _create_row_dict(self, 
                        row: pd.Series, 
                        exposure_time: float, 
                        claim_occurred: int,
                        time_elapsed: float,
                        claims_so_far: int) -> dict:
        """
        Create a row dictionary with all necessary fields.
        
        Parameters
        ----------
        row : pd.Series
            Original row data from risk_df
        exposure_time : float
            Exposure time for this row
        claim_occurred : int
            1 if claim occurred, 0 otherwise
        time_elapsed : float
            Current time elapsed in contract
        claims_so_far : int
            Number of claims before this row
            
        Returns
        -------
        dict
            Row dictionary with all fields
        """
        row_dict = row.to_dict()
        row_dict['exposure_time'] = exposure_time
        row_dict['claim_occurred'] = claim_occurred
        row_dict['start_time'] = time_elapsed
        row_dict['end_time'] = time_elapsed + exposure_time
        if self.claim_counter is not None:
            row_dict[self.claim_counter] = claims_so_far
        # Generate claim cost when severity modelling is enabled
        if self.severity_column is not None:
            if claim_occurred == 1:
                mean_cost = row[self.severity_column]
                std_cost = mean_cost * self.severity_cv
                row_dict[self.claim_cost_column] = self.severity_generator(
                    mean_cost, std_cost, self._severity_rng,
                )
            else:
                row_dict[self.claim_cost_column] = 0.0
        return row_dict
    
    def _simulate_contract(self,
                          row: pd.Series,
                          total_duration: float,
                          max_exposure: float,
                          generator: Union[str, Callable],
                          start_time: float,
                          renewal_mode: str = 'claim') -> list:
        """
        Simulate claims for a single contract, creating multiple rows as needed.
        
        Parameters
        ----------
        row : pd.Series
            Original row data from risk_df
        total_duration : float
            Total time to simulate
        max_exposure : float
            Maximum exposure per row
        generator : str or callable
            Generator type or custom function (stored for reference, actual function is in self._time_to_claim_func)
        start_time : float
            The initial time offset for this contract (from start_time_column, or 0.0 if not provided)
        renewal_mode : str
            'claim' or 'contract_end' - controls how contract splits into images
            
        Returns
        -------
        list of dict
            List of row dictionaries for this contract
        """
        rows = []
        time_elapsed = 0.0
        claims_so_far = 0  # Track number of claims before current row
        
        if renewal_mode == 'claim':
            # Original logic: contract renews immediately after each claim
            while time_elapsed < total_duration:
                # Calculate global time and get parameters at this time
                t_global = start_time + time_elapsed
                params = self._get_params_at_time(row, t_global)
                
                time_elapsed, claims_so_far, interval_rows = self._process_interval_claim_mode(
                    row, time_elapsed, total_duration, max_exposure, params, claims_so_far
                )
                rows.extend(interval_rows)
        else:  # renewal_mode == 'contract_end'
            # New logic: contract renews at max_exposure intervals
            while time_elapsed < total_duration:
                # Calculate global time and get parameters at this time
                t_global = start_time + time_elapsed
                params = self._get_params_at_time(row, t_global)
                
                time_elapsed, claims_so_far, interval_rows = self._process_interval_contract_end_mode(
                    row, time_elapsed, total_duration, max_exposure, params, claims_so_far
                )
                rows.extend(interval_rows)
        
        return rows
    
    def _process_interval_claim_mode(self,
                                     row: pd.Series,
                                     time_elapsed: float,
                                     total_duration: float,
                                     max_exposure: float,
                                     params: Dict[str, float],
                                     claims_so_far: int) -> tuple:
        """
        Process one interval in 'claim' renewal mode.
        
        Returns
        -------
        tuple
            (new_time_elapsed, new_claims_so_far, rows_generated)
        """
        rows = []
        remaining_time = total_duration - time_elapsed
        exposure_time = min(max_exposure, remaining_time)
        
        # Generate time to next claim
        time_to_claim = self._time_to_claim_func(params)
        
        if time_to_claim <= exposure_time:
            # Claim occurs at time_to_claim
            row_dict = self._create_row_dict(row, time_to_claim, 1, time_elapsed, claims_so_far)
            rows.append(row_dict)
            time_elapsed += time_to_claim
            claims_so_far += 1
        else:
            # No claim in this period
            row_dict = self._create_row_dict(row, exposure_time, 0, time_elapsed, claims_so_far)
            rows.append(row_dict)
            time_elapsed += exposure_time
        
        return time_elapsed, claims_so_far, rows
    
    def _process_interval_contract_end_mode(self,
                                           row: pd.Series,
                                           time_elapsed: float,
                                           total_duration: float,
                                           max_exposure: float,
                                           params: Dict[str, float],
                                           claims_so_far: int) -> tuple:
        """
        Process one interval in 'contract_end' renewal mode.
        Processes all claims within a single max_exposure interval.
        
        Returns
        -------
        tuple
            (new_time_elapsed, new_claims_so_far, rows_generated)
        """
        rows = []
        remaining_time = total_duration - time_elapsed
        interval_end = time_elapsed + min(max_exposure, remaining_time)
        
        # Process claims within this interval
        while time_elapsed < interval_end:
            time_left_in_interval = interval_end - time_elapsed
            
            # Generate time to next claim
            time_to_claim = self._time_to_claim_func(params)
            
            if time_to_claim <= time_left_in_interval:
                # Claim occurs within this interval
                row_dict = self._create_row_dict(row, time_to_claim, 1, time_elapsed, claims_so_far)
                rows.append(row_dict)
                time_elapsed += time_to_claim
                claims_so_far += 1
            else:
                # No claim in remaining time of this interval
                row_dict = self._create_row_dict(row, time_left_in_interval, 0, time_elapsed, claims_so_far)
                rows.append(row_dict)
                time_elapsed += time_left_in_interval
        
        return time_elapsed, claims_so_far, rows
