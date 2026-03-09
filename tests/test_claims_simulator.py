import pytest
import numpy as np
import pandas as pd
from claimsimulator.simulation.claims_simulator import ClaimsSimulator


def test_poisson_claims_basic():
    """Test basic Poisson claims simulation"""
    risk_df = pd.DataFrame({
        'policy_id': [1, 2, 3],
        'risk': [0.1, 0.5, 1.0],
        'duration': [2.0, 3.0, 1.0]
    })

    simulator = ClaimsSimulator(
        generator='Poisson',
        param_columns={'rate': 'risk'},
        time_to_simulate='duration',
        max_exposure=1.0,
        random_seed=42
    )
    result = simulator.simulate(risk_df=risk_df)

    # Verify output structure
    assert 'exposure' in result.columns
    assert 'claim' in result.columns
    assert 'start_time' in result.columns
    assert 'end_time' in result.columns

    # Verify we have at least as many rows as input
    assert len(result) >= len(risk_df)

    # Verify claim is binary
    assert result['claim'].isin([0, 1]).all()

    # Verify exposure is positive
    assert (result['exposure'] > 0).all()


def test_max_exposure_respected():
    """Test that max_exposure constraint is respected"""
    risk_df = pd.DataFrame({
        'risk': [0.01],  # Low risk to avoid claims
        'duration': [5.0]
    })

    simulator = ClaimsSimulator(
        generator='Poisson',
        param_columns={'rate': 'risk'},
        time_to_simulate='duration',
        max_exposure=1.0,
        random_seed=42
    )
    result = simulator.simulate(risk_df=risk_df)

    # All exposures should be <= max_exposure
    assert (result['exposure'] <= 1.0).all()


def test_time_continuity():
    """Test that time intervals are continuous"""
    risk_df = pd.DataFrame({
        'risk': [0.5],
        'duration': [3.0]
    })

    simulator = ClaimsSimulator(
        generator='Poisson',
        param_columns={'rate': 'risk'},
        time_to_simulate='duration',
        max_exposure=1.0,
        random_seed=42
    )
    result = simulator.simulate(risk_df=risk_df)

    result = result.sort_values('start_time')

    # Verify end_time = start_time + exposure
    np.testing.assert_array_almost_equal(
        result['end_time'].values,
        result['start_time'].values + result['exposure'].values
    )

    # Verify continuity (next start_time = previous end_time)
    if len(result) > 1:
        for i in range(len(result) - 1):
            assert np.isclose(result.iloc[i]['end_time'], result.iloc[i + 1]['start_time'])


def test_total_duration_respected():
    """Test that total simulated time matches contract duration"""
    risk_df = pd.DataFrame({
        'risk': [0.3],
        'duration': [4.5]
    })

    simulator = ClaimsSimulator(
        generator='Poisson',
        param_columns={'rate': 'risk'},
        time_to_simulate='duration',
        max_exposure=1.0,
        random_seed=42
    )
    result = simulator.simulate(risk_df=risk_df)

    total_exposure = result['exposure'].sum()
    assert np.isclose(total_exposure, 4.5, atol=1e-6)


def test_claim_at_end_of_interval():
    """Test that claims are generated with high-risk contracts"""
    risk_df = pd.DataFrame({
        'risk': [2.0] * 10,
        'duration': [5.0] * 10
    })

    simulator = ClaimsSimulator(
        generator='Poisson',
        param_columns={'rate': 'risk'},
        time_to_simulate='duration',
        max_exposure=1.0,
        random_seed=42
    )
    result = simulator.simulate(risk_df=risk_df)

    claim_rows = result[result['claim'] == 1]
    assert len(claim_rows) > 0


def test_negative_binomial_mixture_simulation():
    """Test NegativeBinomialMixture claims simulation"""
    risk_df = pd.DataFrame({
        'risk': [0.5, 1.0],
        'dispersion': [0.5, 0.5],
        'duration': [3.0, 3.0]
    })

    simulator = ClaimsSimulator(
        generator='NegativeBinomialMixture',
        param_columns={'rate': 'risk', 'dispersion': 'dispersion'},
        time_to_simulate='duration',
        max_exposure=1.0,
        random_seed=42
    )
    result = simulator.simulate(risk_df=risk_df)

    assert 'exposure' in result.columns
    assert 'claim' in result.columns
    assert len(result) >= len(risk_df)


def test_negative_binomial_meanvar_simulation():
    """Test NegativeBinomialMeanVar claims simulation"""
    risk_df = pd.DataFrame({
        'risk': [0.5, 1.0],
        'overdispersion': [0.3, 0.3],
        'duration': [3.0, 3.0]
    })

    simulator = ClaimsSimulator(
        generator='NegativeBinomialMeanVar',
        param_columns={'rate': 'risk', 'overdispersion': 'overdispersion'},
        time_to_simulate='duration',
        max_exposure=1.0,
        random_seed=42
    )
    result = simulator.simulate(risk_df=risk_df)

    assert 'exposure' in result.columns
    assert 'claim' in result.columns
    assert len(result) >= len(risk_df)


def test_fixed_time_to_simulate():
    """Test using a fixed value for time_to_simulate"""
    risk_df = pd.DataFrame({
        'risk': [0.2, 0.4, 0.6]
    })

    simulator = ClaimsSimulator(
        generator='Poisson',
        param_columns={'rate': 'risk'},
        time_to_simulate=2.5,
        max_exposure=1.0,
        random_seed=42
    )
    result = simulator.simulate(risk_df=risk_df)

    total_exposures = result.groupby('contract_id')['exposure'].sum()
    assert np.allclose(total_exposures, 2.5)


def test_custom_column_names():
    """Test using custom column names"""
    risk_df = pd.DataFrame({
        'my_risk': [0.3],
        'my_duration': [2.0]
    })

    simulator = ClaimsSimulator(
        generator='Poisson',
        param_columns={'rate': 'my_risk'},
        claim_column='has_claim',
        exposure_column='time_exposed',
        time_to_simulate='my_duration',
        random_seed=42
    )
    result = simulator.simulate(risk_df=risk_df)

    assert 'has_claim' in result.columns
    assert 'time_exposed' in result.columns


def test_zero_risk():
    """Test behavior with zero risk (no claims expected)"""
    risk_df = pd.DataFrame({
        'risk': [0.0],
        'duration': [3.0]
    })

    simulator = ClaimsSimulator(
        generator='Poisson',
        param_columns={'rate': 'risk'},
        time_to_simulate='duration',
        max_exposure=1.0,
        random_seed=42
    )
    result = simulator.simulate(risk_df=risk_df)

    assert len(result) >= 1
    assert result['claim'].sum() == 0


def test_reproducibility():
    """Test that results are reproducible with same random seed"""
    risk_df = pd.DataFrame({
        'risk': [0.5, 1.0, 0.3],
        'duration': [2.0, 3.0, 1.5]
    })

    simulator = ClaimsSimulator(
        generator='Poisson',
        param_columns={'rate': 'risk'},
        time_to_simulate='duration',
        random_seed=123
    )

    result1 = simulator.simulate(risk_df=risk_df)
    result2 = simulator.simulate(risk_df=risk_df)

    pd.testing.assert_frame_equal(result1, result2)


def test_claim_counter():
    """Test that claim_counter tracks cumulative claims per contract"""
    risk_df = pd.DataFrame({
        'risk': [3.0],
        'duration': [10.0]
    })

    simulator = ClaimsSimulator(
        generator='Poisson',
        param_columns={'rate': 'risk'},
        time_to_simulate='duration',
        max_exposure=1.0,
        claim_counter='n_previous_claims',
        random_seed=42
    )
    result = simulator.simulate(risk_df=risk_df)

    assert 'n_previous_claims' in result.columns
    # First row should have 0 previous claims
    assert result.iloc[0]['n_previous_claims'] == 0
    # Counter should be non-decreasing
    assert (result['n_previous_claims'].diff().dropna() >= 0).all()


def test_contract_end_renewal_mode():
    """Test contract_end renewal mode"""
    risk_df = pd.DataFrame({
        'risk': [2.0],
        'duration': [3.0]
    })

    simulator = ClaimsSimulator(
        generator='Poisson',
        param_columns={'rate': 'risk'},
        time_to_simulate='duration',
        max_exposure=1.0,
        renewal_mode='contract_end',
        random_seed=42
    )
    result = simulator.simulate(risk_df=risk_df)

    assert 'exposure' in result.columns
    assert 'claim' in result.columns
    total_exposure = result['exposure'].sum()
    assert np.isclose(total_exposure, 3.0, atol=1e-6)


def test_custom_generator_requires_param_columns():
    """Test that a custom generator function requires param_columns"""
    def custom_gen(params):
        return np.random.exponential(1.0 / params['rate'])

    with pytest.raises(ValueError, match="param_columns must be provided"):
        ClaimsSimulator(generator=custom_gen)


def test_custom_generator():
    """Test using a custom generator callable"""
    def custom_gen(params):
        return np.random.exponential(1.0 / params['rate']) if params['rate'] > 0 else np.inf

    risk_df = pd.DataFrame({
        'risk': [0.5],
        'duration': [3.0]
    })

    simulator = ClaimsSimulator(
        generator=custom_gen,
        param_columns={'rate': 'risk'},
        time_to_simulate='duration',
        max_exposure=1.0,
        random_seed=42
    )
    result = simulator.simulate(risk_df=risk_df)

    assert 'exposure' in result.columns
    assert 'claim' in result.columns


# ── Severity / claim-cost tests ───────────────────────────────────────


def test_severity_column_produces_claim_cost():
    """When severity_column is set, claim rows get a positive cost and non-claim rows get 0."""
    risk_df = pd.DataFrame({
        'risk': [3.0] * 20,
        'severity': [5000.0] * 20,
        'duration': [5.0] * 20,
    })

    simulator = ClaimsSimulator(
        generator='Poisson',
        param_columns={'rate': 'risk'},
        time_to_simulate='duration',
        max_exposure=1.0,
        severity_column='severity',
        severity_cv=0.2,
        random_seed=42,
    )
    result = simulator.simulate(risk_df)

    assert 'claim_cost' in result.columns

    # Non-claim rows must have 0 cost
    assert (result.loc[result['claim'] == 0, 'claim_cost'] == 0.0).all()

    # Claim rows must have positive cost
    claim_rows = result.loc[result['claim'] == 1]
    assert len(claim_rows) > 0, "Expected at least one claim with rate=3"
    assert (claim_rows['claim_cost'] > 0).all()


def test_severity_mean_approximately_correct():
    """Average sampled cost should be close to the severity_column mean."""
    n = 200
    risk_df = pd.DataFrame({
        'risk': [5.0] * n,
        'severity': [3000.0] * n,
        'duration': [10.0] * n,
    })

    simulator = ClaimsSimulator(
        generator='Poisson',
        param_columns={'rate': 'risk'},
        time_to_simulate='duration',
        max_exposure=1.0,
        severity_column='severity',
        severity_cv=0.15,
        random_seed=42,
    )
    result = simulator.simulate(risk_df)

    claim_costs = result.loc[result['claim'] == 1, 'claim_cost']
    assert len(claim_costs) > 50, "Need enough claims for a meaningful average"
    assert abs(claim_costs.mean() - 3000.0) < 300, (
        f"Mean claim cost {claim_costs.mean():.0f} too far from 3000"
    )


def test_no_severity_column_no_cost_column():
    """When severity_column is None the output should NOT contain a cost column."""
    risk_df = pd.DataFrame({
        'risk': [0.5],
        'duration': [2.0],
    })

    simulator = ClaimsSimulator(
        generator='Poisson',
        param_columns={'rate': 'risk'},
        time_to_simulate='duration',
        random_seed=42,
    )
    result = simulator.simulate(risk_df)

    assert 'claim_cost' not in result.columns


def test_severity_missing_column_raises():
    """Referencing a missing severity column should raise ValueError."""
    risk_df = pd.DataFrame({
        'risk': [0.5],
        'duration': [2.0],
    })

    simulator = ClaimsSimulator(
        generator='Poisson',
        param_columns={'rate': 'risk'},
        time_to_simulate='duration',
        severity_column='nonexistent',
        random_seed=42,
    )

    with pytest.raises(ValueError, match="severity_column"):
        simulator.simulate(risk_df)


def test_custom_severity_generator():
    """A user-supplied severity generator should be used instead of the default."""
    def fixed_cost(mean, std, rng):
        return mean * 2.0   # always double the mean

    risk_df = pd.DataFrame({
        'risk': [5.0] * 10,
        'severity': [1000.0] * 10,
        'duration': [5.0] * 10,
    })

    simulator = ClaimsSimulator(
        generator='Poisson',
        param_columns={'rate': 'risk'},
        time_to_simulate='duration',
        max_exposure=1.0,
        severity_column='severity',
        severity_generator=fixed_cost,
        random_seed=42,
    )
    result = simulator.simulate(risk_df)

    claim_costs = result.loc[result['claim'] == 1, 'claim_cost']
    assert len(claim_costs) > 0
    assert (claim_costs == 2000.0).all()


def test_custom_claim_cost_column_name():
    """The output column name should respect claim_cost_column."""
    risk_df = pd.DataFrame({
        'risk': [3.0] * 5,
        'severity': [2000.0] * 5,
        'duration': [3.0] * 5,
    })

    simulator = ClaimsSimulator(
        generator='Poisson',
        param_columns={'rate': 'risk'},
        time_to_simulate='duration',
        severity_column='severity',
        claim_cost_column='loss_amount',
        random_seed=42,
    )
    result = simulator.simulate(risk_df)

    assert 'loss_amount' in result.columns
    assert 'claim_cost' not in result.columns
