import numpy as np
import pandas as pd


def gini(t, p, w):
    """Calculates the Gini coefficient from true values, predicted scores, and weights.

    Args:
        t (np.array or list): An array of the true binary outcomes (0 or 1).
        p (np.array or list): An array of the predicted scores or probabilities.
        w (np.array or list): An array of weights for each observation.

    Returns:
        float: The calculated Gini coefficient, a value between -1 and 1.
    """
    data = pd.DataFrame({"t": t, "p": p, "w": w})

    # Handle edge case: empty data
    if len(data) == 0:
        return 0.0

    data = data.sort_values("p", ascending=False)
    data["w"] = data["w"] / data["w"].sum()

    nu_ = np.cumsum(data["t"])

    # Handle edge case: all targets are 0 (no positive events)
    if len(nu_) == 0 or nu_.iloc[-1] == 0:
        return 0.0

    nu_ = nu_ / nu_.iloc[-1]
    nu_ = [0, *list(nu_)]
    dx = np.cumsum(data["w"])
    dx = [0, *list(dx)]

    auc = sum(np.add(nu_[:-1], nu_[1:]) * (np.array(dx[1:]) - np.array(dx[:-1])) / 2)
    gini_coefficient = 2 * auc - 1

    return gini_coefficient


def mae(t, p, w):
    """Calculates Mean Absolute Error (MAE).

    Args:
        t (np.array or list): True values.
        p (np.array or list): Predicted values.
        w (np.array or list): Weights.

    Returns:
        float: MAE.
    """
    t, p, w = np.array(t), np.array(p), np.array(w)
    return np.average(np.abs(t - p), weights=w)


def mse(t, p, w):
    """Calculates Mean Squared Error (MSE).

    Args:
        t (np.array or list): True values.
        p (np.array or list): Predicted values.
        w (np.array or list): Weights.

    Returns:
        float: MSE.
    """
    t, p, w = np.array(t), np.array(p), np.array(w)
    return np.average((t - p) ** 2, weights=w)


def rmse(t, p, w):
    """Calculates Root Mean Squared Error (RMSE).

    Args:
        t (np.array or list): True values.
        p (np.array or list): Predicted values.
        w (np.array or list): Weights.

    Returns:
        float: RMSE.
    """
    return np.sqrt(mse(t, p, w))


def mape(t, p, w):
    """Calculates Mean Absolute Percentage Error (MAPE).

    Args:
        t (np.array or list): True values.
        p (np.array or list): Predicted values.
        w (np.array or list): Weights.

    Returns:
        float: MAPE.
    """
    t, p, w = np.array(t), np.array(p), np.array(w)
    # Filter out zero true values to avoid division by zero
    mask = t != 0
    if not np.any(mask):
        return np.nan
    return np.average(np.abs((t[mask] - p[mask]) / t[mask]), weights=w[mask])


def r2(t, p, w):
    """Calculates R-squared (R2).

    Args:
        t (np.array or list): True values.
        p (np.array or list): Predicted values.
        w (np.array or list): Weights.

    Returns:
        float: R2 score.
    """
    t, p, w = np.array(t), np.array(p), np.array(w)
    t_mean = np.average(t, weights=w)
    ss_res = np.sum(w * (t - p) ** 2)
    ss_tot = np.sum(w * (t - t_mean) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1 - (ss_res / ss_tot)


def _poisson_deviance(t, p, w):
    """Helper to calculate Poisson Deviance."""
    # Ensure p is strictly positive for log
    p_safe = np.clip(p, a_min=1e-15, a_max=None)

    # 2 * w * (t * log(t/p) - (t - p))
    term1 = np.zeros_like(t, dtype=float)
    mask = t > 0
    term1[mask] = t[mask] * np.log(t[mask] / p_safe[mask])

    return 2 * np.sum(w * (term1 - (t - p_safe)))


def poisson_deviance_ratio(t, p, w):
    """Calculates Expected Deviance Ratio for Poisson distribution.

    Defined as 1 - (Deviance(model) / Deviance(null_model)).

    Args:
        t (np.array or list): True values.
        p (np.array or list): Predicted values.
        w (np.array or list): Weights.

    Returns:
        float: Poisson Deviance Ratio.
    """
    t, p, w = np.array(t), np.array(p), np.array(w)
    dev_model = _poisson_deviance(t, p, w)

    t_mean = np.average(t, weights=w)
    p_null = np.full_like(p, t_mean)
    dev_null = _poisson_deviance(t, p_null, w)

    if dev_null == 0:
        return 0.0

    return 1 - (dev_model / dev_null)


def _gamma_deviance(t, p, w):
    """Helper to calculate Gamma Deviance."""
    # Clip to avoid log(0) or div by 0
    epsilon = 1e-15
    t_safe = np.clip(t, a_min=epsilon, a_max=None)
    p_safe = np.clip(p, a_min=epsilon, a_max=None)

    # 2 * w * (-log(t/p) + (t-p)/p)
    term1 = -np.log(t_safe / p_safe)
    term2 = (t_safe - p_safe) / p_safe

    return 2 * np.sum(w * (term1 + term2))


def gamma_deviance_ratio(t, p, w):
    """Calculates Expected Deviance Ratio for Gamma distribution.

    Defined as 1 - (Deviance(model) / Deviance(null_model)).

    Args:
        t (np.array or list): True values.
        p (np.array or list): Predicted values.
        w (np.array or list): Weights.

    Returns:
        float: Gamma Deviance Ratio.
    """
    t, p, w = np.array(t), np.array(p), np.array(w)
    dev_model = _gamma_deviance(t, p, w)

    t_mean = np.average(t, weights=w)
    p_null = np.full_like(p, t_mean)
    dev_null = _gamma_deviance(t, p_null, w)

    if dev_null == 0:
        return 0.0

    return 1 - (dev_model / dev_null)


# =============================================================================
# Calibration Quality Ratio (CQR)
# =============================================================================
#
# CQR measures how well a model's predicted rates match observed rates
# when data is grouped into bins, normalized to a 0–1 scale.
#
# The idea:
#   1. Sort records by predicted rate and split into equal-exposure bins.
#   2. In each bin, compare the observed rate vs. the predicted rate.
#   3. Aggregate those per-bin errors into a single "calibration error".
#   4. Normalize by the spread of predictions (their variance),
#      so the metric lives on [0, 1].
#
# Formula:
#   CQR = max(0, 1 - calibration_error / prediction_spread)
#
#   - CQR = 1 → perfect calibration (observed matches predicted in every bin)
#   - CQR = 0 → calibration error is as large as the prediction spread
#
# Two flavors:
#   - method='mse'   uses squared errors  and Var(pred) as the normalizer
#   - method='mape'  uses absolute percentage errors and MAPD(pred) as the normalizer
#
# Two normalization options:
#   - norm='linear'  →  CQR = 1 − E / S
#   - norm='sqrt'    →  CQR = 1 − √(E / S)   (RMSE-style, used in analytics)
#
# Predictions are rescaled so their weighted mean matches the observed rate
# before any computation, ensuring CQR measures shape mis-calibration only.
# =============================================================================


def _build_equal_exposure_bins(pred, weight, n_bins):
    """Split data into bins with approximately equal total exposure.

    Records are sorted by ``pred``.  Each record is assigned a bin index
    based on where its cumulative weight falls relative to equal-weight
    thresholds.

    Args:
        pred (np.ndarray): Predicted rates, shape (n,).
        weight (np.ndarray): Exposure weights, shape (n,).
        n_bins (int): Desired number of bins.

    Returns:
        np.ndarray: Bin index for every record (0-based), shape (n,).
    """
    sort_idx = np.argsort(pred)
    cum_weight = np.cumsum(weight[sort_idx])
    # Assign each sorted record to a bin via its cumulative-weight fraction
    bin_sorted = np.minimum(
        (cum_weight / cum_weight[-1] * n_bins).astype(int),
        n_bins - 1,
    )
    # Map back to the original record order
    bin_idx = np.empty_like(bin_sorted)
    bin_idx[sort_idx] = bin_sorted
    return bin_idx


def _binned_calibration_error(pred, observation, weight, bin_idx, method):
    """Compute the weighted-average per-bin calibration error.

    For each bin b:
        observed_rate_b  = sum(observation_b) / sum(weight_b)
        predicted_rate_b = weighted_mean(pred_b)
        error_b          = observed_rate_b − predicted_rate_b

    For ``'mse'`` the errors are squared.
    For ``'mape'`` the errors are divided by the predicted rate (percentage).

    Args:
        pred (np.ndarray): Predicted rates, shape (n,).
        observation (np.ndarray): Observed claim counts, shape (n,).
        weight (np.ndarray): Exposure weights, shape (n,).
        bin_idx (np.ndarray): Bin assignment for each record, shape (n,).
        method (str): 'mse' → squared errors, 'mape' → absolute percentage errors.

    Returns:
        float: Weighted average of the per-bin errors.
    """
    n_bins = bin_idx.max() + 1
    w_b = np.bincount(bin_idx, weights=weight, minlength=n_bins)
    obs_b = np.bincount(bin_idx, weights=observation, minlength=n_bins)
    pred_b = np.bincount(bin_idx, weights=pred * weight, minlength=n_bins)

    active = w_b > 0
    obs_rate = obs_b[active] / w_b[active]
    pred_rate = pred_b[active] / w_b[active]
    diff = obs_rate - pred_rate

    if method == 'mse':
        errors = diff ** 2
    else:  # mape
        # |obs - pred| / pred  (pred is always > 0 after rescaling)
        errors = np.abs(diff) / np.maximum(pred_rate, 1e-15)
    return np.average(errors, weights=w_b[active])


def _prediction_spread(pred, weight, method):
    """Weighted spread of predictions: variance (mse) or MAPD (mape).

    These are independent of binning and invariant to shuffling
    (a permutation preserves the marginal distribution).

    Args:
        pred (np.ndarray): Predicted rates, shape (n,).
        weight (np.ndarray): Exposure weights, shape (n,).
        method (str): 'mse' → variance, 'mape' → mean absolute percentage deviation.

    Returns:
        float: The spread measure.
    """
    mean_pred = np.average(pred, weights=weight)
    if method == 'mse':
        return np.average((pred - mean_pred) ** 2, weights=weight)
    else:  # mape
        # Mean absolute percentage deviation:  E[|p - μ| / μ]
        if mean_pred <= 0:
            return 0.0
        return np.average(np.abs(pred - mean_pred) / mean_pred, weights=weight)


def calibration_quality_ratio(pred, observation, weight, n_bins=30,
                              method='mse', norm='linear'):
    """Calibration Quality Ratio (CQR) — calibration quality on a 0–1 scale.

    Steps:
        1. Rescale ``pred`` so its weighted mean equals the observed rate.
        2. Build equal-exposure bins by sorting on ``pred``.
        3. In each bin, measure the gap between observed and predicted rate.
        4. Aggregate into a single calibration error number.
        5. Normalize by the spread of predictions so the result is in [0, 1].

    Args:
        pred (array-like): Predicted rates per unit of exposure.
        observation (array-like): Observed claim counts.
        weight (array-like): Exposure weights.
        n_bins (int): Number of equal-exposure bins (default 30).
        method (str): Error aggregation method.
            ``'mse'``  — squared errors,  normalized by Var(pred).
            ``'mape'``  — absolute percentage errors, normalized by MAPD(pred).
        norm (str): How the ratio E/S is mapped to [0, 1].
            ``'linear'`` — ``CQR = 1 − E / S``  (default).
            ``'sqrt'``   — ``CQR = 1 − √(E / S)``  (RMSE normalisation).

    Returns:
        float: CQR score in [0, 1].  1 = perfect calibration, 0 = worst.
    """
    if method not in ('mse', 'mape'):
        raise ValueError(f"method must be 'mse' or 'mape', got '{method}'")
    if norm not in ('linear', 'sqrt'):
        raise ValueError(f"norm must be 'linear' or 'sqrt', got '{norm}'")

    pred = np.asarray(pred, dtype=float)
    observation = np.asarray(observation, dtype=float)
    weight = np.asarray(weight, dtype=float)

    # Rescale predictions so their weighted mean matches the observed rate.
    # This isolates shape mis-calibration from a global level offset.
    obs_rate = observation.sum() / weight.sum()
    pred_mean = np.average(pred, weights=weight)
    if pred_mean > 0:
        pred = pred * (obs_rate / pred_mean)

    # Normalizer (independent of binning)
    spread = _prediction_spread(pred, weight, method)
    if spread == 0:
        return 0.0              # constant predictions → no calibration info

    # Bin, measure, aggregate
    bin_idx = _build_equal_exposure_bins(pred, weight, n_bins)
    cal_error = _binned_calibration_error(
        pred, observation, weight, bin_idx, method
    )

    # Normalize
    ratio = cal_error / spread
    if norm == 'sqrt':
        ratio = np.sqrt(ratio)
    return max(0.0, 1.0 - ratio)