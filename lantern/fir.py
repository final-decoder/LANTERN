"""FIR response estimation and area integration."""

import numpy as np
from scipy import linalg


def make_fir_design(timeseries, design, tr, horizon, hrf_len=None):
    """Build a design matrix of lagged condition regressors.

    Parameters
    ----------
    timeseries : (n_timepoints,) array
    design : (n_conditions, n_timepoints) array
    tr : float
        Repetition time in seconds.
    horizon : float
        Maximum lag to include, in seconds.
    hrf_len : int, optional
        Number of FIR bins; defaults to int(horizon / tr).

    Returns
    -------
    X : (n_timepoints, n_conditions * n_lags) array
    """
    n_conditions = design.shape[0]
    n_lags = hrf_len or int(np.ceil(horizon / tr))
    n_time = timeseries.shape[0]
    X = np.zeros((n_time, n_conditions * n_lags))
    for j in range(n_conditions):
        for lag in range(n_lags):
            col = j * n_lags + lag
            X[lag:, col] = design[j, : n_time - lag]
    return X


def estimate_fir_area(
    timeseries,
    design,
    tr,
    horizon,
    nuisance=None,
    lesion_mask=None,
):
    """Estimate per-region FIR impulse responses and integrate their area.

    Implements stage 1 of the pipeline: fit a first-level model with one
    regressor per condition and per lag, then sum the estimated coefficients
    over the finite horizon to obtain M_{rj} = \\int F_{rj}(t) dt.

    Parameters
    ----------
    timeseries : (n_timepoints, n_regions) array
    design : (n_conditions, n_timepoints) array
    tr : float
    horizon : float
    nuisance : (n_timepoints, n_nuisance) array, optional
    lesion_mask : (n_regions,) bool array, optional
        True for parcels that are unavailable; they are excluded from fitting.

    Returns
    -------
    M : (n_regions, n_conditions) array
        Integrated response matrix; rows for unavailable regions are set to 0.
    """
    n_time, n_regions = timeseries.shape
    n_conditions = design.shape[0]
    X = make_fir_design(timeseries[:, 0], design, tr, horizon)
    if nuisance is not None:
        X = np.column_stack([X, nuisance])

    M = np.zeros((n_regions, n_conditions))
    n_lags = int(np.ceil(horizon / tr))
    for r in range(n_regions):
        if lesion_mask is not None and lesion_mask[r]:
            continue
        beta = linalg.lstsq(X, timeseries[:, r])[0]
        fir = beta[: n_conditions * n_lags].reshape(n_conditions, n_lags)
        M[r] = fir.sum(axis=1) * tr
    return M


def response_error_budget(
    timeseries,
    design,
    tr,
    horizon,
    nuisance=None,
    lesion_mask=None,
    n_bootstrap=2000,
    blocks=None,
):
    """Bootstrap response matrices to estimate operator error.

    Run/block bootstrap repeats FIR fitting; returns a simultaneous
    spectral-norm error budget epsilon_Z per queried output set.

    Parameters
    ----------
    timeseries, design, tr, horizon, nuisance, lesion_mask
        As in estimate_fir_area.
    n_bootstrap : int
    blocks : (n_timepoints,) array, optional
        Block labels for dependent bootstrap; defaults to runs.

    Returns
    -------
    M_hat : (n_regions, n_conditions) array
        Bootstrap-mean integrated response.
    epsilon : (n_regions,) array
        Region-level 1 / ||m_r|| multiplier used in row-normalization error.
    M_boot : (n_bootstrap, n_regions, n_conditions) array
        Resampled response matrices.
    """
    n_time = timeseries.shape[0]
    if blocks is None:
        blocks = np.zeros(n_time, dtype=int)
    unique_blocks = np.unique(blocks)
    n_blocks = unique_blocks.size

    M_boot = []
    for _ in range(n_bootstrap):
        sample_blocks = np.random.choice(unique_blocks, size=n_blocks, replace=True)
        idx = np.concatenate([np.where(blocks == b)[0] for b in sample_blocks])
        M_b = estimate_fir_area(
            timeseries[idx],
            design[:, idx],
            tr,
            horizon,
            nuisance[idx] if nuisance is not None else None,
            lesion_mask,
        )
        M_boot.append(M_b)
    M_boot = np.stack(M_boot)
    M_hat = M_boot.mean(axis=0)

    # Region-level operator error propagated through row normalization.
    # Epsilon here is the max over bootstraps of ||hat_m_r - m_r||_2 / ||m_r||_2.
    row_norms = np.linalg.norm(M_hat, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1.0
    diffs = M_boot - M_hat[None, ...]
    relative = np.linalg.norm(diffs, axis=2) / row_norms.squeeze()
    epsilon = np.quantile(relative, 0.95, axis=0)
    epsilon[row_norms.squeeze() == 0] = np.inf
    return M_hat, epsilon, M_boot
