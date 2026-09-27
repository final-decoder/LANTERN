"""FIR response estimation and area integration."""

import numpy as np
from joblib import Parallel, delayed
from scipy import linalg


def make_drift_basis(n_time, tr, high_pass):
    """Discrete cosine basis for low-frequency drift removal.

    Parameters
    ----------
    n_time : int
        Number of timepoints.
    tr : float
        Repetition time in seconds.
    high_pass : float
        Cutoff period in Hz; cosine terms with period below the cutoff
        (equivalently, frequency above ``high_pass``) are retained.

    Returns
    -------
    D : (n_time, n_basis) array
        Orthogonalized cosine regressors.
    """
    frame_period = tr * n_time
    n_basis = int(np.floor(2 * frame_period * high_pass))
    if n_basis < 1:
        return np.zeros((n_time, 0))
    t = np.arange(n_time)
    D = np.zeros((n_time, n_basis))
    for k in range(1, n_basis + 1):
        D[:, k - 1] = np.cos(np.pi * k * (t + 0.5) / n_time)
    # Re-orthogonalize against constant drift and numerical error accumulation.
    q, _ = linalg.qr(D, mode="economic")
    return q


def detect_outlier_frames(timeseries, std_threshold=3.0):
    """Flag high-motion frames from the time series themselves.

    Uses the robust z-score of successive frame-to-frame differences,
    aggregated over parcels. Returns a boolean mask of frames to keep.
    """
    diffs = np.abs(np.diff(timeseries, axis=0))
    fd_proxy = diffs.mean(axis=1)
    med = np.median(fd_proxy)
    mad = np.median(np.abs(fd_proxy - med)) * 1.4826
    if mad == 0:
        return np.ones(timeseries.shape[0], dtype=bool)
    z = np.abs(fd_proxy - med) / mad
    keep = np.ones(timeseries.shape[0], dtype=bool)
    keep[1:] = z <= std_threshold
    return keep


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
    high_pass=None,
    scrub=True,
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
    high_pass : float, optional
        When set, a discrete cosine drift basis with this cutoff (Hz) is
        appended to the nuisance regressors.
    scrub : bool
        When True, frames flagged by detect_outlier_frames are downweighted
        via a weights vector in the least-squares fit.

    Returns
    -------
    M : (n_regions, n_conditions) array
        Integrated response matrix; rows for unavailable regions are set to 0.
    """
    n_time, n_regions = timeseries.shape
    n_conditions = design.shape[0]
    X = make_fir_design(timeseries[:, 0], design, tr, horizon)
    if high_pass is not None:
        drift = make_drift_basis(n_time, tr, high_pass)
        X = np.column_stack([X, drift])
    if nuisance is not None:
        X = np.column_stack([X, nuisance])

    if scrub:
        keep = detect_outlier_frames(timeseries)
        w = keep.astype(float)
    else:
        w = np.ones(n_time)

    M = np.zeros((n_regions, n_conditions))
    n_lags = int(np.ceil(horizon / tr))
    for r in range(n_regions):
        if lesion_mask is not None and lesion_mask[r]:
            continue
        Xw = X * w[:, None]
        yw = timeseries[:, r] * w
        beta = linalg.lstsq(Xw, yw)[0]
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
    n_jobs=1,
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
    n_jobs : int
        Parallel workers for the bootstrap loop; -1 uses all cores.

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

    def one_resample(seed):
        rng = np.random.default_rng(seed)
        sample_blocks = rng.choice(unique_blocks, size=n_blocks, replace=True)
        idx = np.concatenate([np.where(blocks == b)[0] for b in sample_blocks])
        return estimate_fir_area(
            timeseries[idx],
            design[:, idx],
            tr,
            horizon,
            nuisance[idx] if nuisance is not None else None,
            lesion_mask,
            scrub=False,  # resampled adjacency is artificial; scrub on raw data only
        )

    seeds = np.random.SeedSequence().generate_state(n_bootstrap)
    M_boot = np.stack(
        Parallel(n_jobs=n_jobs)(
            delayed(one_resample)(int(s)) for s in seeds
        )
    )
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
