import numpy as np

from lantern.fir import (
    detect_outlier_frames,
    estimate_fir_area,
    make_drift_basis,
    make_fir_design,
    response_error_budget,
)

rng = np.random.default_rng(0)
TR, HORIZON = 2.0, 40.0
N_COND, N_REG, N_TIME = 4, 8, 240


def _synthetic():
    design = np.zeros((N_COND, N_TIME))
    for j in range(N_COND):
        design[j, j * 30 : j * 30 + 20] = 1.0
    hrf = np.exp(-np.arange(20) / 6.0)
    ts = np.zeros((N_TIME, N_REG))
    for r in range(N_REG):
        for j in range(N_COND):
            ts[:, r] += np.convolve(design[j], hrf, mode="same")
        ts[:, r] *= np.exp(rng.normal())
        ts[:, r] += 0.3 * rng.normal(size=N_TIME)
    return ts, design


def test_drift_basis_orthonormal():
    D = make_drift_basis(200, 2.0, 0.01)
    assert D.shape[0] == 200
    gram = D.T @ D
    np.testing.assert_allclose(gram, np.eye(D.shape[1]), atol=1e-8)


def test_outlier_detection_flags_spike():
    ts, _ = _synthetic()
    ts_clean = ts.copy()
    ts[50] += 50.0
    keep = detect_outlier_frames(ts)
    assert not keep[50]
    keep_clean = detect_outlier_frames(ts_clean)
    assert keep_clean.sum() > 0.9 * N_TIME


def test_area_recovery_without_noise():
    design = np.zeros((N_COND, N_TIME))
    for j in range(N_COND):
        design[j, j * 30 : j * 30 + 20] = 1.0
    M_true = rng.normal(size=(N_REG, N_COND))
    X = make_fir_design(np.zeros(N_TIME), design, TR, HORIZON)
    n_lags = int(np.ceil(HORIZON / TR))
    M_hat = np.zeros((N_REG, N_COND))
    for r in range(N_REG):
        beta = np.repeat(M_true[r] / (n_lags * TR), n_lags)
        ts_r = X @ beta
        M_hat[r] = estimate_fir_area(
            ts_r[:, None], design, TR, HORIZON, scrub=False
        )[0]
    np.testing.assert_allclose(M_hat, M_true, rtol=1e-6, atol=1e-6)


def test_lesion_mask_zeros_rows():
    ts, design = _synthetic()
    mask = np.zeros(N_REG, dtype=bool)
    mask[3] = True
    M = estimate_fir_area(ts, design, TR, HORIZON, lesion_mask=mask)
    assert np.all(M[3] == 0.0)


def test_error_budget_shapes_and_magnitude():
    ts, design = _synthetic()
    blocks = np.floor(np.arange(N_TIME) / 30).astype(int)
    M_hat, eps, M_boot = response_error_budget(
        ts, design, TR, HORIZON, blocks=blocks, n_bootstrap=50, n_jobs=2
    )
    assert M_boot.shape == (50, N_REG, N_COND)
    assert eps.shape == (N_REG,)
    assert np.all(eps >= 0.0)
