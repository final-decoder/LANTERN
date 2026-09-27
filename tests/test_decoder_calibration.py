import numpy as np

from lantern.calibration import conformal_interval, cv_plus_interval
from lantern.decoder import MonotoneSplineDecoder, worst_case_loss

rng = np.random.default_rng(7)
N_TRAIN, N_CAL, N_TEST = 120, 60, 40
N_RES, N_CTX = 5, 3


def _data(n):
    z = rng.normal(size=(n, N_CTX))
    lo = rng.uniform(0.0, 0.6, size=(n, N_RES))
    hi = lo + rng.uniform(0.05, 0.3, size=(n, N_RES))
    y = 50.0 + 10.0 * z[:, 0] + 5.0 * lo[:, 0] + rng.normal(scale=2.0, size=n)
    return z, lo, hi, y


def _decoder():
    return MonotoneSplineDecoder(
        n_context=N_CTX, n_reserve=N_RES, knots=[0.0, 0.25, 0.5, 0.75],
        l2_lambda=0.1,
    )


def test_decoder_monotone_in_reserve():
    z, lo, hi, y = _data(N_TRAIN)
    dec = _decoder().fit(z, lo, hi, y, max_iter=2000)
    c = rng.uniform(0.0, 0.8, size=(1, N_RES))
    c_up = c.copy()
    c_up[0, 2] += 0.2
    p0 = dec.predict(z[:1], c)
    p1 = dec.predict(z[:1], c_up)
    assert p1[0] >= p0[0] - 1e-8


def test_fit_cv_selects_lambda():
    z, lo, hi, y = _data(N_TRAIN)
    dec = _decoder()
    dec.fit_cv(z, lo, hi, y, l2_grid=[0.01, 1.0], n_folds=3, max_iter=1000)
    assert dec.l2_lambda in (0.01, 1.0)
    assert len(dec.cv_scores_) == 2


def test_worst_case_loss_nonneg():
    z, lo, hi, y = _data(30)
    dec = _decoder().fit(z, lo, hi, y, max_iter=1000)
    assert worst_case_loss(dec, z, lo, hi, y) >= 0.0


def test_conformal_coverage():
    z, lo, hi, y = _data(N_TRAIN)
    dec = _decoder().fit(z, lo, hi, y, max_iter=2000)
    zc, loc, hic, yc = _data(N_CAL)
    interval_fn, q = conformal_interval(dec, zc, loc, hic, yc, alpha=0.10)
    zt, lot, hit, yt = _data(N_TEST)
    lower, upper = interval_fn(zt, lot, hit)
    cov = ((yt >= lower) & (yt <= upper)).mean()
    assert 0.80 <= cov <= 1.0
    assert q >= 0.0


def test_cv_plus_interval_valid():
    z, lo, hi, y = _data(N_TRAIN)
    make = lambda: _decoder()
    interval_fn, q = cv_plus_interval(
        make, z, lo, hi, y, alpha=0.10, n_folds=3, max_iter=1000
    )
    zt, lot, hit, yt = _data(N_TEST)
    lower, upper = interval_fn(zt, lot, hit)
    assert np.all(lower <= upper)
    assert np.all((lower >= 0.0) & (upper <= 100.0))
