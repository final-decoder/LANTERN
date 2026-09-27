"""Split conformal calibration for outcome prediction intervals."""

import numpy as np
from sklearn.model_selection import KFold


def conformal_interval(decoder, z, lo, hi, y_calib, alpha=0.10, clip=(0.0, 100.0)):
    """Calibrate residuals and return interval function for new examples.

    Split-conformal scores are the excess of the true outcome beyond the
    decoder's reserve envelope; the quantile expands the envelope
    symmetrically. The final interval is clipped to the WAB-AQ
    range [0, 100].

    Parameters
    ----------
    decoder : MonotoneSplineDecoder
        Fitted decoder (fixed during calibration).
    z : (n_cal, n_context) array
    lo, hi : (n_cal, n_reserve) arrays
        Normalized reserve envelopes for calibration participants.
    y_calib : (n_cal,) array
        Observed outcomes.
    alpha : float
        Target miscoverage.
    clip : tuple
        Outcome bounds.

    Returns
    -------
    predict_interval : callable
        Function taking (z_test, lo_test, hi_test) and returning
        (lower, upper) arrays.
    q : float
        Calibrated residual quantile.
    """
    pred_lo = decoder.predict(z, lo)
    pred_hi = decoder.predict(z, hi)
    scores = np.maximum(
        np.maximum(pred_lo - y_calib, y_calib - pred_hi),
        0.0,
    )
    n = scores.size
    k = int(np.ceil((n + 1) * (1 - alpha)))
    q = np.sort(scores)[min(k - 1, n - 1)]

    def predict_interval(z_test, lo_test, hi_test):
        lo_pred = decoder.predict(z_test, lo_test)
        hi_pred = decoder.predict(z_test, hi_test)
        lower = np.clip(lo_pred - q, clip[0], clip[1])
        upper = np.clip(hi_pred + q, clip[0], clip[1])
        return lower, upper

    return predict_interval, q


def cv_plus_interval(make_decoder, z, lo, hi, y, alpha=0.10, clip=(0.0, 100.0),
                     n_folds=5, seed=0, **fit_kwargs):
    """CV+ conformal intervals without a held-out calibration set.

    Each participant's residual is scored against a decoder fitted without
    their fold; out-of-fold envelope predictions and scores then define
    symmetric expansions for new examples around the full-data decoder.

    Parameters
    ----------
    make_decoder : callable
        Factory returning a fresh, unfitted MonotoneSplineDecoder.
    z, lo, hi, y : arrays as in conformal_interval
    alpha : float
    clip : tuple
    n_folds, seed : int
    fit_kwargs : dict
        Forwarded to the decoder's fit method.

    Returns
    -------
    predict_interval : callable
        Function taking (z_test, lo_test, hi_test) and returning
        (lower, upper) arrays.
    q : float
        Calibrated score quantile.
    """
    z = np.asarray(z, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    y = np.asarray(y, dtype=float)
    n = y.size

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof_lo = np.zeros(n)
    oof_hi = np.zeros(n)
    for tr, te in kf.split(z):
        dec = make_decoder()
        dec.fit(z[tr], lo[tr], hi[tr], y[tr], **fit_kwargs)
        oof_lo[te] = dec.predict(z[te], lo[te])
        oof_hi[te] = dec.predict(z[te], hi[te])

    scores = np.maximum(
        np.maximum(oof_lo - y, y - oof_hi),
        0.0,
    )
    k = int(np.ceil((n + 1) * (1 - alpha)))
    q = np.sort(scores)[min(k - 1, n - 1)]

    decoder = make_decoder()
    decoder.fit(z, lo, hi, y, **fit_kwargs)

    def predict_interval(z_test, lo_test, hi_test):
        lo_pred = decoder.predict(z_test, lo_test)
        hi_pred = decoder.predict(z_test, hi_test)
        lower = np.clip(lo_pred - q, clip[0], clip[1])
        upper = np.clip(hi_pred + q, clip[0], clip[1])
        return lower, upper

    return predict_interval, q
