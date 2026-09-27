"""Split conformal calibration for outcome prediction intervals."""

import numpy as np


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
