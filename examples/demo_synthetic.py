"""Run the LANTERN pipeline end-to-end on a synthetic response matrix."""

import numpy as np
from lantern.fir import response_error_budget
from lantern.reserve import (
    reserve_log_volume,
    reserve_intervals,
    build_reserve_features,
)
from lantern.decoder import MonotoneSplineDecoder
from lantern.calibration import conformal_interval

np.random.seed(0)

# Synthetic data: 18 regions, 6 conditions, 200 timepoints.
n_regions, n_conditions, n_time = 18, 6, 200
tr, horizon = 2.0, 40.0
n_groups = 6
groups = [[i * 3 + j for j in range(3)] for i in range(n_groups)]

# Two-condition design with on/off blocks.
design = np.zeros((n_conditions, n_time))
for j in range(n_conditions):
    design[j, j * 20 : j * 20 + 15] = 1.0

# Regional responses with heterogeneous gains to test invariance.
M_true = np.random.randn(n_regions, n_conditions)
gains = np.exp(np.random.randn(n_regions))
timeseries = np.zeros((n_time, n_regions))
for r in range(n_regions):
    for j in range(n_conditions):
        timeseries[:, r] += np.convolve(
            design[j], np.random.rand(int(horizon / tr)), mode="same"
        )
    timeseries[:, r] *= gains[r]
    timeseries[:, r] += 0.5 * np.random.randn(n_time)

# Stage 1: estimate integrated responses with a block-bootstrap error budget.
blocks = np.floor(np.arange(n_time) / 20).astype(int)
M_hat, epsilon, _ = response_error_budget(
    timeseries, design, tr, horizon, blocks=blocks, n_bootstrap=200
)

# Stage 2: reserve and interval construction.
tau = 0.25
intervals, _ = build_reserve_features(M_hat, epsilon, groups, tau)
print("Number of output-set features:", len(intervals))
print("Example interval:", list(intervals.items())[0])

# Stages 3-4: fit decoder and calibrate on held-out synthetic cohorts.
n_train, n_cal = 50, 20
z = np.random.randn(n_train + n_cal, 3)
y = np.random.rand(n_train + n_cal) * 100

lo = np.random.rand(n_train + n_cal, len(intervals))
hi = lo + 0.1 * np.random.rand(n_train + n_cal, len(intervals))

decoder = MonotoneSplineDecoder(
    n_context=3, n_reserve=len(intervals), knots=[0, 0.25, 0.5, 0.75]
)
decoder.fit(z[:n_train], lo[:n_train], hi[:n_train], y[:n_train])

interval_fn, q = conformal_interval(
    decoder, z[n_train:], lo[n_train:], hi[n_train:], y[n_train:]
)
lo_test, hi_test = lo[:5], hi[:5]
lower, upper = interval_fn(z[:5], lo_test, hi_test)
print("Calibrated interval width:", (upper - lower).mean())
