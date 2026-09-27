"""Sensitivity and diagnostics: tau scan, CV lambda, leave-one-group-out."""

import numpy as np
from lantern.fir import response_error_budget
from lantern.reserve import (
    build_reserve_features,
    leave_one_group_out_volumes,
    tau_sensitivity,
)
from lantern.decoder import MonotoneSplineDecoder

np.random.seed(1)

n_regions, n_conditions, n_time = 18, 6, 240
tr, horizon = 2.0, 40.0
groups = [[i * 3 + j for j in range(3)] for i in range(6)]

# Synthetic cohort with heterogeneous regional gains.
design = np.zeros((n_conditions, n_time))
for j in range(n_conditions):
    design[j, j * 30 : j * 30 + 20] = 1.0

n_subjects = 30
M_all = np.zeros((n_subjects, n_regions, n_conditions))
EPS_all = np.zeros((n_subjects, n_regions))
for s in range(n_subjects):
    ts = np.zeros((n_time, n_regions))
    for r in range(n_regions):
        for j in range(n_conditions):
            ts[:, r] += np.convolve(
                design[j], np.random.rand(int(horizon / tr)), mode="same"
            )
        ts[:, r] *= np.exp(np.random.randn())
        ts[:, r] += 0.5 * np.random.randn(n_time)
    blocks = np.floor(np.arange(n_time) / 30).astype(int)
    M_all[s], EPS_all[s], _ = response_error_budget(
        ts, design, tr, horizon, blocks=blocks, n_bootstrap=100, n_jobs=-1
    )

# Cohort-level error budget: median across subjects for stability.
epsilon = np.median(EPS_all, axis=0)

# tau sensitivity: interval curves are finite, with nonnegative widths;
# upper bounds may exceed 1 because the operator-error envelope is conservative.
taus = [0.1, 0.25, 0.5, 1.0, 2.0]
mid, half_width = tau_sensitivity(M_all[0], epsilon, groups, taus)
key = "1_2_3"
print("tau scan, set", key, "midpoints:", np.round(mid[key], 3))
assert np.all(np.isfinite(mid[key])) and np.all(mid[key] >= 0.0)
assert np.all(half_width[key] >= 0.0)

# leave-one-group-out diagnostics on the cohort mean response.
M_mean = M_all.mean(axis=0)
logo = leave_one_group_out_volumes(M_mean, groups, tau=0.25)
print("Leave-one-group-out volumes:", {k: round(v, 3) for k, v in logo.items()})

# CV selection of the decoder penalty on synthetic outcomes.
intervals, _ = build_reserve_features(M_mean, epsilon, groups, 0.25)
keys = list(intervals.keys())
lo = np.tile([intervals[k][0] for k in keys], (n_subjects, 1))
hi = np.tile([intervals[k][1] for k in keys], (n_subjects, 1))
z = np.random.randn(n_subjects, 3)
y = 60.0 + 8.0 * z[:, 0] - 4.0 * lo[:, 0] + np.random.randn(n_subjects) * 3

decoder = MonotoneSplineDecoder(n_context=3, n_reserve=len(keys),
                                knots=[0.0, 0.25, 0.5, 0.75])
decoder.fit_cv(z, lo, hi, y, l2_grid=[0.01, 0.1, 1.0, 10.0], n_folds=5,
               max_iter=3000)
print("CV-selected l2_lambda:", decoder.l2_lambda)
print("CV scores:", {k: round(v, 3) for k, v in decoder.cv_scores_.items()})
