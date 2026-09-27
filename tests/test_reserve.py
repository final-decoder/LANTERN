import numpy as np
import pytest

from lantern.reserve import (
    build_reserve_features,
    leave_one_group_out_volumes,
    reserve_intervals,
    reserve_log_volume,
    row_projector,
    tau_sensitivity,
)

rng = np.random.default_rng(42)
M = rng.normal(size=(18, 6))
GROUPS = [[i * 3 + j for j in range(3)] for i in range(6)]
EPS = np.full(18, 0.05)
TAU = 0.25


def test_row_projector_gain_invariance():
    gains = np.exp(rng.normal(size=18))
    M_scaled = M * gains[:, None]
    np.testing.assert_allclose(
        row_projector(M_scaled), row_projector(M), atol=1e-10
    )


def test_row_projector_zero_row():
    Mz = M.copy()
    Mz[0] = 0.0
    assert np.all(row_projector(Mz)[0] == 0.0)


def test_log_volume_monotone_in_subset():
    base = reserve_log_volume(M, range(12), TAU)
    extended = reserve_log_volume(M, range(18), TAU)
    assert extended >= base - 1e-10


def test_interval_contains_point_estimate():
    lo, hi = reserve_intervals(M, EPS, sorted(GROUPS[0]), TAU)
    point = reserve_log_volume(M, GROUPS[0], TAU)
    assert lo - 1e-10 <= point <= hi + 1e-10


def test_build_features_keys_and_ordering():
    intervals, group_sets = build_reserve_features(M, EPS, GROUPS, TAU)
    assert len(intervals) == 2 ** 6 - 1
    for lo, hi in intervals.values():
        assert 0.0 <= lo <= hi


def test_tau_sensitivity_shapes():
    taus = [0.1, 0.25, 1.0]
    mid, half_width = tau_sensitivity(M, EPS, GROUPS, taus)
    assert set(mid) == set(half_width)
    for v in mid.values():
        assert v.shape == (len(taus),)


def test_leave_one_group_out():
    volumes = leave_one_group_out_volumes(M, GROUPS, TAU)
    assert set(volumes) == {str(i + 1) for i in range(6)}
    full = reserve_log_volume(M, sorted(set().union(*GROUPS)), TAU)
    assert all(v <= full + 1e-10 for v in volumes.values())


def test_tightening_shrinks_or_preserves():
    intervals, group_sets = build_reserve_features(M, EPS, GROUPS, TAU)
    from lantern.reserve import tighten_intervals
    from itertools import combinations

    lo = {k: v[0] for k, v in intervals.items()}
    hi = {k: v[1] for k, v in intervals.items()}
    tightened = tighten_intervals(intervals, group_sets, M.shape[1])
    for k in intervals:
        assert tightened[k][0] >= lo[k] - 1e-10
        assert tightened[k][1] <= hi[k] + 1e-10
