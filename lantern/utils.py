"""Utilities: group definitions, bootstrap, metrics."""

import itertools
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score


def group_unions(groups):
    """Return all 2^n - 1 nonempty unions of anatomical groups."""
    unions = {}
    for r in range(1, len(groups) + 1):
        for combo in itertools.combinations(range(len(groups)), r):
            name = "_".join(str(i + 1) for i in combo)
            unions[name] = set().union(*[set(groups[i]) for i in combo])
    return unions


def bootstrap_fir(timeseries, design, tr, horizon, blocks=None, n_boot=2000):
    """Convenience wrapper; see lantern.fir.response_error_budget."""
    from .fir import response_error_budget

    return response_error_budget(
        timeseries, design, tr, horizon, blocks=blocks, n_bootstrap=n_boot
    )


def mae_r2(y_true, y_pred):
    """Return MAE and R2."""
    return mean_absolute_error(y_true, y_pred), r2_score(y_true, y_pred)


def coverage_width(lower, upper, y_true):
    """Empirical coverage and mean width of prediction intervals."""
    covered = ((y_true >= lower) & (y_true <= upper)).mean()
    width = (upper - lower).mean()
    return covered, width


def retention_curve(errors, uncertainty, fractions=None):
    """MAE as a function of retained fraction sorted by uncertainty score."""
    if fractions is None:
        fractions = np.linspace(0.1, 1.0, 10)
    order = np.argsort(uncertainty)
    sorted_err = np.asarray(errors)[order]
    out = []
    for f in fractions:
        n_keep = max(1, int(np.round(f * len(errors))))
        out.append(sorted_err[:n_keep].mean())
    return fractions, np.asarray(out)


def cosine_identification(profile_run1, profile_run2, standardize=True):
    """Cross-run patient identification by centered cosine similarity.

    Each profile is standardized by fitting-cohort statistics and then
    centered across coordinates before similarity computation.
    """
    if standardize:
        mu = profile_run1.mean(axis=0)
        sd = profile_run1.std(axis=0)
        sd[sd == 0] = 1.0
        p1 = (profile_run1 - mu) / sd
        p2 = (profile_run2 - mu) / sd
    else:
        p1, p2 = profile_run1, profile_run2
    p1 = p1 - p1.mean(axis=1, keepdims=True)
    p2 = p2 - p2.mean(axis=1, keepdims=True)
    sim = p1 @ p2.T
    n1 = np.linalg.norm(p1, axis=1, keepdims=True)
    n2 = np.linalg.norm(p2, axis=1, keepdims=True)
    sim = sim / (n1 @ n2.T)
    ids = np.argmax(sim, axis=1)
    accuracy = (ids == np.arange(len(ids))).mean()
    return accuracy, sim
