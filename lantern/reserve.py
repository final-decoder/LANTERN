"""Gain-invariant reserve construction and interval propagation."""

import itertools
import numpy as np
from scipy import linalg


def row_projector(M):
    """Return P_r(M) = m_r m_r^T / ||m_r||^2 for each row.

    Zero rows produce zero matrices, preserving the lesion-mask convention.
    """
    n = M.shape[0]
    P = np.zeros((n, M.shape[1], M.shape[1]))
    norms = np.linalg.norm(M, axis=1)
    for r in range(n):
        if norms[r] > 0:
            m = M[r] / norms[r]
            P[r] = np.outer(m, m)
    return P


def normalize_rows(M):
    """Normalize nonzero rows of M to unit length; zero rows remain zero."""
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return M / norms


def reserve_log_volume(M, subset, tau):
    """Compute capa_tau(Z) = log det(I + tau^{-1} K_Z).

    Parameters
    ----------
    M : (n_regions, n_conditions) array
    subset : sequence of int
        Region indices Z \\subseteq X.
    tau : float
        Resolution parameter.

    Returns
    -------
    float
        Log-volume reserve for output set Z.
    """
    Mz = M[list(subset)]
    if Mz.size == 0:
        return 0.0
    Kz = row_projector(Mz).sum(axis=0)
    s2 = linalg.eigvalsh(Kz)
    s2 = s2[s2 > 0]
    return float(np.sum(np.log1p(s2 / tau)))


def reserve_intervals(M_hat, epsilon, subset, tau):
    """Initial lower/upper bounds from propagated operator error.

    epsilon_Z bounds the spectral distance between normalized matrices,
    and the bounds follow from Weyl perturbation of singular values.
    """
    Mz = normalize_rows(M_hat[list(subset)])
    s = linalg.svdvals(Mz)
    q = min(Mz.shape)
    s = np.pad(s, (0, q - s.size), constant_values=0.0)
    eps = epsilon[list(subset)].max() if len(subset) else 0.0
    lower = np.sum(np.log1p(np.maximum(s - eps, 0.0) ** 2 / tau))
    upper = np.sum(np.log1p((s + eps) ** 2 / tau))
    return float(lower), float(upper)


def c_max(len_z, k, tau):
    """Normalizing constant q * log(1 + |Z| / (q * tau)), q = min(|Z|, k)."""
    q = min(len_z, k)
    if q == 0:
        return 1.0
    return q * np.log1p(len_z / (q * tau))


def normalize_reserve(lower, upper, len_z, k, tau):
    """Divide reserve bounds by the known maximum for decoder input."""
    denom = c_max(len_z, k, tau)
    return lower / denom, upper / denom


def tighten_intervals(intervals, group_sets, k, max_iter=100):
    """Apply submodularity and monotonicity to tighten reserve intervals.

    intervals : dict[str -> (lower, upper)]
    group_sets : dict[str -> set]

    Returns
    -------
    tightened : dict[str -> (lower, upper)]
    """
    keys = list(intervals.keys())
    lo = {k: intervals[k][0] for k in keys}
    hi = {k: intervals[k][1] for k in keys}

    for _ in range(max_iter):
        changed = False
        for a, b in itertools.combinations(keys, 2):
            A, B = group_sets[a], group_sets[b]
            un = frozenset(A | B)
            inter = frozenset(A & B)
            if un not in group_sets or inter not in group_sets:
                continue
            un_key = [k for k, v in group_sets.items() if v == un][0]
            inter_key = [k for k, v in group_sets.items() if v == inter][0]

            # Submodularity: capa(A U B) + capa(A I B) <= capa(A) + capa(B)
            new_hi_un = min(hi[un_key], hi[a] + hi[b] - lo[inter_key])
            new_lo_inter = max(lo[inter_key], lo[a] + lo[b] - hi[un_key])
            if new_hi_un < hi[un_key]:
                hi[un_key] = new_hi_un
                changed = True
            if new_lo_inter > lo[inter_key]:
                lo[inter_key] = new_lo_inter
                changed = True

            # Monotonicity: A \subseteq B implies capa(A) <= capa(B)
            if A <= B:
                if hi[a] > hi[b]:
                    hi[a] = hi[b]
                    changed = True
                if lo[b] < lo[a]:
                    lo[b] = lo[a]
                    changed = True
            elif B <= A:
                if hi[b] > hi[a]:
                    hi[b] = hi[a]
                    changed = True
                if lo[a] < lo[b]:
                    lo[a] = lo[b]
                    changed = True
        if not changed:
            break
    return {k: (lo[k], hi[k]) for k in keys}


def build_reserve_features(M_hat, epsilon, groups, tau):
    """Construct normalized reserve intervals for all nonempty group unions.

    Parameters
    ----------
    M_hat : (n_regions, n_conditions) array
    epsilon : (n_regions,) array
    groups : list[list[int]]
        Six anatomical groups as lists of parcel indices.
    tau : float

    Returns
    -------
    features : dict[str -> (lo, hi)]
    group_sets : dict[str -> set]
    """
    k = M_hat.shape[1]
    group_sets = {}
    for r in range(1, len(groups) + 1):
        for combo in itertools.combinations(range(len(groups)), r):
            union = set().union(*[set(groups[i]) for i in combo])
            name = "_".join(str(i + 1) for i in combo)
            group_sets[name] = frozenset(union)

    intervals = {}
    for name, subset in group_sets.items():
        lo, hi = reserve_intervals(M_hat, epsilon, sorted(subset), tau)
        lo_n, hi_n = normalize_reserve(lo, hi, len(subset), k, tau)
        intervals[name] = (lo_n, hi_n)

    tightened = tighten_intervals(intervals, group_sets, k)
    return tightened, group_sets


def tau_sensitivity(M_hat, epsilon, groups, taus):
    """Scan the resolution parameter and collect feature curves.

    For each tau in ``taus`` the full normalized interval set is rebuilt;
    the returned arrays track, per output-set key, how the midpoint and
    half-width of every interval vary with tau.

    Parameters
    ----------
    M_hat : (n_regions, n_conditions) array
    epsilon : (n_regions,) array
    groups : list[list[int]]
    taus : sequence of float

    Returns
    -------
    mid : dict[str -> (len(taus),) array]
    half_width : dict[str -> (len(taus),) array]
    """
    keys, mid, half_width = None, {}, {}
    for tau in taus:
        intervals, group_sets = build_reserve_features(M_hat, epsilon, groups, tau)
        if keys is None:
            keys = list(intervals.keys())
        for key in keys:
            lo, hi = intervals[key]
            mid.setdefault(key, []).append(0.5 * (lo + hi))
            half_width.setdefault(key, []).append(0.5 * (hi - lo))
    mid = {k: np.asarray(v) for k, v in mid.items()}
    half_width = {k: np.asarray(v) for k, v in half_width.items()}
    return mid, half_width


def leave_one_group_out_volumes(M, groups, tau):
    """Reserve volume of every leave-one-group-out subset.

    Complements the all-union features with a diagnostic view: for each
    anatomical group g, the log-volume of the response restricted to the
    remaining parcels, i.e. how much diversity survives without g.

    Parameters
    ----------
    M : (n_regions, n_conditions) array
    groups : list[list[int]]
    tau : float

    Returns
    -------
    volumes : dict[str -> float]
        Keyed by the held-out group index (1-based), as in group names.
    """
    all_regions = set().union(*[set(g) for g in groups])
    volumes = {}
    for i, group in enumerate(groups):
        complement = sorted(all_regions - set(group))
        volumes[str(i + 1)] = reserve_log_volume(M, complement, tau)
    return volumes
