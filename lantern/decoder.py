"""Monotone spline decoder trained against worst-case endpoint error."""

import numpy as np
from scipy.optimize import minimize


class MonotoneSplineDecoder:
    """g_theta(z, c) = beta_0 + beta^T z + sum_{q,b} w_{qb} (c_q - t_b)_+.

    Nonnegative weights w_{qb} enforce monotonicity in every reserve
    coordinate at fixed clinical context.
    """

    def __init__(self, n_context, n_reserve, knots, l2_lambda=1.0):
        self.n_context = n_context
        self.n_reserve = n_reserve
        self.knots = np.asarray(knots)
        self.l2_lambda = l2_lambda
        self.theta_ = None

    def _unpack(self, theta):
        beta0 = theta[0]
        beta = theta[1 : 1 + self.n_context]
        w = theta[1 + self.n_context :].reshape(
            self.n_reserve, self.n_knots
        )
        return beta0, beta, w

    @property
    def n_knots(self):
        return self.knots.size

    @property
    def n_params(self):
        return 1 + self.n_context + self.n_reserve * self.n_knots

    def _relu_basis(self, c):
        """Compute sum_b w_{qb} (c_q - t_b)_+ for each coordinate q."""
        c = np.atleast_2d(c)
        diff = c[:, :, None] - self.knots[None, None, :]
        return np.maximum(diff, 0.0)  # (n, n_reserve, n_knots)

    def predict(self, z, c):
        """Evaluate decoder at clinical context z and reserve vector c."""
        beta0, beta, w = self._unpack(self.theta_)
        z = np.atleast_2d(z)
        c = np.atleast_2d(c)
        linear = beta0 + z @ beta
        relu = self._relu_basis(c) * w[None, :, :]
        return linear + relu.sum(axis=(1, 2))

    def fit(self, z, lo, hi, y, max_iter=10000, lr=0.01):
        """Minimize average worst-case squared error over reserve intervals.

        Maximize over the interval endpoints by taking the larger of the
        two squared residuals, plus an L2 penalty.
        """
        z = np.asarray(z, dtype=float)
        lo = np.asarray(lo, dtype=float)
        hi = np.asarray(hi, dtype=float)
        y = np.asarray(y, dtype=float)

        def loss(theta):
            self.theta_ = theta
            pred_lo = self.predict(z, lo)
            pred_hi = self.predict(z, hi)
            err_lo = (y - pred_lo) ** 2
            err_hi = (y - pred_hi) ** 2
            worst = np.maximum(err_lo, err_hi)
            pen = self.l2_lambda * np.sum(theta[1:] ** 2)
            return worst.mean() + pen

        theta0 = np.zeros(self.n_params)
        # Enforce nonnegative spline weights via bounds.
        bounds = [(None, None)] * (1 + self.n_context)
        bounds += [(0.0, None)] * (self.n_reserve * self.n_knots)

        result = minimize(
            loss,
            theta0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max_iter},
        )
        self.theta_ = result.x
        return self

    def fit_cv(self, z, lo, hi, y, l2_grid=None, n_folds=5, seed=0,
               max_iter=10000, lr=0.01):
        """Select l2_lambda by K-fold CV, then refit on the full data.

        The CV score is the worst-case endpoint loss on held-out folds.
        Stratification is by outcome deciles so folds stay outcome-balanced.

        Parameters
        ----------
        z, lo, hi, y : as in fit
        l2_grid : sequence of float, optional
            Defaults to a log-spaced grid from 1e-3 to 1e2.
        n_folds : int
        seed : int
            Seed for fold assignment.
        max_iter, lr : as in fit

        Returns
        -------
        self
            With ``self.l2_lambda`` set to the CV-selected value and
            ``self.cv_scores_`` holding the per-grid-point mean losses.
        """
        if l2_grid is None:
            l2_grid = np.logspace(-3, 2, 6)
        z, lo, hi, y = map(lambda a: np.asarray(a, dtype=float), (z, lo, hi, y))

        deciles = np.quantile(y, np.linspace(0, 1, 11))
        fold_id = np.digitize(y, deciles[1:-1])
        rng = np.random.default_rng(seed)
        jitter = rng.permutation(len(y)) % n_folds
        fold_id = (fold_id + jitter) % n_folds

        self.cv_scores_ = {}
        for lam in l2_grid:
            fold_losses = []
            for f in range(n_folds):
                tr, te = fold_id != f, fold_id == f
                candidate = MonotoneSplineDecoder(
                    self.n_context, self.n_reserve, self.knots, l2_lambda=lam
                )
                candidate.fit(z[tr], lo[tr], hi[tr], y[tr],
                              max_iter=max_iter, lr=lr)
                fold_losses.append(
                    worst_case_loss(candidate, z[te], lo[te], hi[te], y[te])
                )
            self.cv_scores_[lam] = float(np.mean(fold_losses))

        best = min(self.cv_scores_, key=self.cv_scores_.get)
        self.l2_lambda = best
        return self.fit(z, lo, hi, y, max_iter=max_iter, lr=lr)


def worst_case_loss(decoder, z, lo, hi, y):
    """Compute the convex worst-case endpoint objective on a batch."""
    pred_lo = decoder.predict(z, lo)
    pred_hi = decoder.predict(z, hi)
    return np.maximum((y - pred_lo) ** 2, (y - pred_hi) ** 2).mean()
