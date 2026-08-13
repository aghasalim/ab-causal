"""CUPED: variance reduction using a pre-experiment covariate.

Idea: replace the outcome Y with Y - theta*(X - mean(X)), where X is measured
*before* randomisation. Because X is independent of assignment, subtracting it
removes variance without moving the expected difference between arms. Optimal
theta is Cov(Y,X)/Var(X), which reduces variance by a factor of (1 - rho^2).

The part worth being careful about
----------------------------------
The guarantee holds only because X is pre-treatment. Use a covariate measured
*during* the experiment -- one that treatment itself moved -- and the adjustment
subtracts part of the effect being measured. It still looks like it is working:
variance drops, confidence intervals tighten, the dashboard gets greener. The
estimate is simply wrong. `demo_post_treatment_bias` measures that, because it
is the same failure as leakage in a predictive model and deserves a number
rather than a warning.
"""
from __future__ import annotations

import numpy as np


def optimal_theta(y: np.ndarray, x: np.ndarray) -> float:
    """Variance-minimising coefficient, estimated on pooled data.

    Pooling across arms is safe for a genuine pre-experiment covariate: X cannot
    depend on assignment, so pooling does not import treatment information.
    """
    vx = np.var(x, ddof=1)
    return 0.0 if vx == 0 else float(np.cov(y, x, ddof=1)[0, 1] / vx)


def adjust(y: np.ndarray, x: np.ndarray, theta: float | None = None):
    theta = optimal_theta(y, x) if theta is None else theta
    return y - theta * (x - x.mean()), theta


def ate(y_t, y_c, x_t=None, x_c=None) -> dict:
    """Difference in means, optionally CUPED-adjusted.

    theta is fitted once on the pooled data and applied to both arms; fitting it
    separately per arm would let it absorb the treatment effect.
    """
    if x_t is not None:
        theta = optimal_theta(np.concatenate([y_t, y_c]), np.concatenate([x_t, x_c]))
        xbar = np.concatenate([x_t, x_c]).mean()
        y_t = y_t - theta * (x_t - xbar)
        y_c = y_c - theta * (x_c - xbar)
    est = y_t.mean() - y_c.mean()
    se = np.sqrt(y_t.var(ddof=1) / len(y_t) + y_c.var(ddof=1) / len(y_c))
    return {"ate": float(est), "se": float(se)}
