"""Decision rules for experiments that get looked at more than once.

The problem
-----------
A fixed-horizon test controls type-I error at one pre-specified moment. Watching
a dashboard and stopping the first time p < 0.05 is a different procedure with a
different error rate, and the gap is not small. Everything here is scored on
`simulate.py`, so the numbers in the README are measured rather than quoted.

Three rules are implemented:

- `naive_peeking`  -- what people actually do. Included to be measured, not used.
- `pocock_boundary` -- one constant z-threshold applied at every look, with the
  threshold *calibrated by simulation* rather than looked up. Calibration and
  validation use disjoint seeds, for the same reason you would not report
  training accuracy.
- `msprt` -- always-valid p-values (Johari et al.). Valid at every n
  simultaneously, including sample sizes you did not plan for, which is what
  makes continuous monitoring legitimate rather than merely corrected.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from .simulate import LookData


def _stop_info(crossed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Given a boolean (reps, looks) matrix of 'boundary crossed at this look',
    return (declared_effect, index_of_stopping_look).

    Replications that never cross are recorded as stopping at the final look,
    which is what actually happens: the experiment runs to its horizon.
    """
    declared = crossed.any(axis=1)
    first = np.argmax(crossed, axis=1)
    first[~declared] = crossed.shape[1] - 1
    return declared, first


def fixed_horizon(look: LookData, alpha: float = 0.05):
    """Test once, at the end. The only rule that needs no correction."""
    crit = stats.norm.ppf(1 - alpha / 2)
    declared = np.abs(look.z[:, -1]) > crit
    stop = np.full(len(declared), look.z.shape[1] - 1)
    return declared, stop


def naive_peeking(look: LookData, alpha: float = 0.05):
    """Stop the first time the uncorrected p-value drops below alpha."""
    crit = stats.norm.ppf(1 - alpha / 2)
    return _stop_info(np.abs(look.z) > crit)


def pocock_boundary(look: LookData, crit: float):
    """Constant z-boundary applied at every look."""
    return _stop_info(np.abs(look.z) > crit)


def calibrate_pocock(
    n_looks: int, alpha: float = 0.05, seed: int = 12345, n_reps: int = 20_000, **kw
) -> float:
    """Find the constant boundary that gives `alpha` family-wise error.

    Solved by simulating under the null and taking the (1-alpha) quantile of each
    replication's maximum |z|. This is exact for the design being simulated,
    whereas a textbook Pocock constant assumes equally-spaced looks and known
    variance.
    """
    from .simulate import simulate_looks

    null = simulate_looks(
        n_reps=n_reps, horizon_days=n_looks, effect=0.0, seed=seed, **kw
    )
    return float(np.quantile(np.abs(null.z).max(axis=1), 1 - alpha))


def msprt(look: LookData, alpha: float = 0.05, tau: float | None = None):
    """Mixture SPRT always-valid p-values for a two-sample mean difference.

    Under H0 the likelihood ratio against a N(0, tau^2) mixture alternative is a
    non-negative martingale, so by Ville's inequality P(sup_n LR >= 1/alpha) <=
    alpha. The running-minimum p-value below is therefore valid at *every* n at
    once -- no horizon needs to be fixed in advance.

    `tau` sets the effect size the test is tuned for. Defaulting it to the
    per-observation sigma means the procedure is most powerful against effects
    of about one standard deviation; smaller tau buys power against smaller
    effects at the cost of detecting large ones more slowly.
    """
    tau = look.sigma if tau is None else tau
    n = look.n_per_arm[None, :]
    # Variance of the difference in means with n per arm.
    v = 2 * look.sigma**2 / n
    lr = np.sqrt(v / (v + tau**2)) * np.exp(
        look.diff**2 * tau**2 / (2 * v * (v + tau**2))
    )
    p = np.minimum.accumulate(np.minimum(1.0, 1.0 / lr), axis=1)
    return _stop_info(p < alpha)
