"""Pre-flight and post-flight checks for a live experiment.

These are the cheap tests that catch the expensive mistakes: an experiment whose
traffic split is broken, or one that never had the statistical power to answer
its own question.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def srm(counts: dict[str, int], expected: dict[str, float] | None = None,
        alpha: float = 0.0005) -> dict:
    """Sample ratio mismatch: did users actually arrive in the intended split?

    A significant result here invalidates the experiment rather than informing
    it -- if assignment is broken, the arms are no longer exchangeable and no
    amount of downstream analysis fixes that.

    The threshold is 0.0005, not 0.05, deliberately: this check runs on every
    experiment, so at 0.05 a large programme would drown in false alarms and
    start ignoring the alarm. The trade-off is intentional and worth stating
    rather than inheriting silently.
    """
    keys = list(counts)
    obs = np.array([counts[k] for k in keys], dtype=float)
    if expected is None:
        exp = np.full(len(keys), obs.sum() / len(keys))
    else:
        w = np.array([expected[k] for k in keys], dtype=float)
        exp = obs.sum() * w / w.sum()
    chi2 = float(((obs - exp) ** 2 / exp).sum())
    p = float(stats.chi2.sf(chi2, df=len(keys) - 1))
    return {
        "chi2": chi2, "p_value": p, "srm_detected": p < alpha,
        "observed": dict(zip(keys, obs.astype(int))),
        "expected": dict(zip(keys, exp.round(1))),
    }


def mde(n_per_arm: int, sigma: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest true effect detectable with `power`, given the sample you have.

    Worth computing *before* running: an experiment that cannot detect an effect
    it would care about is not an inconclusive result, it is a wasted one.
    """
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    return float((z_a + z_b) * np.sqrt(2 * sigma**2 / n_per_arm))


def required_n(effect: float, sigma: float, alpha: float = 0.05,
               power: float = 0.8) -> int:
    """Per-arm sample size needed to detect `effect`."""
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    return int(np.ceil(2 * sigma**2 * (z_a + z_b) ** 2 / effect**2))


def mde_proportion(n_per_arm: int, baseline: float, alpha: float = 0.05,
                   power: float = 0.8) -> float:
    """MDE for a conversion-rate metric, in absolute percentage points."""
    return mde(n_per_arm, np.sqrt(baseline * (1 - baseline)), alpha, power)
