"""Simulation harness for online experiments.

Why this file exists before any analysis code
---------------------------------------------
On real data you never observe the true effect, so you cannot tell whether an
estimator is right -- only whether it is confident. That makes real data useless
for validating a method. In simulation the truth is set by construction, so a
procedure can be held to the only standard that matters: **when there is no
effect, how often does it claim one?**

Every decision rule in `sequential.py` is scored against this harness before it
is allowed near the LaLonde data.

Design note: `simulate_looks` returns the *z-statistic at each interim look* for
every replication, and all decision rules then consume that same matrix. This
makes the comparisons paired -- naive peeking and a corrected boundary see
byte-identical data -- so differences between rules are not contaminated by
simulation noise.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LookData:
    """Interim-analysis results across replications.

    z: (n_reps, n_looks) two-sample z-statistic at each look
    n_per_arm: (n_looks,) cumulative sample size per arm at each look
    effect: the true difference in means used to generate the data
    """
    z: np.ndarray
    n_per_arm: np.ndarray
    effect: float
    diff: np.ndarray  # (n_reps, n_looks) observed difference in means
    sigma: float


def simulate_looks(
    n_reps: int = 5_000,
    n_per_day: int = 100,
    horizon_days: int = 14,
    effect: float = 0.0,
    sigma: float = 1.0,
    seed: int = 0,
) -> LookData:
    """Simulate `n_reps` two-arm experiments observed daily.

    Data is generated per-user and accumulated, rather than drawing summary
    statistics per day, so the correlation between successive looks is the real
    thing. That correlation is the entire reason peeking misbehaves: consecutive
    looks share most of their data, so they are not independent tests, and
    treating them as such is what a Bonferroni correction gets wrong here.
    """
    rng = np.random.default_rng(seed)
    n_total = n_per_day * horizon_days

    control = rng.normal(0.0, sigma, size=(n_reps, n_total))
    treat = rng.normal(effect, sigma, size=(n_reps, n_total))

    idx = np.arange(n_per_day, n_total + 1, n_per_day)  # cumulative n at each look

    c_cum = np.cumsum(control, axis=1)[:, idx - 1]
    t_cum = np.cumsum(treat, axis=1)[:, idx - 1]
    c_sq = np.cumsum(control**2, axis=1)[:, idx - 1]
    t_sq = np.cumsum(treat**2, axis=1)[:, idx - 1]

    n = idx.astype(float)
    c_mean, t_mean = c_cum / n, t_cum / n
    # Unbiased sample variance from running sums.
    c_var = (c_sq - n * c_mean**2) / (n - 1)
    t_var = (t_sq - n * t_mean**2) / (n - 1)

    diff = t_mean - c_mean
    se = np.sqrt(c_var / n + t_var / n)
    return LookData(z=diff / se, n_per_arm=n, effect=effect, diff=diff, sigma=sigma)


def false_positive_rate(decisions: np.ndarray) -> float:
    """Fraction of replications that declared an effect. Under a null
    simulation this is the realised type-I error."""
    return float(decisions.mean())


def expected_sample_size(stop_look: np.ndarray, n_per_arm: np.ndarray) -> float:
    """Average per-arm sample size at the moment of stopping.

    The reason anyone peeks is to stop early, so a corrected rule that controls
    error but never stops early has not actually solved the user's problem.
    Reporting this alongside error rates keeps that trade-off visible.
    """
    return float(n_per_arm[stop_look].mean())
