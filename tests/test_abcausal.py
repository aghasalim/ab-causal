"""Tests assert the *claims*, not just that the code runs.

Several of these would fail if a refactor quietly made a headline result
disappear -- which is the failure mode that matters for a repo whose entire
value is its numbers.
"""
import numpy as np
import pytest

from src.abcausal import cuped, diagnostics, sequential
from src.abcausal.simulate import simulate_looks

SMALL = dict(n_reps=4000, n_per_day=100, horizon_days=10)


@pytest.fixture(scope="module")
def null():
    return simulate_looks(effect=0.0, seed=101, **SMALL)


# --- the harness must be calibrated before anything else is believable -----

def test_fixed_horizon_hits_nominal_alpha(null):
    declared, _ = sequential.fixed_horizon(null, alpha=0.05)
    assert 0.04 < declared.mean() < 0.06


# --- the headline claim ----------------------------------------------------

def test_naive_peeking_inflates_type_one_error(null):
    fixed, _ = sequential.fixed_horizon(null)
    peek, _ = sequential.naive_peeking(null)
    assert peek.mean() > 3 * fixed.mean()


def test_calibrated_boundary_restores_error_control():
    crit = sequential.calibrate_pocock(n_looks=10, alpha=0.05, seed=555, n_reps=8000)
    # Validated on a different seed than it was calibrated on.
    fresh = simulate_looks(effect=0.0, seed=777, **SMALL)
    declared, _ = sequential.pocock_boundary(fresh, crit)
    assert 0.03 < declared.mean() < 0.07
    assert crit > 1.96, "a corrected boundary must be stricter than uncorrected"


def test_msprt_is_conservative_not_merely_correct(null):
    """Anytime-validity is a stronger guarantee than fixed-look correction, and
    the price is visible: realised error well below nominal."""
    declared, _ = sequential.msprt(null, alpha=0.05, tau=0.1)
    assert declared.mean() < 0.05


# --- CUPED -----------------------------------------------------------------

def test_cuped_reduces_variance_without_bias():
    rng = np.random.default_rng(3)
    rho, effect, n = 0.8, 0.2, 3000
    plain, adj = [], []
    for _ in range(600):
        x_t, x_c = rng.normal(size=n), rng.normal(size=n)
        y_t = effect + rho * x_t + np.sqrt(1 - rho**2) * rng.normal(size=n)
        y_c = rho * x_c + np.sqrt(1 - rho**2) * rng.normal(size=n)
        plain.append(cuped.ate(y_t, y_c)["ate"])
        adj.append(cuped.ate(y_t, y_c, x_t, x_c)["ate"])
    assert np.var(adj) < 0.5 * np.var(plain)
    assert abs(np.mean(adj) - effect) < 0.01


def test_cuped_on_a_mediator_returns_the_direct_effect_not_the_total():
    """Guards the counter-example behind the README's warning.

    Treatment -> X -> Y with no direct path, so the *total* effect is entirely
    mediated. CUPED adjusts X away and should therefore report approximately
    zero while the true total effect is 0.16. Plain diff-in-means gets it right.
    """
    rng = np.random.default_rng(5)
    beta, shift, n = 0.8, 0.2, 4000
    total = beta * shift  # 0.16, all of it through X; no direct path
    plain, adj = [], []
    for _ in range(400):
        x_c = rng.normal(size=n)
        x_t = rng.normal(size=n) + shift
        y_c = beta * x_c + rng.normal(size=n) * 0.6
        y_t = beta * x_t + rng.normal(size=n) * 0.6
        plain.append(cuped.ate(y_t, y_c)["ate"])
        adj.append(cuped.ate(y_t, y_c, x_t, x_c)["ate"])
    assert np.mean(plain) == pytest.approx(total, abs=0.01)
    assert abs(np.mean(adj)) < 0.02, "CUPED should have removed the whole effect"


def test_optimal_theta_is_zero_for_unrelated_covariate():
    rng = np.random.default_rng(9)
    y, x = rng.normal(size=5000), rng.normal(size=5000)
    assert abs(cuped.optimal_theta(y, x)) < 0.05


# --- diagnostics -----------------------------------------------------------

def test_srm_passes_on_a_fair_split():
    assert not diagnostics.srm({"a": 10_000, "b": 10_050})["srm_detected"]


def test_srm_catches_a_broken_split():
    assert diagnostics.srm({"a": 10_000, "b": 9_000})["srm_detected"]


def test_srm_handles_intended_uneven_split():
    """A deliberate 90/10 split must not be flagged just for being uneven."""
    r = diagnostics.srm({"a": 90_000, "b": 10_000}, expected={"a": 0.9, "b": 0.1})
    assert not r["srm_detected"]


def test_mde_and_required_n_are_inverses():
    n = diagnostics.required_n(effect=0.1, sigma=1.0)
    assert diagnostics.mde(n_per_arm=n, sigma=1.0) == pytest.approx(0.1, rel=0.02)


def test_mde_shrinks_with_sample_size():
    assert diagnostics.mde(10_000, 1.0) < diagnostics.mde(1_000, 1.0)


# --- observational estimators on data whose ATT is known -------------------

def confounded(n=6000, seed=0, tau=1500.0):
    """LaLonde-shaped synthetic data. Younger, black, no-degree, low-earning
    people are more likely to be treated, and earnings depend on the same
    covariates, so the raw difference is badly confounded. The effect is a
    constant tau, so the ATT is tau."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    d = pd.DataFrame({
        "age": rng.uniform(18, 55, n),
        "education": rng.integers(6, 17, n).astype(float),
        "black": rng.binomial(1, 0.4, n).astype(float),
        "hispanic": rng.binomial(1, 0.1, n).astype(float),
        "married": rng.binomial(1, 0.3, n).astype(float),
        "nodegree": rng.binomial(1, 0.4, n).astype(float),
        "re74": rng.gamma(2, 4000, n),
        "re75": rng.gamma(2, 4000, n),
    })
    z = 1.5 - 0.05 * d.age + 0.8 * d.black + 0.6 * d.nodegree - 0.0002 * d.re75
    d["treat"] = rng.binomial(1, 1 / (1 + np.exp(-z))).astype(float)
    d["re78"] = (1000 + 80 * d.education + 0.5 * d.re75 + 0.2 * d.re74 - 30 * d.age
                 + tau * d.treat + rng.normal(0, 1500, n))
    return d


def test_adjusted_estimators_recover_a_known_att_that_naive_misses():
    from src.abcausal import observational as ob

    tau = 1500.0
    d = confounded(tau=tau)
    ps = ob.propensity(d, "linear")
    assert abs(ob.naive(d)["att"] - tau) > 1000, "the fixture is not confounded"
    # Both nuisance models are correctly specified here, so AIPW should be close.
    assert ob.aipw(d, ps)["att"] == pytest.approx(tau, abs=150)
    assert ob.ipw(d, ps)["att"] == pytest.approx(tau, abs=200)
    assert ob.match_nn(d, ps)["att"] == pytest.approx(tau, abs=300)
