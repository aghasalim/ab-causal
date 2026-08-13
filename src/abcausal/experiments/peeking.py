"""How badly does peeking break a test, and what does fixing it cost?

Reports, for each decision rule, the three numbers that actually matter
together: type-I error under the null, power under a real effect, and the
average sample size at which the experiment stopped. Any rule can look good on
one of those in isolation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config, sequential
from ..simulate import expected_sample_size, simulate_looks

HORIZON = 14
N_PER_DAY = 100
N_REPS = 20_000
# Sized for ~80% power at the fixed horizon, so the power column has room to
# move in both directions.
TRUE_EFFECT = 0.10


def run() -> pd.DataFrame:
    # Calibration uses its own seed; every evaluation below uses different ones.
    # Reusing the calibration draw would report the boundary's training error.
    crit = sequential.calibrate_pocock(
        n_looks=HORIZON, alpha=0.05, seed=999_001, n_per_day=N_PER_DAY
    )
    print(f"calibrated Pocock boundary: |z| > {crit:.3f}  (vs 1.960 uncorrected)")

    null = simulate_looks(
        n_reps=N_REPS, n_per_day=N_PER_DAY, horizon_days=HORIZON, effect=0.0, seed=1
    )
    alt = simulate_looks(
        n_reps=N_REPS, n_per_day=N_PER_DAY, horizon_days=HORIZON,
        effect=TRUE_EFFECT, seed=2,
    )

    rules = {
        "fixed horizon (test once)": lambda d: sequential.fixed_horizon(d),
        "peek daily, stop at p<0.05": lambda d: sequential.naive_peeking(d),
        "peek daily, Pocock boundary": lambda d: sequential.pocock_boundary(d, crit),
        # tau is tuned to the effect being simulated, which is what a
        # practitioner does with their MDE. The default tau=sigma is a bad
        # choice here and `sweep_tau` below quantifies how bad.
        "peek daily, mSPRT (always-valid)": lambda d: sequential.msprt(d, tau=TRUE_EFFECT),
    }

    rows = []
    for name, rule in rules.items():
        fp, _ = rule(null)
        tp, stop_alt = rule(alt)
        rows.append({
            "rule": name,
            "type-I error": round(float(fp.mean()), 4),
            "power": round(float(tp.mean()), 4),
            "avg n/arm at stop": int(expected_sample_size(stop_alt, alt.n_per_arm)),
        })

    out = pd.DataFrame(rows)
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.REPORTS / "peeking.csv", index=False)
    print()
    print(out.to_string(index=False))
    naive = out.loc[out["rule"].str.startswith("peek daily, stop"), "type-I error"].iloc[0]
    print(f"\nnominal 5% test, run daily for {HORIZON} days, actually errs {naive:.1%} of the time")
    print(f"-> {config.REPORTS / 'peeking.csv'}")
    return out


def sweep_tau(null=None, alt=None) -> pd.DataFrame:
    """mSPRT's power depends sharply on `tau`, and the library default is not a
    safe choice. Worth its own table because "use always-valid p-values" is
    common advice that omits the one parameter that decides whether it works.
    """
    null = null or simulate_looks(
        n_reps=N_REPS, n_per_day=N_PER_DAY, horizon_days=HORIZON, effect=0.0, seed=1
    )
    alt = alt or simulate_looks(
        n_reps=N_REPS, n_per_day=N_PER_DAY, horizon_days=HORIZON,
        effect=TRUE_EFFECT, seed=2,
    )
    rows = []
    for tau in (1.0, 0.5, 0.25, 0.10, 0.05, 0.02):
        fp, _ = sequential.msprt(null, tau=tau)
        tp, stop = sequential.msprt(alt, tau=tau)
        rows.append({
            "tau": tau,
            "tau / true effect": round(tau / TRUE_EFFECT, 2),
            "type-I error": round(float(fp.mean()), 4),
            "power": round(float(tp.mean()), 4),
            "avg n/arm at stop": int(expected_sample_size(stop, alt.n_per_arm)),
        })
    out = pd.DataFrame(rows)
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.REPORTS / "msprt_tau.csv", index=False)
    print("\nmSPRT sensitivity to tau (true effect = %.2f):" % TRUE_EFFECT)
    print(out.to_string(index=False))
    return out


if __name__ == "__main__":
    run()
    sweep_tau()
