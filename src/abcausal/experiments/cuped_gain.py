"""What CUPED buys, and what it costs when the covariate is the wrong one."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config, cuped

N_REPS = 4_000
N_PER_ARM = 2_000
TRUE_EFFECT = 0.10


def _draw(rng, rho: float, effect: float = TRUE_EFFECT):
    """Pre-period covariate X correlated rho with the outcome's noise."""
    x_t, x_c = rng.normal(size=N_PER_ARM), rng.normal(size=N_PER_ARM)
    e_t, e_c = rng.normal(size=N_PER_ARM), rng.normal(size=N_PER_ARM)
    noise_t = rho * x_t + np.sqrt(1 - rho**2) * e_t
    noise_c = rho * x_c + np.sqrt(1 - rho**2) * e_c
    return x_t, x_c, effect + noise_t, noise_c


def run() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for rho in (0.0, 0.3, 0.5, 0.7, 0.9):
        plain, adj = [], []
        for _ in range(N_REPS):
            x_t, x_c, y_t, y_c = _draw(rng, rho)
            plain.append(cuped.ate(y_t, y_c)["ate"])
            adj.append(cuped.ate(y_t, y_c, x_t, x_c)["ate"])
        plain, adj = np.array(plain), np.array(adj)
        vr = 1 - adj.var() / plain.var()
        rows.append({
            "corr(X, Y)": rho,
            "bias (plain)": round(float(plain.mean() - TRUE_EFFECT), 5),
            "bias (CUPED)": round(float(adj.mean() - TRUE_EFFECT), 5),
            "variance reduction": round(float(vr), 4),
            # Var goes from V to V(1-rho^2), so the *reduction* is rho^2.
            "predicted (rho^2)": round(rho**2, 4),
            "equiv. sample saving": f"{vr:.0%}",
        })
    out = pd.DataFrame(rows)
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.REPORTS / "cuped.csv", index=False)
    print(out.to_string(index=False))
    return out


BETA = 0.8  # how strongly the covariate drives the outcome


def demo_post_treatment_bias() -> pd.DataFrame:
    """Adjusting on a covariate that treatment itself moved.

    Causal structure: treatment -> X -> Y, plus a direct path. So the total
    effect splits into a mediated part and a direct part, held at 0.10 in total
    throughout.

    CUPED subtracts the variation X explains -- including the part of it that
    *is* the treatment working. What comes back is the direct effect only. The
    procedure is not malfunctioning; it is answering a different question than
    the one being asked, and nothing in the output says so. The standard error
    is unchanged, so the usual diagnostic is blind to it.
    """
    rng = np.random.default_rng(11)
    rows = []
    for mediated in (0.0, 0.25, 0.5, 1.0):
        shift = mediated * TRUE_EFFECT / BETA  # treatment's push on X
        direct = (1 - mediated) * TRUE_EFFECT
        plain_e, adj_e, adj_se = [], [], []
        for _ in range(N_REPS):
            x_c = rng.normal(size=N_PER_ARM)
            x_t = rng.normal(size=N_PER_ARM) + shift
            y_c = BETA * x_c + rng.normal(size=N_PER_ARM) * 0.6
            y_t = direct + BETA * x_t + rng.normal(size=N_PER_ARM) * 0.6
            plain_e.append(cuped.ate(y_t, y_c)["ate"])
            r = cuped.ate(y_t, y_c, x_t, x_c)
            adj_e.append(r["ate"])
            adj_se.append(r["se"])
        rows.append({
            "% of effect via covariate": f"{mediated:.0%}",
            "true TOTAL effect": TRUE_EFFECT,
            "plain diff-in-means": round(float(np.mean(plain_e)), 4),
            "CUPED estimate": round(float(np.mean(adj_e)), 4),
            "CUPED error vs total": round(float(np.mean(adj_e) - TRUE_EFFECT), 4),
            "CUPED SE (unchanged)": round(float(np.mean(adj_se)), 4),
        })
    out = pd.DataFrame(rows)
    out.to_csv(config.REPORTS / "cuped_post_treatment_bias.csv", index=False)
    print("\nCUPED on a covariate that treatment moved (total effect fixed at 0.10):")
    print(out.to_string(index=False))
    return out


if __name__ == "__main__":
    run()
    demo_post_treatment_bias()
