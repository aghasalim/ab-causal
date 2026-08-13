"""Recovering an experimental answer from observational data -- LaLonde/NSW.

This is the honest test of any causal method, because the answer is known. The
NSW job-training programme was randomised, so the difference in means over the
experimental sample is unbiased: **+$1,794**. Replacing the randomised controls
with survey respondents (CPS or PSID) gives observational data on the *same*
treated people, where the naive comparison reports large negative effects.

An estimator is therefore not being judged on plausibility here. It either gets
back to roughly $1,794 or it does not.

Estimand: ATT, the effect on the treated. The control pools are a different
population from the trainees, so the population-average effect is not what the
experiment measured and not what these methods should target.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import config

SPECS = {
    # Linear in the covariates -- the obvious first thing to try.
    "linear": lambda d: d[config.COVARIATES],
    # Dehejia & Wahba's richer specification. Included because the choice of
    # specification turns out to matter more than the choice of estimator.
    "dehejia-wahba": lambda d: _dw_design(d),
}


def _dw_design(d: pd.DataFrame) -> pd.DataFrame:
    x = d[config.COVARIATES].copy()
    x["age2"] = d["age"] ** 2
    x["educ2"] = d["education"] ** 2
    x["re74_2"] = d["re74"] ** 2
    x["re75_2"] = d["re75"] ** 2
    x["u74"] = (d["re74"] == 0).astype(float)
    x["u75"] = (d["re75"] == 0).astype(float)
    x["age_educ"] = d["age"] * d["education"]
    return x


def load(controls: str = "cps") -> pd.DataFrame:
    """Treated units from the experiment + a non-experimental control pool."""
    exp = pd.read_stata(config.RAW / "nsw_dw.dta")
    ctl = pd.read_stata(config.RAW / f"{controls}_controls.dta")
    return pd.concat([exp[exp[config.TREATMENT] == 1], ctl], ignore_index=True)


def experimental_benchmark() -> dict:
    d = pd.read_stata(config.RAW / "nsw_dw.dta")
    t = d.loc[d[config.TREATMENT] == 1, config.OUTCOME]
    c = d.loc[d[config.TREATMENT] == 0, config.OUTCOME]
    se = np.sqrt(t.var(ddof=1) / len(t) + c.var(ddof=1) / len(c))
    return {"method": "randomised experiment", "att": float(t.mean() - c.mean()),
            "se": float(se), "n_control": int(len(c))}


def naive(d: pd.DataFrame) -> dict:
    t = d.loc[d[config.TREATMENT] == 1, config.OUTCOME]
    c = d.loc[d[config.TREATMENT] == 0, config.OUTCOME]
    se = np.sqrt(t.var(ddof=1) / len(t) + c.var(ddof=1) / len(c))
    return {"method": "naive difference", "att": float(t.mean() - c.mean()), "se": float(se)}


def ols(d: pd.DataFrame, spec: str = "linear") -> dict:
    x = SPECS[spec](d)
    X = sm.add_constant(pd.concat([d[[config.TREATMENT]], x], axis=1).astype(float))
    m = sm.OLS(d[config.OUTCOME].astype(float), X).fit(cov_type="HC1")
    return {"method": f"OLS ({spec})", "att": float(m.params[config.TREATMENT]),
            "se": float(m.bse[config.TREATMENT])}


def propensity(d: pd.DataFrame, spec: str = "linear") -> np.ndarray:
    X = sm.add_constant(SPECS[spec](d).astype(float))
    m = sm.Logit(d[config.TREATMENT].astype(float), X).fit(disp=0, maxiter=200)
    return m.predict(X).to_numpy()


def ipw(d: pd.DataFrame, ps: np.ndarray, trim: float = 0.0) -> dict:
    """ATT via inverse-probability weighting: controls reweighted by ps/(1-ps).

    `trim` drops units with extreme propensity scores. Without it a single
    control with ps near 1 can dominate the estimate -- the weights are
    unbounded, which is the practical failure mode of IPW.
    """
    t = d[config.TREATMENT].to_numpy().astype(bool)
    y = d[config.OUTCOME].to_numpy().astype(float)
    keep = (ps > trim) & (ps < 1 - trim) if trim > 0 else np.ones(len(d), bool)
    t, y, ps = t[keep], y[keep], ps[keep]

    w = np.where(t, 1.0, ps / (1 - ps))
    yt = y[t].mean()
    yc = np.average(y[~t], weights=w[~t])
    n_eff = w[~t].sum() ** 2 / (w[~t] ** 2).sum()  # Kish effective sample size
    return {"method": f"IPW (trim={trim})", "att": float(yt - yc),
            "n_effective_controls": int(n_eff), "n_dropped": int((~keep).sum())}


def match_nn(d: pd.DataFrame, ps: np.ndarray) -> dict:
    """1-NN propensity matching with replacement, ATT."""
    t = d[config.TREATMENT].to_numpy().astype(bool)
    y = d[config.OUTCOME].to_numpy().astype(float)
    pt, pc, yc = ps[t], ps[~t], y[~t]
    idx = np.abs(pt[:, None] - pc[None, :]).argmin(axis=1)
    return {"method": "1-NN PS matching", "att": float((y[t] - yc[idx]).mean()),
            "n_unique_controls_used": int(len(np.unique(idx)))}


def aipw(d: pd.DataFrame, ps: np.ndarray, spec: str = "linear") -> dict:
    """Doubly robust: consistent if *either* the outcome model or the
    propensity model is right, which is a weaker requirement than either alone."""
    t = d[config.TREATMENT].to_numpy().astype(bool)
    y = d[config.OUTCOME].to_numpy().astype(float)
    X = sm.add_constant(SPECS[spec](d).astype(float))
    m0 = sm.OLS(y[~t], X[~t]).fit()
    mu0 = m0.predict(X).to_numpy()
    w = ps / (1 - ps)
    corr = (w[~t] * (y[~t] - mu0[~t])).sum() / t.sum()
    return {"method": f"AIPW ({spec})", "att": float((y[t] - mu0[t]).mean() - corr)}


def standardised_diff(d: pd.DataFrame, weights: np.ndarray | None = None) -> pd.DataFrame:
    """Standardised mean differences per covariate. |SMD| < 0.1 is the usual
    rule of thumb for adequate balance."""
    t = d[config.TREATMENT].to_numpy().astype(bool)
    rows = []
    for c in config.COVARIATES:
        v = d[c].to_numpy().astype(float)
        if weights is None:
            mc, vc = v[~t].mean(), v[~t].var()
        else:
            w = weights[~t]
            mc = np.average(v[~t], weights=w)
            vc = np.average((v[~t] - mc) ** 2, weights=w)
        pooled = np.sqrt((v[t].var() + vc) / 2)
        rows.append({"covariate": c,
                     "smd": float((v[t].mean() - mc) / pooled) if pooled else 0.0})
    return pd.DataFrame(rows)
