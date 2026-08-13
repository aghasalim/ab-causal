"""Which observational estimators recover the experimental answer?

Scored against +$1,794 from the randomised NSW sample. Reported as absolute
error against that benchmark, per control pool and per specification, because
the interesting result is not "matching works" but *how much the answer moves
when you change something you had no principled reason to choose*.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config, observational as ob


def run() -> pd.DataFrame:
    bench = ob.experimental_benchmark()
    truth = bench["att"]
    print(f"experimental benchmark: ${truth:,.0f} (SE ${bench['se']:,.0f})\n")

    rows = [{"controls": "randomised", "spec": "-", **bench, "abs error": 0.0}]

    for pool in ("cps", "psid"):
        d = ob.load(pool)
        n = int(d[config.TREATMENT].eq(0).sum())
        rows.append({"controls": pool, "spec": "-", **ob.naive(d),
                     "n_control": n, "abs error": abs(ob.naive(d)["att"] - truth)})

        for spec in ("linear", "dehejia-wahba"):
            ps = ob.propensity(d, spec)
            ests = [
                ob.ols(d, spec),
                ob.ipw(d, ps, trim=0.0),
                ob.ipw(d, ps, trim=0.01),
                ob.match_nn(d, ps),
                ob.aipw(d, ps, spec),
            ]
            for e in ests:
                rows.append({"controls": pool, "spec": spec, **e,
                             "abs error": abs(e["att"] - truth)})

    out = pd.DataFrame(rows)
    out["att"] = out["att"].round(0)
    out["abs error"] = out["abs error"].round(0)
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.REPORTS / "lalonde.csv", index=False)

    cols = ["controls", "spec", "method", "att", "abs error"]
    print(out[cols].to_string(index=False))

    adj = out[out["spec"] != "-"]
    print(f"\nspread across adjusted estimates: "
          f"${adj['att'].min():,.0f} to ${adj['att'].max():,.0f}")
    print(f"-> {config.REPORTS / 'lalonde.csv'}")
    return out


def balance_report() -> pd.DataFrame:
    """Covariate balance before and after weighting, CPS pool."""
    d = ob.load("cps")
    ps = ob.propensity(d, "dehejia-wahba")
    before = ob.standardised_diff(d).rename(columns={"smd": "before"})
    after = ob.standardised_diff(d, weights=ps / (1 - ps)).rename(columns={"smd": "after"})
    m = before.merge(after, on="covariate")
    m["|before|"] = m["before"].abs().round(3)
    m["|after|"] = m["after"].abs().round(3)
    out = m[["covariate", "|before|", "|after|"]]
    out.to_csv(config.REPORTS / "lalonde_balance.csv", index=False)
    print("\ncovariate balance, CPS pool (|SMD| < 0.1 is the usual threshold):")
    print(out.to_string(index=False))
    return out


def overlap_report() -> pd.DataFrame:
    """How much of the control pool is even comparable to a trainee?"""
    rows = []
    for pool in ("cps", "psid"):
        d = ob.load(pool)
        ps = ob.propensity(d, "dehejia-wahba")
        t = d[config.TREATMENT].to_numpy().astype(bool)
        lo, hi = ps[t].min(), ps[t].max()
        rows.append({
            "controls": pool,
            "n control": int((~t).sum()),
            "controls inside treated PS range": int(((ps[~t] >= lo) & (ps[~t] <= hi)).sum()),
            "controls with PS < 0.01": int((ps[~t] < 0.01).sum()),
            "max control weight ps/(1-ps)": round(float((ps[~t] / (1 - ps[~t])).max()), 1),
        })
    out = pd.DataFrame(rows)
    out.to_csv(config.REPORTS / "lalonde_overlap.csv", index=False)
    print("\noverlap:")
    print(out.to_string(index=False))
    return out


if __name__ == "__main__":
    run()
    balance_report()
    overlap_report()
