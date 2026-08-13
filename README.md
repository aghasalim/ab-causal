# A/B testing and causal inference — checking the methods against known answers

[![ci](https://github.com/aghasalim/ab-causal/actions/workflows/ci.yml/badge.svg)](https://github.com/aghasalim/ab-causal/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Built by a third-year Applied Computer Science (AI) student.

The problem with a causal inference project is that you can't tell whether it
worked. A prediction model can be checked against a held-out label. An estimate
of "what would have happened otherwise" has nothing to check against, because the
otherwise never happened. So the usual portfolio version of this — run a t-test
on a marketing dataset, report a p-value — proves nothing, because it would look
identical if the method were completely wrong.

So this repo only uses situations where **the true answer is known**: simulations
where I set the effect myself, and one famous dataset where a randomised
experiment already told us the answer. Every method gets scored against that
before I'd trust it anywhere else.

---

## Three things I measured

### 1. Checking your test daily turns a 5% error rate into 22%

20,000 simulated A/A tests — no real effect at all — looked at once a day for 14
days. Reproduce with `make peeking`.

| decision rule | type-I error | power | avg n/arm at stop |
|---|---|---|---|
| fixed horizon (test once) | **5.2%** | 75.0% | 1400 |
| peek daily, stop at p<0.05 | **22.3%** | 84.2% | 702 |
| peek daily, Pocock boundary | 5.0% | 60.1% | 1010 |
| peek daily, mSPRT (always-valid) | 0.9% | 42.0% | 1198 |

The first row is the sanity check: a correct fixed test errs 5.2% of the time,
which is what 5% should look like. That's what makes the second row believable.
**A "95% confident" result, checked daily for two weeks, is wrong 22% of the
time.**

The trap in this table is the power column. Naive peeking has the *highest*
power (84%), which looks like an argument for it. It isn't — it declares
significance more often whether or not anything is there. You cannot read that
column without the one next to it, and a dashboard only shows you one of them.

Both corrections work, and neither is free. The Pocock boundary (|z| > 2.63
instead of 1.96, calibrated by simulation on a different seed than it's validated
on) restores 5% error and still stops early on average — 1010 users instead of
1400. mSPRT is *more* conservative than asked: 0.9% error against a nominal 5%.
That's not a bug. Its guarantee holds at every sample size simultaneously, which
is strictly stronger than holding at 14 planned looks, and the ~18 points of
power is what that stronger guarantee costs.

**mSPRT is also not plug-and-play**, which I only found by getting it wrong. It
needs a `tau` telling it what effect size to expect, and the obvious default
(`tau = sigma`) gave 25% power where a tuned one gave 42%:

| tau / true effect | 10× | 5× | 2.5× | **1×** | 0.5× | 0.2× |
|---|---|---|---|---|---|---|
| power | 25.0% | 32.2% | 39.1% | **42.0%** | 28.4% | 0.4% |

Set it 5× too small and the method essentially stops working. "Use always-valid
p-values" is common advice that usually omits the one parameter deciding whether
it does anything.

### 2. CUPED works exactly as advertised — until the covariate is downstream of treatment

Variance reduction tracks the theoretical ρ² closely (`make cuped`):

| corr(X, Y) | 0.3 | 0.5 | 0.7 | 0.9 |
|---|---|---|---|---|
| measured reduction | 0.085 | 0.255 | 0.526 | 0.810 |
| predicted (ρ²) | 0.09 | 0.25 | 0.49 | 0.81 |

At ρ=0.9 that's 81% less variance — the same precision from roughly five times
fewer users, for free, and unbiased throughout.

The failure mode is what I'd actually want to talk about. CUPED's guarantee rests
entirely on the covariate being measured *before* randomisation. Use one that
treatment moved, and it subtracts away part of the effect you're measuring:

| % of effect flowing through the covariate | 0% | 25% | 50% | 100% |
|---|---|---|---|---|
| plain difference in means | 0.100 | 0.101 | 0.099 | 0.101 |
| **CUPED estimate** | 0.100 | 0.075 | 0.050 | **0.000** |
| CUPED standard error | 0.019 | 0.019 | 0.019 | **0.019** |

True effect is 0.10 in every column. In the last one, CUPED reports **zero
effect** on a feature that works perfectly — and the standard error is identical
to the column where it's right. Nothing gets wider, nothing looks unstable. You'd
ship "no impact" with a tight confidence interval.

I got this wrong myself first: my initial test asserted CUPED was "biased" and it
failed, because CUPED was correctly returning the *direct* effect while I was
calling the *total* effect the truth. The real hazard isn't that the number is
wrong, it's that it silently answers a different question.

### 3. Every observational method got close to the right answer — and I could only tell because I already knew it

The [LaLonde/NSW](https://users.nber.org/~rdehejia/nswdata.html) job-training
programme was randomised, so the honest effect is known: **+$1,794** (SE $671).
The standard exercise replaces the randomised controls with survey respondents,
which is what observational data actually looks like (`make lalonde`).

| | estimate |
|---|---|
| **randomised experiment (truth)** | **+$1,794** |
| naive comparison, CPS controls | −$8,498 |
| naive comparison, PSID controls | −$15,205 |

The naive analysis doesn't just miss, it gets the **sign** wrong: a programme
that raised earnings looks like one that destroyed them, by five figures.

Adjustment helps enormously. Regression, IPW, propensity matching and doubly-robust
estimators all pull back into the right neighbourhood. But:

- across the 20 adjusted estimates, the answers span **$237 to $3,843**
- the true value sits inside that range, along with almost everything else
- switching between two defensible propensity specifications moved one estimator
  from $2,047 to $3,843

And the diagnostic you'd normally trust says everything is fine. After weighting,
covariate balance is textbook:

| covariate | \|SMD\| before | \|SMD\| after |
|---|---|---|
| black | 2.432 | 0.043 |
| re75 | 1.747 | 0.026 |
| married | 1.234 | 0.046 |
| age | 0.797 | 0.114 |

Every covariate under the usual 0.1 threshold, and the estimates still range 16×.
**Good balance is necessary, not sufficient.** It says the groups now look alike
on the things you measured; it's silent on everything you didn't.

Overlap explains the fragility. Of 15,992 CPS controls, only 4,472 fall inside
the treated propensity range and 14,705 have a propensity below 0.01 — most of
the "sample" contributes nothing. In PSID a single control carries a weight of
93.8, meaning one person stands in for 94.

The uncomfortable conclusion, and the reason I built it this way: the estimator
that nailed it here (IPW, Dehejia–Wahba specification, trimmed — $1,764, off by
$31) is only identifiable as the winner *because the experiment told me the
answer*. On real observational data I'd have had twenty numbers between $237 and
$3,843 and no way to choose. That's an argument for running experiments where you
can, and for reporting a spread rather than a point estimate where you can't.

---

## Running it

```bash
make setup && make experiments
```

Reproduces every number above. No credentials, no API keys, no paid data — the
simulations are self-contained and LaLonde is public.

```bash
make test
```

12 tests. They assert the *claims*, not just that the code runs: that the fixed-horizon
test really hits 5%, that peeking really inflates it, that the calibrated boundary
really restores control on a seed it wasn't calibrated on, and that CUPED on a
mediator really does erase the effect. If a refactor quietly broke a headline
result, these fail.

```bash
make app
```

An analyser with the checks worth running on a live test: sample-ratio mismatch,
a significance threshold that adjusts for how many times you've looked, and an
MDE calculator for deciding whether an experiment can answer its question before
you run it.

---

## Design notes

**Why simulate first.** Every decision rule is scored on `simulate.py` before it
touches real data. `simulate_looks` returns the z-statistic at every interim look
and all rules consume that same matrix, so comparisons are paired — naive peeking
and the corrected boundary see byte-identical experiments, and differences
between them aren't simulation noise.

**Why the boundary is calibrated rather than looked up.** A textbook Pocock
constant assumes equally spaced looks and known variance. Solving for it by
simulation makes it exact for the design actually being run. Calibration and
validation use different seeds, for the same reason you wouldn't report training
accuracy.

**Why data is accumulated per-user rather than drawn per-day.** Consecutive looks
share most of their data, so they're heavily correlated. That correlation *is*
why peeking misbehaves and why a Bonferroni correction over looks is the wrong
tool — it would treat 14 nested looks as 14 independent tests and badly
overcorrect.

**Why ATT rather than ATE for LaLonde.** The survey control pools are a different
population from the trainees, so the population-average effect isn't what the
experiment measured and isn't the right target.

---

## Layout

```
src/abcausal/
  simulate.py       simulation harness — truth is known by construction
  sequential.py     fixed-horizon, naive peeking, Pocock, mSPRT
  cuped.py          variance reduction, and how it breaks
  observational.py  OLS, IPW, matching, AIPW, balance and overlap diagnostics
  diagnostics.py    SRM, MDE, required sample size
  experiments/      the three runnable studies above
app/                Streamlit analyser
tests/              12 tests asserting the claims
```

## License

MIT — see [LICENSE](LICENSE). LaLonde data is public, courtesy of Rajeev Dehejia
and NBER.
