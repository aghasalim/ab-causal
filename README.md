# A/B testing and causal inference, checking the methods against known answers

**[▶ Live demo](https://ab-causal.streamlit.app/)** · analyse a live test with a
threshold that adjusts for how many times you've looked, and an Evidence tab
showing the simulations each rule was scored against.

[![ci](https://github.com/aghasalim/ab-causal/actions/workflows/ci.yml/badge.svg)](https://github.com/aghasalim/ab-causal/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

The problem with a causal inference project is that you can't tell whether it
worked. A prediction model can be checked against a held-out label. An estimate
of "what would have happened otherwise" has nothing to check against, because the
otherwise never happened. So the usual portfolio version of this, run a t-test
on a marketing dataset, report a p-value, proves nothing, because it would look
identical if the method were completely wrong.

So this repo only uses situations where **the true answer is known**: simulations
where I set the effect myself, and one famous dataset where a randomised
experiment already told us the answer. Every method gets scored against that
before I'd trust it anywhere else.

Three methods went in. Below is what each one scored, and what each one cost.

---

## Does peeking really break a test? Yes, four-fold

20,000 simulated A/A tests, no real effect at all, looked at once a day for 14 days.
The fixed-horizon test errs 5.2% of the time, which is what a correct 5% test looks
like, and that is what makes the next number believable: peeking daily and stopping
at p<0.05 turns a nominal 5% test into a 22.3% one.

Both standard corrections restore it, Pocock to 5.0%, mSPRT to 0.9%, and both cost
power, dropping from 0.750 to 0.601 and 0.419. Put as percentages, that is
power, 75.0% at fixed horizon against 60.1% and 42.0%. Nothing here is free, and
the sample saving is what you are buying with that power.

![peeking, and what the corrections cost](reports/figures/peeking.png)

Worked through at length in [notes/METHODS.md](notes/METHODS.md#1-checking-your-test-daily-turns-a-5-error-rate-into-22).

## Does CUPED deliver what the derivation promises? Yes, until you break its one assumption

Variance reduction tracks the theoretical ρ² closely (`make cuped`):

| corr(X, Y) | 0.3 | 0.5 | 0.7 | 0.9 |
|---|---|---|---|---|
| measured reduction | 0.085 | 0.255 | 0.526 | 0.810 |
| predicted (ρ²) | 0.09 | 0.25 | 0.49 | 0.81 |

At ρ=0.9 that's 81% less variance, the same precision from roughly five times fewer
users, for free, and unbiased throughout. Matching theory across the whole
correlation range is the check that the implementation does what the derivation
says, rather than something that merely looks like it.

That guarantee rests on the covariate being measured before randomisation. When
treatment moves the covariate instead, CUPED subtracts the effect away: with all of
a true 0.10 effect flowing through it, the estimate comes back 0.000. Its standard
error stays at 0.019, the same as in the column where it is right, so nothing looks
unstable. A silent zero with a healthy error bar is the worst failure mode a
variance-reduction technique can have.

![CUPED against its own theory](reports/figures/cuped.png)

Worked through at length in [notes/METHODS.md](notes/METHODS.md#2-cuped-works-exactly-as-advertised-until-the-covariate-is-downstream-of-treatment).

## Can observational estimators recover a randomised answer? Close enough to fool me

The [LaLonde/NSW](https://users.nber.org/~rdehejia/nswdata.html) job-training
programme was randomised, so the honest effect is known: **+$1,794** (SE $671).
That single fact is what lets the rest of this section be measurement rather than
argument.

Start with the failure. The naive difference on observational controls
returns -$8,498 on CPS and -$15,205 on PSID: wrong by more than five times the
effect, and the wrong sign.

Adjustment gets back to the right neighbourhood, and that is the trap. Regression,
IPW, matching and the doubly-robust estimators give 20 adjusted estimates spanning
$237 to $3,843, with the true $1,794 sitting inside that range along with almost
everything else. The closest is IPW on the Dehejia-Wahba specification with
trimming, $1,764, off by $31. Picking that one out as the winner needed the
experimental answer, which on real observational data I would not have.

The overlap diagnostics show why it is fragile. PSID keeps 1,068 of 2,490
controls inside the treated propensity range, and a single control can carry an
IPW weight of 93.8. Balance and overlap are reported here as preconditions,
not as results.

![observational estimates against the randomised benchmark](reports/figures/lalonde.png)
![covariate balance before and after](reports/figures/balance.png)
![how much of the control pool is usable](reports/figures/overlap.png)

Worked through at length in [notes/METHODS.md](notes/METHODS.md#3-every-observational-method-got-close-to-the-right-answer-and-i-could-only-tell-because-i-already-knew-it).

---

## How to run the three studies

```bash
make setup && make experiments
```

Reproduces every number above. No credentials, no API keys, no paid data, the
simulations are self-contained and LaLonde is public. Every published number is
also recomputed independently by the implementations in `verify/`, and CI fails
the build if any of them disagrees.

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

## Why the simulator comes first

Every decision rule is scored on `simulate.py` before it touches real data.
`simulate_looks` returns the z-statistic at every interim look and all rules
consume that same matrix, so comparisons are paired: naive peeking and the
corrected boundary see byte-identical experiments, and differences between them
aren't simulation noise.

Worked through at length in [notes/METHODS.md](notes/METHODS.md#3-design-notes).

## What lives where

```
src/abcausal/
  simulate.py       simulation harness, truth is known by construction
  sequential.py     fixed-horizon, naive peeking, Pocock, mSPRT
  cuped.py          variance reduction, and how it breaks
  observational.py  OLS, IPW, matching, AIPW, balance and overlap diagnostics
  diagnostics.py    SRM, MDE, required sample size
  experiments/      the three runnable studies above
app/                Streamlit analyser
tests/              12 tests asserting the claims
```

MIT licensed, terms in [LICENSE](LICENSE). The LaLonde data is public, courtesy
of Rajeev Dehejia and NBER.

## Where the three methods come from

- **Deng, Xu, Kohavi, Walker. Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data. WSDM 2013.** CUPED, the variance reduction implemented here.
- **Rosenbaum, Rubin. The Central Role of the Propensity Score in Observational Studies for Causal Effects. Biometrika 70, 1983.** propensity scores.
- **Kohavi, Tang, Xu. Trustworthy Online Controlled Experiments. Cambridge University Press, 2020.** the experiment design practices the harness checks.
