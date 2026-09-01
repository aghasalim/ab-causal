# A/B testing and causal inference, checking the methods against known answers

**[▶ Live demo](https://ab-causal.streamlit.app/)** · analyse a live test with a
threshold that adjusts for how many times you've looked, and an Evidence tab
showing the simulations each rule was scored against.

[![ci](https://github.com/aghasalim/ab-causal/actions/workflows/ci.yml/badge.svg)](https://github.com/aghasalim/ab-causal/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Built by a third-year Applied Computer Science (AI) student.

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

---


---

## Abstract

A/B testing and causal inference are both areas where the correct answer is
usually unknown, so methods get adopted on plausibility. This work scores three of
them against known answers.

**Peeking.** Testing daily and stopping at p<0.05 turns a nominal 5% test into a
22.3% one. Both standard corrections restore it, Pocock to 5.0%, mSPRT to 0.9%
and both cost power, dropping from 0.750 to 0.601 and 0.419. Nothing here is free,
and the sample saving is what you are buying with that power.

**CUPED.** Variance reduction tracks the theoretical rho^2 across the correlation
range, and the bias stays at zero, which is the check that the implementation does
what the derivation says.

**LaLonde.** Because the randomised answer is known ($1,794), observational
estimators can be scored rather than argued about. The naive difference on
observational controls returns -$8,498 on CPS and -$15,205 on PSID: wrong by more
than five times the effect, and the wrong sign. Adjustment recovers the ballpark,
but the overlap diagnostics show why it is fragile, PSID keeps 1,068 of 2,490
controls inside the treated propensity range, and a single control can carry an
IPW weight of 93.8.

**Contributions.** (i) Sequential-testing rules scored on type-I error, power and
sample together, so the trade is visible. (ii) A CUPED implementation validated
against its own theory. (iii) LaLonde estimates against the experimental benchmark
with balance and overlap reported as preconditions rather than results.

---

## 1. Three things I measured

### 1. Checking your test daily turns a 5% error rate into 22%
20,000 simulated A/A tests, no real effect at all, looked at once a day for 14 days.
The fixed-horizon test errs 5.2% of the time, which is what a correct 5% test looks
like, and that is what makes the next number believable: peeking daily and stopping
at p<0.05 errs 22.3% of the time. Pocock pulls it back to 5.0% and mSPRT to 0.9%.
Both pay for it in power, 75.0% at fixed horizon against 60.1% and 42.0%.

Full detail in [notes/METHODS.md](notes/METHODS.md#1-checking-your-test-daily-turns-a-5-error-rate-into-22).
### 2. CUPED works exactly as advertised, until the covariate is downstream of treatment
Variance reduction tracks the theoretical ρ² closely (`make cuped`):

| corr(X, Y) | 0.3 | 0.5 | 0.7 | 0.9 |
|---|---|---|---|---|
| measured reduction | 0.085 | 0.255 | 0.526 | 0.810 |
| predicted (ρ²) | 0.09 | 0.25 | 0.49 | 0.81 |

At ρ=0.9 that's 81% less variance, the same precision from roughly five times fewer users, for free, and unbiased throughout.

That guarantee rests on the covariate being measured before randomisation. When
treatment moves the covariate instead, CUPED subtracts the effect away: with all of
a true 0.10 effect flowing through it, the estimate comes back 0.000. Its standard
error stays at 0.019, the same as in the column where it is right, so nothing looks
unstable.

Full detail in [notes/METHODS.md](notes/METHODS.md#2-cuped-works-exactly-as-advertised-until-the-covariate-is-downstream-of-treatment).
### 3. Every observational method got close to the right answer, and I could only tell because I already knew it
The [LaLonde/NSW](https://users.nber.org/~rdehejia/nswdata.html) job-training programme was randomised, so the honest effect is known: **+$1,794** (SE $671).

Adjustment gets back to the right neighbourhood, and that is the trap. Regression,
IPW, matching and the doubly-robust estimators give 20 adjusted estimates spanning
$237 to $3,843, with the true $1,794 sitting inside that range along with almost
everything else. The closest is IPW on the Dehejia-Wahba specification with
trimming, $1,764, off by $31. Picking that one out as the winner needed the
experimental answer, which on real observational data I would not have.

![peeking, and what the corrections cost](reports/figures/peeking.png)
![CUPED against its own theory](reports/figures/cuped.png)
![observational estimates against the randomised benchmark](reports/figures/lalonde.png)
![covariate balance before and after](reports/figures/balance.png)
![how much of the control pool is usable](reports/figures/overlap.png)

Full detail in [notes/METHODS.md](notes/METHODS.md#3-every-observational-method-got-close-to-the-right-answer-and-i-could-only-tell-because-i-already-knew-it).
## 2. Running it

```bash
make setup && make experiments
```

Reproduces every number above. No credentials, no API keys, no paid data, the
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

## 3. Design notes
**Why simulate first.** Every decision rule is scored on `simulate.py` before it touches real data. `simulate_looks` returns the z-statistic at every interim look and all rules consume that same matrix, so comparisons are paired, naive peeking and the corrected boundary see byte-identical experiments, and differences between them aren't simulation noise.

Full detail in [notes/METHODS.md](notes/METHODS.md#3-design-notes).
## 4. Every result is derived twice

The whole point of this repo is that a causal estimate has nothing to check
against, so I only use cases where the true answer is known. That argument has a
hole in it: every number here comes out of one Python simulation, and the tests
check that the simulation runs, not that it is right. A simulation is exactly the
kind of code where a wrong answer still looks entirely plausible.

So the published results are re-derived by eight implementations in eight
languages, and CI fails if any of them disagrees. Most of them draw their own
random numbers rather than reading the Python's output, so they are replications
and not just recomputations.

| implementation | what it re-derives | how |
| --- | --- | --- |
| [`verify/lalonde.sql`](verify/lalonde.sql) | the LaLonde estimate table and its spread | SQLite, from the raw estimates |
| [`verify/cuped_kernel.c`](verify/cuped_kernel.c) | that measured variance reduction equals rho squared | its own simulation, all 5 rows within 4 sd |
| [`verify/gocheck`](verify/gocheck) | every file in `reports/` is well formed, every derived column rederives | structural |
| [`verify/verify.R`](verify/verify.R) | the peeking and mSPRT error rates | base R, its own draws, within 4 standard errors |
| [`verify/Pocock.java`](verify/Pocock.java) | the Pocock boundary | recalibrates it to 2.6248 from its own draws |
| [`verify/readme_claims.rb`](verify/readme_claims.rb) | every figure quoted in the prose against the file it came from | text to data |
| [`verify/post_treatment.mjs`](verify/post_treatment.mjs) | the mediation table, and that CUPED returns the direct effect | its own simulation |
| [`verify/peekmc`](verify/peekmc) | how much of each published rate is Monte Carlo noise | 200,000 replications against the published 20,000 |

Run them with [`./verify/verify.sh`](verify/verify.sh). Each is skipped with a
message if its toolchain is missing.

**What this caught.** Corrupting the variance reduction at rho = 0.9 is rejected
by the C, the Go and the Ruby. Corrupting the headline 22.3% peeking rate is
rejected by the R and the Rust, both of which re-simulate rather than read it,
and by the Ruby. Changing only the prose, leaving every data file untouched, is
rejected by the Ruby alone. CI does this to itself on every run: it corrupts
`reports/peeking.csv`, requires the harness to reject it, restores it and
requires a pass, because a check that cannot fail is not evidence.

**What the Rust adds.** The published rates are 20,000 replication estimates and
carry their own Monte Carlo error, which nothing here had measured. Running
200,000 replications gives a reference each published rate can be compared
against, and every one lands inside it. That was an assumption before.

## 5. Repository layout

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

## 6. Licence

MIT, see [LICENSE](LICENSE). LaLonde data is public, courtesy of Rajeev Dehejia
and NBER.

## References

The papers and sources this implementation follows. Each one is here because
the code uses the method, the dataset or the metric it describes.

- **Deng, Xu, Kohavi, Walker. Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data. WSDM 2013.** CUPED, the variance reduction implemented here.
- **Rosenbaum, Rubin. The Central Role of the Propensity Score in Observational Studies for Causal Effects. Biometrika 70, 1983.** propensity scores.
- **Kohavi, Tang, Xu. Trustworthy Online Controlled Experiments. Cambridge University Press, 2020.** the experiment design practices the harness checks.
