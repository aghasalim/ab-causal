# Independent re-simulation of the peeking experiment, in base R.
#
# Every number in reports/peeking.csv and reports/msprt_tau.csv comes out of one
# numpy simulation driven by src/abcausal/experiments/peeking.py. The tests
# check that peeking inflates the error rate, not that it inflates it to 22.3%.
# If the harness in simulate.py built the z-statistic wrongly, every rule scored
# against it would be wrong together and the tests would still pass, because
# they compare rules to each other.
#
# So this redraws the experiment from scratch. R's own generator, its own
# accumulation of the running mean and variance, its own decision rules, no
# shared code with the Python. The published rates are 20,000-replication Monte
# Carlo estimates, so they are required to land inside a binomial error band
# around the R estimate rather than to match exactly.
#
# No packages, so CI needs nothing beyond the R that is already on the runner.

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) > 0) args[1] else "."
set.seed(20260901)

N_REPS <- 20000      # same as the published run, so the bands are comparable
N_PER_DAY <- 100
HORIZON <- 14
TRUE_EFFECT <- 0.10
SIGMA <- 1.0
ALPHA <- 0.05
BAND <- 4            # standard errors of the difference of two estimates

failures <- 0

# One simulated experiment per row, observed once a day. Data is generated per
# user and accumulated, so successive looks share most of their data, which is
# the whole reason peeking misbehaves. Only running sums are kept.
simulate_looks <- function(effect) {
    z <- matrix(0.0, N_REPS, HORIZON)
    diff <- matrix(0.0, N_REPS, HORIZON)
    c_sum <- numeric(N_REPS); c_sq <- numeric(N_REPS)
    t_sum <- numeric(N_REPS); t_sq <- numeric(N_REPS)

    for (day in seq_len(HORIZON)) {
        ctrl <- matrix(rnorm(N_REPS * N_PER_DAY, 0, SIGMA), N_REPS, N_PER_DAY)
        trt <- matrix(rnorm(N_REPS * N_PER_DAY, effect, SIGMA), N_REPS, N_PER_DAY)
        c_sum <- c_sum + rowSums(ctrl); c_sq <- c_sq + rowSums(ctrl^2)
        t_sum <- t_sum + rowSums(trt);  t_sq <- t_sq + rowSums(trt^2)

        n <- day * N_PER_DAY
        c_mean <- c_sum / n; t_mean <- t_sum / n
        c_var <- (c_sq - n * c_mean^2) / (n - 1)   # unbiased, ddof = 1
        t_var <- (t_sq - n * t_mean^2) / (n - 1)
        diff[, day] <- t_mean - c_mean
        z[, day] <- diff[, day] / sqrt(c_var / n + t_var / n)
    }
    list(z = z, diff = diff, n = seq_len(HORIZON) * N_PER_DAY)
}

# Given a boolean (reps x looks) matrix of "boundary crossed here", the rule's
# decision and the look it stopped at. A replication that never crosses runs to
# its horizon, which is what actually happens.
stop_info <- function(crossed) {
    declared <- rowSums(crossed) > 0
    first <- max.col(crossed, ties.method = "first")
    first[!declared] <- ncol(crossed)
    list(declared = declared, stop = first)
}

fixed_horizon <- function(d) {
    crit <- qnorm(1 - ALPHA / 2)
    list(declared = abs(d$z[, HORIZON]) > crit, stop = rep(HORIZON, nrow(d$z)))
}

naive_peeking <- function(d) stop_info(abs(d$z) > qnorm(1 - ALPHA / 2))

# Mixture SPRT. Under the null the likelihood ratio against a N(0, tau^2)
# alternative is a non-negative martingale, so the running minimum of 1/LR is a
# p-value valid at every n at once.
msprt <- function(d, tau) {
    v <- matrix(2 * SIGMA^2 / d$n, nrow(d$z), HORIZON, byrow = TRUE)
    lr <- sqrt(v / (v + tau^2)) * exp(d$diff^2 * tau^2 / (2 * v * (v + tau^2)))
    # pmin copies attributes from its first argument, so the matrix goes first.
    p <- t(apply(pmin(1 / lr, 1), 1, cummin))
    stop_info(p < ALPHA)
}

cat("re-simulating", N_REPS, "experiments of", HORIZON, "daily looks at",
    N_PER_DAY, "users per arm per day\n\n")
null <- simulate_looks(0.0)
alt <- simulate_looks(TRUE_EFFECT)

peeking <- read.csv(file.path(root, "reports", "peeking.csv"),
                    check.names = FALSE)
tau_tab <- read.csv(file.path(root, "reports", "msprt_tau.csv"),
                    check.names = FALSE)

# Two independent Monte Carlo estimates of the same rate, both on N_REPS draws,
# differ with this standard error. Nothing here is tuned: the band is whatever
# the sample size says it is.
rate_se <- function(p) sqrt(2 * max(p * (1 - p), 1e-8) / N_REPS)

check_rate <- function(label, got, published) {
    se <- rate_se(got)
    zscore <- abs(got - published) / se
    ok <- zscore <= BAND
    failures <<- failures + !ok
    cat(sprintf("  %-34s R %.4f  published %.4f  se %.4f  %4.1f se  %s\n",
                label, got, published, se, zscore, if (ok) "ok" else "FAIL"))
}

check_n <- function(label, stops, n_per_arm, published) {
    ns <- n_per_arm[stops]
    got <- mean(ns)
    se <- sqrt(2) * sd(ns) / sqrt(N_REPS)
    # A rule that never stops early has no spread at all, so there is no band to
    # give it: it has to hit the horizon exactly.
    zscore <- if (se > 0) abs(got - published) / se else 0
    ok <- zscore <= BAND && (se > 0 || got == published)
    failures <<- failures + !ok
    cat(sprintf("  %-34s R %6.1f  published %6.0f  se %5.1f  %4.1f se  %s\n",
                label, got, published, se, zscore, if (ok) "ok" else "FAIL"))
}

rules <- list(
    "fixed horizon (test once)" = fixed_horizon,
    "peek daily, stop at p<0.05" = naive_peeking,
    "peek daily, mSPRT (always-valid)" = function(d) msprt(d, TRUE_EFFECT)
)

cat("type-I error under the null, against reports/peeking.csv\n")
for (name in names(rules)) {
    row <- which(peeking$rule == name)
    if (length(row) != 1) {
        cat(sprintf("  %-34s no such rule in peeking.csv  FAIL\n", name))
        failures <- failures + 1
        next
    }
    check_rate(name, mean(rules[[name]](null)$declared),
               peeking[["type-I error"]][row])
}

cat("\npower under a true effect of", TRUE_EFFECT, "\n")
for (name in names(rules)) {
    row <- which(peeking$rule == name)
    if (length(row) != 1) next
    check_rate(name, mean(rules[[name]](alt)$declared), peeking$power[row])
}

cat("\naverage sample size per arm at the moment of stopping\n")
for (name in names(rules)) {
    row <- which(peeking$rule == name)
    if (length(row) != 1) next
    check_n(name, rules[[name]](alt)$stop, alt$n,
            peeking[["avg n/arm at stop"]][row])
}

# The Pocock row is calibrated inside the Python rather than read off a table,
# so it is checked in verify/Pocock.java instead of here.

cat("\nmSPRT power against tau, against reports/msprt_tau.csv\n")
r_power <- numeric(nrow(tau_tab))
for (i in seq_len(nrow(tau_tab))) {
    tau <- tau_tab$tau[i]
    r_power[i] <- mean(msprt(alt, tau)$declared)
    check_rate(sprintf("tau = %.2f", tau), r_power[i], tau_tab$power[i])
}

# The claim the tau table exists to make: power peaks where tau matches the
# effect being looked for, and collapses when tau is set far below it. That is
# qualitative, so it does not depend on the band above.
best <- tau_tab$tau[which.max(r_power)]
cat(sprintf("\nR puts peak mSPRT power at tau = %.2f, the true effect is %.2f\n",
            best, TRUE_EFFECT))
if (abs(best - TRUE_EFFECT) > 1e-9) {
    cat("FAIL: R does not reproduce the tuning claim the tau table is built on\n")
    failures <- failures + 1
}
smallest <- which.min(tau_tab$tau)
if (r_power[smallest] > 0.05) {
    cat(sprintf("FAIL: at tau = %.2f R gets %.4f power, the README says the\n",
                tau_tab$tau[smallest], r_power[smallest]),
        "method essentially stops working there\n")
    failures <- failures + 1
}

if (failures > 0) {
    cat(sprintf("\n%d checks failed\n", failures))
    quit(status = 1)
}
cat("\nR reproduces every published rate within", BAND,
    "standard errors, from its own draws\n")
