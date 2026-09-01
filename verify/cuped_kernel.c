/* Recompute the CUPED variance-reduction column of reports/cuped.csv.
 *
 * The published table comes from src/abcausal/experiments/cuped_gain.py, which
 * uses numpy for the draws and pandas for the table. Nothing else in the repo
 * checks it. This is the same estimator written from scratch: its own PRNG, its
 * own normals, its own pooled theta, and no shared code with the Python.
 *
 * Protocol, copied from cuped_gain.py so the comparison is like for like:
 *   4000 replications, 2000 users per arm, true effect 0.10,
 *   X ~ N(0,1) per user, outcome noise = rho*X + sqrt(1-rho^2)*E,
 *   theta = cov(Y,X)/var(X) fitted once on the pooled 4000 users of a
 *   replication, then variance reduction = 1 - var(adjusted)/var(plain)
 *   across replications.
 *
 * The published number is one 4000-replication draw, so it carries Monte Carlo
 * error and cannot be required to match exactly. Rather than pick a tolerance,
 * this runs REPLICATES independent 4000-replication estimates and measures the
 * spread, then requires the published value to sit inside 4 sd of the mean of
 * those. The band is therefore whatever the estimator's own noise says it is.
 *
 * Only the per-replication means are needed, so nothing is stored per user.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_REPS 4000
#define N_PER_ARM 2000
#define TRUE_EFFECT 0.10
#define MAX_ROWS 32
#define REPLICATES 12
#define SIGMA 4.0

/* xorshift128+, seeded through splitmix64. */
static unsigned long long s0, s1;

static unsigned long long splitmix64(unsigned long long *x) {
    unsigned long long z = (*x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static void seed_rng(unsigned long long seed) {
    s0 = splitmix64(&seed);
    s1 = splitmix64(&seed);
}

static unsigned long long next_u64(void) {
    unsigned long long x = s0, y = s1;
    s0 = y;
    x ^= x << 23;
    s1 = x ^ y ^ (x >> 17) ^ (y >> 26);
    return s1 + y;
}

/* Uniform on (0,1), never exactly 0 so log() is safe. */
static double next_unif(void) {
    return ((double)(next_u64() >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}

/* Marsaglia polar method: two normals per accepted pair. */
static double spare;
static int have_spare = 0;

static double next_normal(void) {
    double u, v, sq, f;
    if (have_spare) { have_spare = 0; return spare; }
    do {
        u = 2.0 * next_unif() - 1.0;
        v = 2.0 * next_unif() - 1.0;
        sq = u * u + v * v;
    } while (sq >= 1.0 || sq == 0.0);
    f = sqrt(-2.0 * log(sq) / sq);
    spare = v * f;
    have_spare = 1;
    return u * f;
}

static double plain[N_REPS], adj[N_REPS];

/* One 4000-replication estimate of the variance reduction at this rho.
 * bias_out receives the mean error of the adjusted estimator. */
static double variance_reduction(double rho, double *bias_out) {
    double k = sqrt(1.0 - rho * rho);
    double mp = 0.0, ma = 0.0, vp = 0.0, va = 0.0;
    int r, i;

    for (r = 0; r < N_REPS; r++) {
        double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
        double xt = 0.0, xc = 0.0, yt = 0.0, yc = 0.0;
        double n = 2.0 * N_PER_ARM, theta, vx, cxy, dx;
        for (i = 0; i < N_PER_ARM; i++) {
            double x = next_normal();
            double y = TRUE_EFFECT + rho * x + k * next_normal();
            xt += x; yt += y;
            sx += x; sy += y; sxx += x * x; sxy += x * y;
        }
        for (i = 0; i < N_PER_ARM; i++) {
            double x = next_normal();
            double y = rho * x + k * next_normal();
            xc += x; yc += y;
            sx += x; sy += y; sxx += x * x; sxy += x * y;
        }
        xt /= N_PER_ARM; xc /= N_PER_ARM; yt /= N_PER_ARM; yc /= N_PER_ARM;
        /* Pooled theta with ddof=1, matching numpy's np.cov(..., ddof=1). */
        vx = (sxx - sx * sx / n) / (n - 1.0);
        cxy = (sxy - sx * sy / n) / (n - 1.0);
        theta = (vx == 0.0) ? 0.0 : cxy / vx;
        dx = xt - xc;
        /* The pooled centring constant cancels in the difference of arms. */
        plain[r] = yt - yc;
        adj[r] = plain[r] - theta * dx;
    }
    for (r = 0; r < N_REPS; r++) { mp += plain[r]; ma += adj[r]; }
    mp /= N_REPS; ma /= N_REPS;
    for (r = 0; r < N_REPS; r++) {
        vp += (plain[r] - mp) * (plain[r] - mp);
        va += (adj[r] - ma) * (adj[r] - ma);
    }
    *bias_out = ma - TRUE_EFFECT;
    return 1.0 - va / vp;
}

/* Minimal CSV field reader: returns the index of a named column, or -1. */
static int column_index(const char *header, const char *name) {
    char buf[4096];
    char *p = buf;
    int idx = 0;
    char sep;
    size_t len = strlen(header);
    if (len >= sizeof buf) return -1;
    memcpy(buf, header, len + 1);
    while (*p) {
        char *start;
        char *end;
        if (*p == '"') {
            start = ++p;
            while (*p && *p != '"') p++;
            end = p;
            if (*p == '"') p++;
        } else {
            start = p;
            while (*p && *p != ',') p++;
            end = p;
        }
        sep = *p;
        while (end > start && (end[-1] == '\n' || end[-1] == '\r')) end--;
        *end = '\0';
        if (strcmp(start, name) == 0) return idx;
        if (sep == ',') p++;
        idx++;
    }
    return -1;
}

/* Field `want` of a CSV line, as a double. Returns 0 on failure. */
static int field_double(const char *line, int want, double *out) {
    char buf[4096];
    char *p = buf;
    int idx = 0;
    char sep;
    size_t len = strlen(line);
    if (len >= sizeof buf) return 0;
    memcpy(buf, line, len + 1);
    while (*p) {
        char *start;
        char *end;
        if (*p == '"') {
            start = ++p;
            while (*p && *p != '"') p++;
            end = p;
            if (*p == '"') p++;
        } else {
            start = p;
            while (*p && *p != ',') p++;
            end = p;
        }
        sep = *p;
        while (end > start && (end[-1] == '\n' || end[-1] == '\r')) end--;
        *end = '\0';
        if (idx == want) {
            char *stop;
            *out = strtod(start, &stop);
            return stop != start;
        }
        if (sep == ',') p++;
        idx++;
    }
    return 0;
}

int main(int argc, char **argv) {
    const char *root = (argc > 1) ? argv[1] : ".";
    char path[4096];
    char line[4096];
    FILE *f;
    double rho[MAX_ROWS], pub[MAX_ROWS], pred[MAX_ROWS], pubbias[MAX_ROWS];
    int n = 0, i, j, bad = 0;
    int c_rho, c_vr, c_pred, c_bias;

    snprintf(path, sizeof path, "%s/reports/cuped.csv", root);
    f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 1; }
    if (!fgets(line, sizeof line, f)) {
        fprintf(stderr, "empty %s\n", path); fclose(f); return 1;
    }
    c_rho = column_index(line, "corr(X, Y)");
    c_vr = column_index(line, "variance reduction");
    c_pred = column_index(line, "predicted (rho^2)");
    c_bias = column_index(line, "bias (CUPED)");
    if (c_rho < 0 || c_vr < 0 || c_pred < 0 || c_bias < 0) {
        fprintf(stderr, "cuped.csv is missing a column this check needs\n");
        fclose(f); return 1;
    }
    while (fgets(line, sizeof line, f) && n < MAX_ROWS) {
        if (line[0] == '\n' || line[0] == '\r') continue;
        if (!field_double(line, c_rho, &rho[n]) ||
            !field_double(line, c_vr, &pub[n]) ||
            !field_double(line, c_pred, &pred[n]) ||
            !field_double(line, c_bias, &pubbias[n])) {
            fprintf(stderr, "unparseable row in cuped.csv: %s", line);
            fclose(f); return 1;
        }
        if (rho[n] < 0.0 || rho[n] > 1.0) {
            fprintf(stderr, "cuped.csv row %d has corr(X, Y) = %g, "
                            "which is not a correlation\n", n + 2, rho[n]);
            fclose(f); return 1;
        }
        n++;
    }
    fclose(f);
    if (n == 0) { fprintf(stderr, "no rows in cuped.csv\n"); return 1; }

    printf("C, CUPED variance reduction: %d independent estimates per row,\n"
           "%d replications of %d per arm each, own PRNG\n\n",
           REPLICATES, N_REPS, N_PER_ARM);
    printf("  rho  published  recomputed  sd(%d runs)  |d|/sd   rho^2   bias\n",
           REPLICATES);
    for (i = 0; i < n; i++) {
        double v[REPLICATES], b[REPLICATES];
        double mean = 0.0, sd = 0.0, bias = 0.0, z, se;
        for (j = 0; j < REPLICATES; j++) {
            seed_rng(0x5EEDC0DEULL + (unsigned long long)(i * 7919 + j * 104729));
            have_spare = 0;
            v[j] = variance_reduction(rho[i], &b[j]);
            mean += v[j];
            bias += b[j];
        }
        mean /= REPLICATES;
        bias /= REPLICATES;
        for (j = 0; j < REPLICATES; j++) sd += (v[j] - mean) * (v[j] - mean);
        sd = sqrt(sd / (REPLICATES - 1));
        if (sd < 1e-12) sd = 1e-12;

        z = fabs(pub[i] - mean) / sd;
        printf("  %.1f  %9.4f  %10.4f  %11.4f  %6.2f  %6.4f  %+.5f%s\n",
               rho[i], pub[i], mean, sd, z, rho[i] * rho[i], bias,
               z > SIGMA ? "  FAIL" : "");
        if (z > SIGMA) bad++;

        /* The theoretical claim the README makes: reduction equals rho^2. The
         * mean of REPLICATES runs has standard error sd/sqrt(REPLICATES), so
         * that is the band it gets, not the wider single-run one. */
        se = sd / sqrt((double)REPLICATES);
        if (fabs(mean - rho[i] * rho[i]) > SIGMA * se) {
            printf("        recomputed %.4f is %.1f se from the theoretical "
                   "%.4f  FAIL\n", mean, fabs(mean - rho[i] * rho[i]) / se,
                   rho[i] * rho[i]);
            bad++;
        }
        /* Deterministic: the published predicted column must be rho^2, and the
         * CUPED estimator must be unbiased. Neither is a Monte Carlo claim. */
        if (fabs(pred[i] - rho[i] * rho[i]) > 5e-5) {
            printf("        published predicted column %.4f is not rho^2 "
                   "%.4f  FAIL\n", pred[i], rho[i] * rho[i]);
            bad++;
        }
        if (fabs(pubbias[i]) > 0.005) {
            printf("        published CUPED bias %.5f is not zero  FAIL\n",
                   pubbias[i]);
            bad++;
        }
    }
    if (bad) {
        printf("\n%d disagreement(s) beyond %.0f sd\n", bad, SIGMA);
        return 1;
    }
    printf("\nall %d rows land within %.0f sd of an independent re-simulation,\n"
           "and the recomputed reduction is rho^2 in every row\n", n, SIGMA);
    return 0;
}
