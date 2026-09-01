// Recalibrate the Pocock boundary, in Java, and rescore the row it produced.
//
// The Pocock row of reports/peeking.csv is the only one whose threshold is not
// a constant. src/abcausal/sequential.py finds it by simulating under the null
// and taking the 95th percentile of each replication's maximum |z|, then
// src/abcausal/experiments/peeking.py applies it to a different seed. So there
// are two things to be wrong about, the quantile and the rule, and the same
// numpy harness produces both. Nothing else here recalibrates it: verify.R and
// verify/peekmc deliberately leave this row alone so that this is an
// independent second opinion rather than a third copy of the same one.
//
// Written to the single-file source launcher, so CI needs `java` and no build
// step: java verify/Pocock.java <repo root>

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public final class Pocock {

    static final int N_REPS = 20_000;      // the count the published run used
    static final int CALIBRATIONS = 8;     // independent calibrations, for a spread
    static final int HORIZON = 14;
    static final int N_PER_DAY = 100;
    static final double TRUE_EFFECT = 0.10;
    static final double ALPHA = 0.05;
    static final double BAND = 4.0;        // standard errors

    // xorshift128+ seeded through splitmix64. Its own generator, so a mistake
    // in numpy's draws cannot be reproduced here by accident.
    static final class Rng {
        private long s0, s1;
        private double spare;
        private boolean hasSpare;

        Rng(long seed) {
            s0 = splitmix(seed);
            s1 = splitmix(s0);
        }

        private static long splitmix(long x) {
            x += 0x9E3779B97F4A7C15L;
            long z = x;
            z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
            z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
            return z ^ (z >>> 31);
        }

        long nextLong() {
            long x = s0, y = s1;
            s0 = y;
            x ^= x << 23;
            s1 = x ^ y ^ (x >>> 17) ^ (y >>> 26);
            return s1 + y;
        }

        double unif() {
            return ((double) (nextLong() >>> 11) + 0.5) * (1.0 / 9007199254740992.0);
        }

        // Marsaglia polar method: two normals per accepted pair.
        double normal() {
            if (hasSpare) { hasSpare = false; return spare; }
            while (true) {
                double u = 2.0 * unif() - 1.0;
                double v = 2.0 * unif() - 1.0;
                double sq = u * u + v * v;
                if (sq < 1.0 && sq > 0.0) {
                    double f = Math.sqrt(-2.0 * Math.log(sq) / sq);
                    spare = v * f;
                    hasSpare = true;
                    return u * f;
                }
            }
        }
    }

    // What one simulated experiment leaves behind. The alternative is the null
    // shifted: adding a constant to every treatment observation moves that
    // arm's mean by exactly that constant and leaves its sample variance
    // unchanged, so z under the alternative is z under the null plus
    // effect/se, exactly. One set of draws therefore serves both.
    static final class Looks {
        final double[] zNull = new double[HORIZON];
        final double[] zAlt = new double[HORIZON];
    }

    static void simulate(Rng rng, Looks out) {
        double cs = 0, css = 0, ts = 0, tss = 0;
        for (int day = 1; day <= HORIZON; day++) {
            for (int i = 0; i < N_PER_DAY; i++) {
                double c = rng.normal();
                double t = rng.normal();
                cs += c; css += c * c;
                ts += t; tss += t * t;
            }
            double n = day * N_PER_DAY;
            double cMean = cs / n, tMean = ts / n;
            double cVar = (css - n * cMean * cMean) / (n - 1);  // ddof = 1
            double tVar = (tss - n * tMean * tMean) / (n - 1);
            double se = Math.sqrt(cVar / n + tVar / n);
            out.zNull[day - 1] = (tMean - cMean) / se;
            out.zAlt[day - 1] = (tMean - cMean + TRUE_EFFECT) / se;
        }
    }

    // Linear interpolation between order statistics: numpy's default and R's
    // type 7. Matching the convention matters, the alternatives shift the
    // answer by up to one order statistic.
    static double quantile(double[] sorted, double q) {
        double pos = q * (sorted.length - 1);
        int lo = (int) Math.floor(pos), hi = (int) Math.ceil(pos);
        if (lo == hi) return sorted[lo];
        return sorted[lo] + (pos - lo) * (sorted[hi] - sorted[lo]);
    }

    // The constant boundary whose family-wise error over HORIZON looks is
    // alpha: the (1-alpha) quantile of the per-replication maximum |z|.
    static double calibrate(long seed) {
        Rng rng = new Rng(seed);
        Looks looks = new Looks();
        double[] maxAbs = new double[N_REPS];
        for (int r = 0; r < N_REPS; r++) {
            simulate(rng, looks);
            double m = 0;
            for (double z : looks.zNull) m = Math.max(m, Math.abs(z));
            maxAbs[r] = m;
        }
        Arrays.sort(maxAbs);
        return quantile(maxAbs, 1 - ALPHA);
    }

    static final class Scored {
        double typeI, power, avgN, avgNsd;
    }

    // Apply a constant boundary at every look, on a seed the boundary was not
    // calibrated on. Replications that never cross run to the horizon.
    static Scored score(long seed, double crit) {
        Rng rng = new Rng(seed);
        Looks looks = new Looks();
        int declaredNull = 0, declaredAlt = 0;
        double sumN = 0, sumNsq = 0;
        for (int r = 0; r < N_REPS; r++) {
            simulate(rng, looks);
            boolean anyNull = false, anyAlt = false;
            int stopAlt = HORIZON;
            for (int d = 0; d < HORIZON; d++) {
                if (!anyNull && Math.abs(looks.zNull[d]) > crit) anyNull = true;
                if (!anyAlt && Math.abs(looks.zAlt[d]) > crit) {
                    anyAlt = true;
                    stopAlt = d + 1;
                }
            }
            if (anyNull) declaredNull++;
            if (anyAlt) declaredAlt++;
            double n = stopAlt * N_PER_DAY;
            sumN += n;
            sumNsq += n * n;
        }
        Scored s = new Scored();
        s.typeI = (double) declaredNull / N_REPS;
        s.power = (double) declaredAlt / N_REPS;
        s.avgN = sumN / N_REPS;
        s.avgNsd = Math.sqrt((sumNsq - N_REPS * s.avgN * s.avgN) / (N_REPS - 1));
        return s;
    }

    static int failures = 0;

    static void checkRate(String label, double got, double published) {
        // Two independent Monte Carlo estimates of the same rate, both on
        // N_REPS draws, differ with this standard error.
        double se = Math.sqrt(2 * Math.max(got * (1 - got), 1e-8) / N_REPS);
        double z = Math.abs(got - published) / se;
        boolean ok = z <= BAND;
        if (!ok) failures++;
        System.out.printf("  %-22s Java %8.4f  published %8.4f  se %.4f  %4.1f se  %s%n",
                label, got, published, se, z, ok ? "ok" : "FAIL");
    }

    static double[] readPeekingRow(Path csv) throws IOException {
        List<String> lines = Files.readAllLines(csv);
        if (lines.size() < 2) throw new IOException(csv + " has no rows");
        String[] header = splitCsv(lines.get(0));
        int cRule = indexOf(header, "rule");
        int cErr = indexOf(header, "type-I error");
        int cPow = indexOf(header, "power");
        int cN = indexOf(header, "avg n/arm at stop");
        for (String line : lines.subList(1, lines.size())) {
            if (line.isBlank()) continue;
            String[] f = splitCsv(line);
            if (f[cRule].startsWith("peek daily, Pocock")) {
                return new double[] {
                    Double.parseDouble(f[cErr].trim()),
                    Double.parseDouble(f[cPow].trim()),
                    Double.parseDouble(f[cN].trim()),
                };
            }
        }
        throw new IOException(csv + " has no Pocock row");
    }

    static int indexOf(String[] header, String name) throws IOException {
        for (int i = 0; i < header.length; i++) if (header[i].equals(name)) return i;
        throw new IOException("no column named " + name);
    }

    // Enough CSV to read a row whose fields may be quoted because they contain
    // a comma, which the rule names do.
    static String[] splitCsv(String line) {
        java.util.List<String> out = new java.util.ArrayList<>();
        StringBuilder cur = new StringBuilder();
        boolean quoted = false;
        for (char ch : line.toCharArray()) {
            if (ch == '"') quoted = !quoted;
            else if (ch == ',' && !quoted) { out.add(cur.toString()); cur.setLength(0); }
            else if (ch != '\r') cur.append(ch);
        }
        out.add(cur.toString());
        return out.toArray(new String[0]);
    }

    // notes/METHODS.md quotes the calibrated boundary in prose. Nothing else
    // reads it, so nothing else would notice if the code moved and the sentence
    // did not.
    static Double readPublishedBoundary(Path methods) throws IOException {
        String text = Files.readString(methods);
        Matcher m = Pattern.compile("\\|z\\|\\s*>\\s*([0-9]+\\.[0-9]+)\\s+instead of")
                .matcher(text);
        return m.find() ? Double.parseDouble(m.group(1)) : null;
    }

    public static void main(String[] args) throws IOException {
        String root = args.length > 0 ? args[0] : ".";
        double[] pub = readPeekingRow(Path.of(root, "reports", "peeking.csv"));
        Double pubCrit = readPublishedBoundary(Path.of(root, "notes", "METHODS.md"));

        System.out.printf(
            "Java, Pocock boundary recalibrated: %d independent calibrations of%n"
            + "%d replications each, then scored on a seed none of them saw%n%n",
            CALIBRATIONS, N_REPS);

        double[] crits = new double[CALIBRATIONS];
        for (int i = 0; i < CALIBRATIONS; i++) {
            crits[i] = calibrate(0x9E3779B9L + i * 104729L);
        }
        double mean = Arrays.stream(crits).average().orElse(0);
        double sd = Math.sqrt(Arrays.stream(crits).map(c -> (c - mean) * (c - mean)).sum()
                / (CALIBRATIONS - 1));

        System.out.printf("  boundary               Java %8.4f  sd %.4f over %d calibrations%n",
                mean, sd, CALIBRATIONS);
        if (pubCrit == null) {
            System.out.println("  FAIL notes/METHODS.md no longer quotes the boundary");
            failures++;
        } else {
            // METHODS.md rounds to two decimals, so the band has to allow that
            // as well as the Monte Carlo spread of a single calibration.
            double tol = BAND * sd + 0.005;
            boolean ok = Math.abs(mean - pubCrit) <= tol;
            if (!ok) failures++;
            System.out.printf("  %-22s Java %8.4f  METHODS.md %6.2f  tol %.4f  %s%n",
                    "quoted in prose", mean, pubCrit, tol, ok ? "ok" : "FAIL");
        }
        if (mean <= 1.96) {
            System.out.println("  FAIL a Pocock boundary below 1.96 would not correct anything");
            failures++;
        }

        Scored s = score(0xBEEFCAFEL, mean);
        System.out.println();
        checkRate("type-I error", s.typeI, pub[0]);
        checkRate("power", s.power, pub[1]);

        double seN = Math.sqrt(2) * s.avgNsd / Math.sqrt(N_REPS);
        double zN = Math.abs(s.avgN - pub[2]) / seN;
        boolean okN = zN <= BAND;
        if (!okN) failures++;
        System.out.printf("  %-22s Java %8.1f  published %8.0f  se %.1f  %4.1f se  %s%n",
                "avg n/arm at stop", s.avgN, pub[2], seN, zN, okN ? "ok" : "FAIL");

        // The claim the row exists to make: a boundary calibrated on one seed
        // restores control on another. That is qualitative and does not depend
        // on the band above.
        if (s.typeI < 0.04 || s.typeI > 0.06) {
            System.out.printf("  FAIL a boundary calibrated for 5%% errs %.4f "
                    + "on a fresh seed, so it is not calibrated%n", s.typeI);
            failures++;
        }

        if (failures > 0) {
            System.out.printf("%n%d checks failed%n", failures);
            System.exit(1);
        }
        System.out.printf("%nJava recalibrates the boundary to %.4f and reproduces the "
                + "Pocock row%nfrom its own draws%n", mean);
    }
}
