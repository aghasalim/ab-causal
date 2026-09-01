//! How much of the published 22.3% is Monte Carlo noise?
//!
//! `src/abcausal/experiments/peeking.py` runs 20,000 replications and reports
//! type-I error to four decimal places. That is a Monte Carlo estimate, so it
//! carries its own error, and nothing in the repository measured it. The README
//! quotes the third digit of it. This does two things the Python run cannot
//! afford:
//!
//!   1. a REFERENCE-replication run, accurate enough to treat as the truth
//!   2. REPLICATES independent runs at the published 20,000, whose spread is
//!      the error bar on the published number
//!
//! Then it checks every published rate sits inside that error bar. A pass means
//! 20,000 replications was enough for the claim being made from it; a failure
//! would mean the README is quoting noise.
//!
//! The simulation is written from scratch: its own generator, its own running
//! mean and variance, its own decision rules, no shared code with the Python.

use std::env;
use std::fs;
use std::process::exit;

const REFERENCE: usize = 200_000;
const PUBLISHED_REPS: usize = 20_000;
const REPLICATES: usize = 25;
const HORIZON: usize = 14;
const N_PER_DAY: usize = 100;
const TRUE_EFFECT: f64 = 0.10;
const ALPHA: f64 = 0.05;
const CRIT: f64 = 1.959_963_984_540_054; // qnorm(0.975)
const SIGMA: f64 = 4.0; // how far a published rate may sit from the reference

/// xorshift128+, seeded through splitmix64. Not cryptographic and not meant to
/// be: it needs to be uniform, fast, and seeded reproducibly so a failure here
/// can be re-run.
struct Rng {
    s0: u64,
    s1: u64,
    spare: f64,
    has_spare: bool,
}

impl Rng {
    fn new(seed: u64) -> Self {
        let mut x = seed;
        let mut split = || {
            x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        Rng { s0: split(), s1: split(), spare: 0.0, has_spare: false }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.s0;
        let y = self.s1;
        self.s0 = y;
        x ^= x << 23;
        self.s1 = x ^ y ^ (x >> 17) ^ (y >> 26);
        self.s1.wrapping_add(y)
    }

    /// Uniform on (0, 1), never exactly 0 so ln() is safe.
    fn unif(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) * (1.0 / 9_007_199_254_740_992.0)
    }

    /// Marsaglia polar method: two normals per accepted pair.
    fn normal(&mut self) -> f64 {
        if self.has_spare {
            self.has_spare = false;
            return self.spare;
        }
        loop {
            let u = 2.0 * self.unif() - 1.0;
            let v = 2.0 * self.unif() - 1.0;
            let sq = u * u + v * v;
            if sq < 1.0 && sq > 0.0 {
                let f = (-2.0 * sq.ln() / sq).sqrt();
                self.spare = v * f;
                self.has_spare = true;
                return u * f;
            }
        }
    }
}

#[derive(Default, Clone, Copy)]
struct Tally {
    declared: u64,
    stop_sum: f64,
    stop_sumsq: f64,
}

impl Tally {
    fn add(&mut self, declared: bool, n_at_stop: f64) {
        self.declared += declared as u64;
        self.stop_sum += n_at_stop;
        self.stop_sumsq += n_at_stop * n_at_stop;
    }
    fn rate(&self, reps: usize) -> f64 {
        self.declared as f64 / reps as f64
    }
    fn mean_n(&self, reps: usize) -> f64 {
        self.stop_sum / reps as f64
    }
}

/// Everything one run measures: three rules, under the null and under a real
/// effect.
#[derive(Default, Clone, Copy)]
struct Run {
    null: [Tally; 3],
    alt: [Tally; 3],
}

const RULES: [&str; 3] = [
    "fixed horizon (test once)",
    "peek daily, stop at p<0.05",
    "peek daily, mSPRT (always-valid)",
];

/// One run of `reps` simulated experiments, observed once a day for HORIZON
/// days at N_PER_DAY users per arm per day.
///
/// The null and the alternative come out of the same draws. Adding a constant
/// to every treatment observation shifts the arm's mean by exactly that
/// constant and leaves its sample variance algebraically unchanged, so the
/// alternative's z is the null's z plus effect/se, exactly. That is the same
/// pairing the Python harness uses between rules, applied one level up, and it
/// halves the number of normals this has to draw.
fn run(reps: usize, seed: u64) -> Run {
    let mut out = Run::default();
    let mut rng = Rng::new(seed);

    for _ in 0..reps {
        let (mut cs, mut css, mut ts, mut tss) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);

        // Running state for the sequential rules.
        let mut naive_stop: Option<usize> = None;
        let mut msprt_stop: Option<usize> = None;
        let mut naive_stop_alt: Option<usize> = None;
        let mut msprt_stop_alt: Option<usize> = None;
        let mut pmin = 1.0f64;
        let mut pmin_alt = 1.0f64;
        let mut z_last = 0.0;
        let mut z_last_alt = 0.0;

        for day in 1..=HORIZON {
            for _ in 0..N_PER_DAY {
                let c = rng.normal();
                let t = rng.normal();
                cs += c;
                css += c * c;
                ts += t;
                tss += t * t;
            }
            let n = (day * N_PER_DAY) as f64;
            let c_mean = cs / n;
            let t_mean = ts / n;
            // Unbiased sample variance from running sums, ddof = 1.
            let c_var = (css - n * c_mean * c_mean) / (n - 1.0);
            let t_var = (tss - n * t_mean * t_mean) / (n - 1.0);
            let se = (c_var / n + t_var / n).sqrt();

            let diff = t_mean - c_mean;
            let diff_alt = diff + TRUE_EFFECT;
            let z = diff / se;
            let z_alt = diff_alt / se;
            z_last = z;
            z_last_alt = z_alt;

            if naive_stop.is_none() && z.abs() > CRIT {
                naive_stop = Some(day);
            }
            if naive_stop_alt.is_none() && z_alt.abs() > CRIT {
                naive_stop_alt = Some(day);
            }

            // Mixture SPRT against a N(0, tau^2) alternative, tau tuned to the
            // effect being looked for, as the published run does.
            let v = 2.0 * 1.0 / n;
            let tau2 = TRUE_EFFECT * TRUE_EFFECT;
            let scale = (v / (v + tau2)).sqrt();
            let lr = scale * (diff * diff * tau2 / (2.0 * v * (v + tau2))).exp();
            let lr_alt = scale * (diff_alt * diff_alt * tau2 / (2.0 * v * (v + tau2))).exp();
            pmin = pmin.min((1.0 / lr).min(1.0));
            pmin_alt = pmin_alt.min((1.0 / lr_alt).min(1.0));
            if msprt_stop.is_none() && pmin < ALPHA {
                msprt_stop = Some(day);
            }
            if msprt_stop_alt.is_none() && pmin_alt < ALPHA {
                msprt_stop_alt = Some(day);
            }
        }

        let horizon_n = (HORIZON * N_PER_DAY) as f64;
        let at = |s: Option<usize>| match s {
            Some(d) => (d * N_PER_DAY) as f64,
            None => horizon_n,
        };

        out.null[0].add(z_last.abs() > CRIT, horizon_n);
        out.null[1].add(naive_stop.is_some(), at(naive_stop));
        out.null[2].add(msprt_stop.is_some(), at(msprt_stop));

        out.alt[0].add(z_last_alt.abs() > CRIT, horizon_n);
        out.alt[1].add(naive_stop_alt.is_some(), at(naive_stop_alt));
        out.alt[2].add(msprt_stop_alt.is_some(), at(msprt_stop_alt));
    }
    out
}

struct Published {
    rule: String,
    type_i: f64,
    power: f64,
    avg_n: f64,
}

fn load(root: &str) -> Vec<Published> {
    let path = format!("{}/reports/peeking.csv", root);
    let text = fs::read_to_string(&path).unwrap_or_else(|e| {
        eprintln!("cannot read {}: {}", path, e);
        exit(2)
    });
    let mut lines = text.lines();
    let header = split_csv(lines.next().unwrap_or_else(|| {
        eprintln!("{} is empty", path);
        exit(2)
    }));
    let col = |name: &str| {
        header.iter().position(|h| h == name).unwrap_or_else(|| {
            eprintln!("{} has no {:?} column", path, name);
            exit(2)
        })
    };
    let (r, e, p, n) = (
        col("rule"),
        col("type-I error"),
        col("power"),
        col("avg n/arm at stop"),
    );
    let mut out = Vec::new();
    for line in lines.filter(|l| !l.trim().is_empty()) {
        let f = split_csv(line);
        if f.len() <= n {
            eprintln!("short row in {}: {}", path, line);
            exit(2);
        }
        let num = |i: usize| {
            f[i].trim().parse::<f64>().unwrap_or_else(|_| {
                eprintln!("{} has an unparseable number: {:?}", path, f[i]);
                exit(2)
            })
        };
        out.push(Published {
            rule: f[r].clone(),
            type_i: num(e),
            power: num(p),
            avg_n: num(n),
        });
    }
    out
}

/// Enough CSV to read a header and a row whose fields may be quoted because
/// they contain a comma, which the rule names do.
fn split_csv(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut quoted = false;
    for ch in line.chars() {
        match ch {
            '"' => quoted = !quoted,
            ',' if !quoted => out.push(std::mem::take(&mut cur)),
            '\r' | '\n' => {}
            _ => cur.push(ch),
        }
    }
    out.push(cur);
    out
}

fn sd(v: &[f64]) -> f64 {
    let m = v.iter().sum::<f64>() / v.len() as f64;
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let root = args.get(1).map(String::as_str).unwrap_or(".");
    let published = load(root);

    println!(
        "Rust, {} replications as a reference, error bar from {} runs of {}\n\
         (the published run is {} replications)\n",
        REFERENCE, REPLICATES, PUBLISHED_REPS, PUBLISHED_REPS
    );

    let reference = run(REFERENCE, 0x5EED_0000_0000_0001);
    let reps: Vec<Run> = (0..REPLICATES)
        .map(|r| run(PUBLISHED_REPS, 0xC0FF_EE00 + r as u64 * 104_729))
        .collect();

    let mut failures = 0;
    for (i, name) in RULES.iter().enumerate() {
        let Some(pubrow) = published.iter().find(|p| p.rule == *name) else {
            println!("{:<34} not in reports/peeking.csv  FAIL", name);
            failures += 1;
            continue;
        };
        println!("{}", name);

        let cases: [(&str, f64, f64, Vec<f64>); 3] = [
            (
                "type-I error",
                pubrow.type_i,
                reference.null[i].rate(REFERENCE),
                reps.iter().map(|r| r.null[i].rate(PUBLISHED_REPS)).collect(),
            ),
            (
                "power",
                pubrow.power,
                reference.alt[i].rate(REFERENCE),
                reps.iter().map(|r| r.alt[i].rate(PUBLISHED_REPS)).collect(),
            ),
            (
                "avg n/arm at stop",
                pubrow.avg_n,
                reference.alt[i].mean_n(REFERENCE),
                reps.iter().map(|r| r.alt[i].mean_n(PUBLISHED_REPS)).collect(),
            ),
        ];

        for (label, pubval, refval, spread) in cases {
            let s = sd(&spread);
            // A rule that never stops early has no spread, so it gets no band:
            // it has to hit the horizon exactly.
            let (z, ok) = if s > 0.0 {
                let z = (pubval - refval).abs() / s;
                (z, z <= SIGMA)
            } else {
                (0.0, pubval == refval)
            };
            failures += !ok as i32;
            println!(
                "  {:<20} published {:>9.4}  reference {:>9.4}  \
                 sd at {} {:>8.4}  {:>4.1} sd  {}",
                label,
                pubval,
                refval,
                PUBLISHED_REPS,
                s,
                z,
                if ok { "ok" } else { "FAIL" }
            );
        }
        println!();
    }

    if failures > 0 {
        println!(
            "{} published values sit further than {} sd from a {} replication reference",
            failures, SIGMA, REFERENCE
        );
        exit(1);
    }
    println!(
        "every published rate is within {} sd of a {} replication reference,\n\
         so {} replications was enough for the claims made from them",
        SIGMA, REFERENCE, PUBLISHED_REPS
    );
}
