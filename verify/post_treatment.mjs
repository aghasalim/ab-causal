// Re-simulate the post-treatment-bias table, in Node, with no dependencies.
//
// reports/cuped_post_treatment_bias.csv carries the failure this repository
// most wants believed: CUPED on a covariate that treatment moved reports a zero
// effect on a feature that works, and the standard error does not widen to warn
// you. Both halves of that come out of one numpy run in
// src/abcausal/experiments/cuped_gain.py.
//
// This redraws the table from scratch. Its own generator, its own pooled theta,
// its own variance arithmetic. It also checks something the Python never did:
// that the standard error the table reports is the estimator's actual sampling
// spread, and not merely a formula that happens to stay flat.
//
// Run: node verify/post_treatment.mjs <repo root>

import { readFileSync } from "node:fs";
import { join } from "node:path";

const N_REPS = 1500; // enough for a band four times tighter than the rounding
const N_PER_ARM = 2000;
const TRUE_EFFECT = 0.1;
const BETA = 0.8; // how strongly the covariate drives the outcome
const NOISE = 0.6; // residual scale, from cuped_gain.py
const BAND = 4;

// xorshift128+ seeded through splitmix64, and Marsaglia's polar method for the
// normals. Small enough to read, which matters more here than being a good
// generator.
function makeRng(seed) {
  let s0, s1;
  let x = BigInt.asUintN(64, BigInt(seed));
  const splitmix = () => {
    x = BigInt.asUintN(64, x + 0x9e3779b97f4a7c15n);
    let z = x;
    z = BigInt.asUintN(64, (z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n);
    z = BigInt.asUintN(64, (z ^ (z >> 27n)) * 0x94d049bb133111ebn);
    return z ^ (z >> 31n);
  };
  s0 = splitmix();
  s1 = splitmix();

  const nextU64 = () => {
    let a = s0;
    const b = s1;
    s0 = b;
    a = BigInt.asUintN(64, a ^ (a << 23n));
    s1 = BigInt.asUintN(64, a ^ b ^ (a >> 17n) ^ (b >> 26n));
    return BigInt.asUintN(64, s1 + b);
  };

  let spare = 0;
  let hasSpare = false;
  return function normal() {
    if (hasSpare) {
      hasSpare = false;
      return spare;
    }
    for (;;) {
      const u = 2 * (Number(nextU64() >> 11n) + 0.5) / 9007199254740992 - 1;
      const v = 2 * (Number(nextU64() >> 11n) + 0.5) / 9007199254740992 - 1;
      const sq = u * u + v * v;
      if (sq < 1 && sq > 0) {
        const f = Math.sqrt((-2 * Math.log(sq)) / sq);
        spare = v * f;
        hasSpare = true;
        return u * f;
      }
    }
  };
}

// One arm's running sums. Nothing is stored per user.
function arm() {
  return { sx: 0, sy: 0, sxx: 0, syy: 0, sxy: 0, n: 0 };
}
function push(a, x, y) {
  a.sx += x; a.sy += y; a.sxx += x * x; a.syy += y * y; a.sxy += x * y; a.n += 1;
}
function moments(a) {
  const n = a.n;
  return {
    mx: a.sx / n,
    my: a.sy / n,
    vx: (a.sxx - (a.sx * a.sx) / n) / (n - 1),
    vy: (a.syy - (a.sy * a.sy) / n) / (n - 1),
    cxy: (a.sxy - (a.sx * a.sy) / n) / (n - 1),
  };
}

// Causal structure: treatment -> X -> Y plus a direct path, with the total
// effect held at TRUE_EFFECT. `mediated` is the share flowing through X.
function simulate(mediated, seed) {
  const normal = makeRng(seed);
  const shift = (mediated * TRUE_EFFECT) / BETA;
  const direct = (1 - mediated) * TRUE_EFFECT;

  let plainSum = 0, plainSq = 0, adjSum = 0, adjSq = 0, seSum = 0;

  for (let r = 0; r < N_REPS; r++) {
    const t = arm(), c = arm();
    for (let i = 0; i < N_PER_ARM; i++) {
      const xc = normal();
      push(c, xc, BETA * xc + normal() * NOISE);
      const xt = normal() + shift;
      push(t, xt, direct + BETA * xt + normal() * NOISE);
    }
    const mt = moments(t), mc = moments(c);

    // Pooled theta over both arms, ddof = 1, exactly as cuped.py fits it.
    const n = t.n + c.n;
    const sx = t.sx + c.sx, sy = t.sy + c.sy;
    const vxPooled = (t.sxx + c.sxx - (sx * sx) / n) / (n - 1);
    const cxyPooled = (t.sxy + c.sxy - (sx * sy) / n) / (n - 1);
    const theta = vxPooled === 0 ? 0 : cxyPooled / vxPooled;

    const plain = mt.my - mc.my;
    // The pooled centring constant cancels in the difference of arms.
    const adj = plain - theta * (mt.mx - mc.mx);
    // Variance of y - theta*x within an arm, from that arm's own moments.
    const vAdj = (m) => m.vy - 2 * theta * m.cxy + theta * theta * m.vx;
    const se = Math.sqrt(vAdj(mt) / t.n + vAdj(mc) / c.n);

    plainSum += plain;
    plainSq += plain * plain;
    adjSum += adj;
    adjSq += adj * adj;
    seSum += se;
  }

  const adjMean = adjSum / N_REPS;
  const plainMean = plainSum / N_REPS;
  return {
    plain: plainMean,
    // The unadjusted estimator is the noisier of the two, so it needs its own
    // spread rather than borrowing the adjusted one's.
    plainSd: Math.sqrt((plainSq - N_REPS * plainMean * plainMean) / (N_REPS - 1)),
    adj: adjMean,
    // The estimator's actual spread across replications.
    empiricalSd: Math.sqrt((adjSq - N_REPS * adjMean * adjMean) / (N_REPS - 1)),
    // The mean of the standard errors the estimator reported for itself.
    reportedSe: seSum / N_REPS,
    direct,
  };
}

function splitCsv(line) {
  const out = [];
  let cur = "";
  let quoted = false;
  for (const ch of line) {
    if (ch === '"') quoted = !quoted;
    else if (ch === "," && !quoted) { out.push(cur); cur = ""; }
    else if (ch !== "\r") cur += ch;
  }
  out.push(cur);
  return out;
}

const root = process.argv[2] ?? ".";
const path = join(root, "reports", "cuped_post_treatment_bias.csv");
const lines = readFileSync(path, "utf8").split("\n").filter((l) => l.trim() !== "");
const header = splitCsv(lines[0]);
const idx = (name) => {
  const i = header.indexOf(name);
  if (i < 0) {
    console.error(`${path} has no ${JSON.stringify(name)} column`);
    process.exit(2);
  }
  return i;
};
const cShare = idx("% of effect via covariate");
const cPlain = idx("plain diff-in-means");
const cEst = idx("CUPED estimate");
const cSe = idx("CUPED SE (unchanged)");

let failures = 0;
const check = (label, got, published, se) => {
  const z = se > 0 ? Math.abs(got - published) / se : 0;
  const ok = z <= BAND;
  if (!ok) failures++;
  console.log(
    `  ${label.padEnd(24)} node ${got.toFixed(4)}  published ${published.toFixed(4)}` +
      `  se ${se.toFixed(4)}  ${z.toFixed(1).padStart(4)} se  ${ok ? "ok" : "FAIL"}`,
  );
};

console.log(
  `node, post-treatment bias re-simulated: ${N_REPS} replications of ` +
    `${N_PER_ARM} per arm,\nown generator\n`,
);

const seList = [];
for (let i = 1; i < lines.length; i++) {
  const f = splitCsv(lines[i]);
  const share = parseFloat(f[cShare]) / 100;
  const pubPlain = parseFloat(f[cPlain]);
  const pubEst = parseFloat(f[cEst]);
  const pubSe = parseFloat(f[cSe]);
  if (!Number.isFinite(share) || !Number.isFinite(pubEst)) {
    console.log(`  row ${i + 1} of the table is unreadable  FAIL`);
    failures++;
    continue;
  }

  const r = simulate(share, 0x5eedn + BigInt(i) * 104729n);
  // Two Monte Carlo means of the same quantity, one on N_REPS draws and the
  // published one on 4000, differ with this standard error.
  const mcSe = (sd) => sd * Math.sqrt(1 / N_REPS + 1 / 4000);
  console.log(`${(share * 100).toFixed(0)}% of the effect flows through the covariate`);
  check("plain diff-in-means", r.plain, pubPlain, mcSe(r.plainSd));
  check("CUPED estimate", r.adj, pubEst, mcSe(r.empiricalSd));

  // What CUPED actually returns is the direct effect, not the total. That is
  // the point of the table, and it is a theoretical prediction, not a fit.
  const zDirect = Math.abs(r.adj - r.direct) / (r.empiricalSd / Math.sqrt(N_REPS));
  const okDirect = zDirect <= BAND;
  if (!okDirect) failures++;
  console.log(
    `  ${"= the direct effect".padEnd(24)} node ${r.adj.toFixed(4)}  ` +
      `direct    ${r.direct.toFixed(4)}  ${zDirect.toFixed(1).padStart(14)} se  ` +
      `${okDirect ? "ok" : "FAIL"}`,
  );

  // The reported standard error has to be the estimator's real spread, or the
  // claim that nothing looks unstable would be an artefact of a bad formula.
  const gap = Math.abs(r.reportedSe - r.empiricalSd) / r.empiricalSd;
  const okSe = gap <= 0.1;
  if (!okSe) failures++;
  console.log(
    `  ${"reported SE is real".padEnd(24)} node ${r.reportedSe.toFixed(4)}  ` +
      `spread    ${r.empiricalSd.toFixed(4)}  ${(gap * 100).toFixed(1).padStart(11)}%  ` +
      `${okSe ? "ok" : "FAIL"}`,
  );
  if (Math.abs(r.reportedSe - pubSe) > 0.0005 + 0.05 * pubSe) {
    console.log(`  FAIL published SE ${pubSe} against ${r.reportedSe.toFixed(4)} here`);
    failures++;
  }
  seList.push(r.reportedSe);
  console.log();
}

// The sentence the table exists for: the standard error is the same in the
// column where CUPED is right as in the column where it reports nothing.
const spread = Math.max(...seList) - Math.min(...seList);
console.log(
  `standard error across all ${seList.length} rows: ` +
    `${Math.min(...seList).toFixed(4)} to ${Math.max(...seList).toFixed(4)}`,
);
if (spread > 0.001) {
  console.log("FAIL the standard error does move, so the table's warning is wrong");
  failures++;
}

if (failures > 0) {
  console.log(`\n${failures} checks failed`);
  process.exit(1);
}
console.log("\nnode reproduces the table, and CUPED returns the direct effect every time");
