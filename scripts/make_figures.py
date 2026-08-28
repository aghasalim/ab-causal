"""Draw the README figures from reports/*.csv.

Reads the saved simulation and LaLonde output only. Nothing here re-runs an
experiment, so a figure can never disagree with a number quoted in the README.

    python scripts/make_figures.py
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from style import PALETTE, titled

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

ALPHA = 0.05

# Red is the rule or the estimate that is wrong, green is the answer we already
# know to be right. The README talks about them that way, so the figures do too.
BROKEN, TRUTH = PALETTE[1], PALETTE[2]
BLUE, ORANGE, PURPLE, GREY = PALETTE[0], PALETTE[3], PALETTE[4], PALETTE[5]


def peeking(out: Path) -> Path:
    """What repeated testing does to the false-positive rate, and what fixes it.

    Two panels rather than three bar charts. The point is the trade between
    error, power and sample, and the trade is only readable if power and sample
    share an axis pair.
    """
    table = pd.read_csv(REPORTS / "peeking.csv")
    short = {
        "fixed horizon (test once)": "fixed horizon",
        "peek daily, stop at p<0.05": "peek daily, uncorrected",
        "peek daily, Pocock boundary": "peek daily, Pocock",
        "peek daily, mSPRT (always-valid)": "peek daily, mSPRT",
    }
    table["label"] = table["rule"].map(short)
    table["colour"] = [BROKEN if e > ALPHA * 1.5 else TRUTH for e in table["type-I error"]]
    positions = np.arange(len(table))

    figure, (left, right) = plt.subplots(1, 2, figsize=(13.6, 4.9))

    left.barh(positions, table["type-I error"] * 100, 0.62, color=table["colour"])
    left.axvline(ALPHA * 100, color="#333333", ls="--", lw=1.4)
    left.text(ALPHA * 100 + 0.4, 3.42, "nominal 5%", fontsize=9, color="#333333",
              va="center")
    for index, value in enumerate(table["type-I error"]):
        left.text(value * 100 + 0.4, index, f"{value:.1%}", va="center", fontsize=9.5,
                  fontweight="bold", color=table["colour"][index])
    left.set_yticks(positions)
    left.set_yticklabels(table["label"])
    left.set_xlim(0, 26)
    left.set_ylim(-0.6, 3.75)
    left.invert_yaxis()
    left.set_xlabel("type-I error rate (% of A/A tests called significant)")
    left.grid(axis="y", visible=False)
    titled(left, "Peeking daily turns a 5% test into a 22% one",
           "20,000 simulated A/A tests, no effect at all, one look a day for 14 days")

    right.scatter(table["avg n/arm at stop"], table["power"] * 100, s=110,
                  color=table["colour"], zorder=3)
    # Hand placed so no label lands on another rule's marker.
    offsets = {"fixed horizon": (0, 13, "center"),
               "peek daily, uncorrected": (12, 0, "left"),
               "peek daily, Pocock": (0, 13, "center"),
               "peek daily, mSPRT": (0, 13, "center")}
    for _, row in table.iterrows():
        dx, dy, ha = offsets[row["label"]]
        right.annotate(f"{row['label']}\n{row['power']:.0%} power, n = {row['avg n/arm at stop']:,}",
                       (row["avg n/arm at stop"], row["power"] * 100),
                       textcoords="offset points", xytext=(dx, dy), ha=ha,
                       va="center" if dx else "bottom", fontsize=8.8, color="#444444")
    right.set_xlim(560, 1620)
    right.set_ylim(33, 99)
    right.set_xlabel("average sample at stop (users per arm)")
    right.set_ylabel("power (% of true effects detected)")
    titled(right, "The rule that looks best here is the one that fails on the left",
           "same runs under a true effect of 0.10 sigma, red is the rule with 22% error")

    figure.tight_layout()
    figure.savefig(out)
    plt.close(figure)
    return out


def cuped(out: Path) -> Path:
    """What CUPED buys, and the covariate choice that quietly breaks it.

    The old right-hand panel plotted bias against rho. Every value sat inside
    +-0.0007 of a true effect of 0.10, so the line was Monte Carlo noise drawn
    at 200x magnification. That claim is a sentence, not a chart, so it is a
    note on the left panel now and the right panel carries the failure mode
    instead.
    """
    gain = pd.read_csv(REPORTS / "cuped.csv")
    post = pd.read_csv(REPORTS / "cuped_post_treatment_bias.csv")
    correlation = gain["corr(X, Y)"]
    worst_bias = max(gain["bias (plain)"].abs().max(), gain["bias (CUPED)"].abs().max())

    figure, (left, right) = plt.subplots(1, 2, figsize=(13.6, 5.0))

    left.plot(correlation, gain["predicted (rho^2)"] * 100, "s--", color=GREY,
              lw=1.6, label="theory, $\\rho^2$")
    left.plot(correlation, gain["variance reduction"] * 100, "o-", color=BLUE,
              lw=2.2, label="measured")
    left.set_xlabel("corr(pre-period covariate, outcome) (unitless)")
    left.set_ylabel("variance reduction (%)")
    left.set_xlim(-0.04, 0.96)
    left.set_ylim(-4, 92)
    left.legend(loc="upper left")
    left.text(0.03, 0.62, f"bias never exceeds {worst_bias:.4f}\nagainst a true effect of 0.10",
              transform=left.transAxes, fontsize=9, color="#5a5a5a", va="top")
    titled(left, "CUPED cuts variance by rho squared, as the derivation says",
           "4,000 replications at each correlation, 2,000 users per arm")

    share = post["% of effect via covariate"].str.rstrip("%").astype(float)
    total = float(post["true TOTAL effect"].iloc[0])
    right.axhline(total, color=TRUTH, ls="--", lw=1.6)
    right.text(2, total + 0.004, f"true total effect = {total:.2f}", fontsize=9,
               color=TRUTH, va="bottom")
    right.plot(share, post["plain diff-in-means"], "o-", color=GREY, lw=2.0,
               label="plain difference in means")
    right.errorbar(share, post["CUPED estimate"], yerr=post["CUPED SE (unchanged)"],
                   fmt="s-", color=BROKEN, lw=2.2, capsize=4, label="CUPED, with its own SE")
    right.axhline(0, color="#999999", lw=1.0, zorder=1)
    right.set_xlabel("share of the true effect flowing through the covariate (%)")
    right.set_ylabel("estimated effect (outcome units)")
    right.set_xlim(-5, 105)
    right.set_ylim(-0.035, 0.145)
    right.legend(loc="lower left")
    titled(right, "A covariate that treatment moved erases the effect, silently",
           "the CUPED interval stays the same width in every column, SE 0.019 throughout")

    figure.tight_layout()
    figure.savefig(out)
    plt.close(figure)
    return out


def lalonde(out: Path) -> Path:
    """Observational estimates against the experimental benchmark.

    Split in two because the naive estimates are five figures negative and the
    twenty adjusted ones live inside a $4,000 window. On one axis the second
    group is a smear against the y axis and the spread is invisible.
    """
    table = pd.read_csv(REPORTS / "lalonde.csv")
    truth = float(table[table.method == "randomised experiment"].att.iloc[0])
    truth_se = float(table[table.method == "randomised experiment"].se.iloc[0])
    naive = table[table.method == "naive difference"]
    adjusted = table[table.spec != "-"].sort_values("att").copy()

    def label(row):
        method = re.sub(r"\s*\((linear|dehejia-wahba)\)$", "", row.method)
        spec = "linear PS" if row.spec == "linear" else "DW PS"
        return f"{row.controls.upper()}, {spec}, {method}"

    adjusted["label"] = [label(r) for r in adjusted.itertuples()]

    figure, (left, right) = plt.subplots(
        1, 2, figsize=(14.2, 7.4), gridspec_kw={"width_ratios": [1, 2.5]}
    )

    bars = [truth] + list(naive.att)
    names = ["randomised\nexperiment", "naive,\nCPS controls", "naive,\nPSID controls"]
    left.bar([0, 1, 2], bars, 0.58, color=[TRUTH, BROKEN, BROKEN])
    for index, value in enumerate(bars):
        sign = "" if value > 0 else "-"
        left.text(index, value + (400 if value > 0 else -400), f"{sign}${abs(value):,.0f}",
                  ha="center", va="bottom" if value > 0 else "top", fontsize=9.5,
                  fontweight="bold")
    left.axhline(0, color="#666666", lw=1.0)
    left.set_xticks([0, 1, 2])
    left.set_xticklabels(names, fontsize=9)
    left.set_xlim(-0.7, 2.7)
    left.set_ylim(-18200, 4200)
    left.set_ylabel("estimated ATT (US dollars)")
    left.grid(axis="x", visible=False)
    titled(left, "Swapping the controls flips the sign",
           "same trainees, same outcome")

    positions = np.arange(len(adjusted))
    right.axvspan(truth - truth_se, truth + truth_se, color=TRUTH, alpha=0.10, lw=0)
    right.axvline(truth, color=TRUTH, ls="--", lw=1.8)
    right.text(truth + 110, 3.1,
               f"randomised benchmark ${truth:,.0f}\nband is one SE (${truth_se:,.0f}) either side",
               fontsize=9, color=TRUTH, va="center")
    colours = [BLUE if c == "cps" else PURPLE for c in adjusted.controls]
    right.hlines(positions, 0, adjusted.att, color="#d5d5d5", lw=1.2, zorder=1)
    for pool, colour in (("cps", BLUE), ("psid", PURPLE)):
        keep = adjusted.controls == pool
        right.scatter(adjusted.att[keep], positions[keep.to_numpy()], s=70,
                      color=colour, zorder=3, label=f"{pool.upper()} controls")
    for edge in (0, len(adjusted) - 1):
        right.annotate(f"${adjusted.att.iloc[edge]:,.0f}",
                       (adjusted.att.iloc[edge], edge), textcoords="offset points",
                       xytext=(11, 0), va="center", fontsize=9.5,
                       fontweight="bold", color=colours[edge])
    right.set_yticks(positions)
    right.set_yticklabels(adjusted.label, fontsize=8.5)
    right.set_xlim(0, 4500)
    right.set_ylim(-0.8, len(adjusted) - 0.2)
    right.set_xlabel("estimated ATT (US dollars)")
    right.grid(axis="y", visible=False)
    right.legend(loc="lower right")
    titled(right, "Every adjusted estimator lands nearby, and they still spread 16-fold",
           "4 estimators, 2 control pools, 2 propensity specifications, sorted by estimate")

    figure.tight_layout()
    figure.savefig(out)
    plt.close(figure)
    return out


def balance(out: Path) -> Path:
    """Covariate imbalance before and after weighting, as a Love plot.

    Paired bars needed eight pairs to say one thing. A dot per state with the
    move drawn between them says it in half the ink, and the after-dots pile up
    against the threshold line where they belong.
    """
    table = pd.read_csv(REPORTS / "lalonde_balance.csv").sort_values("|before|")
    positions = np.arange(len(table))

    figure, ax = plt.subplots(figsize=(10.0, 5.4))
    ax.hlines(positions, table["|after|"], table["|before|"], color="#cccccc", lw=1.6,
              zorder=1)
    ax.scatter(table["|before|"], positions, s=80, color=BROKEN, zorder=3,
               label="before weighting")
    ax.scatter(table["|after|"], positions, s=80, color=TRUTH, zorder=3,
               label="after weighting")
    for index, value in enumerate(table["|before|"]):
        ax.text(value + 0.05, index, f"{value:.2f}", va="center", fontsize=9,
                color=BROKEN)
    ax.axvline(0.1, color="#666666", ls="--", lw=1.3)
    ax.text(0.13, -0.55, "0.1, the usual rule of thumb", fontsize=9, color="#666666",
            va="center")
    ax.set_yticks(positions)
    ax.set_yticklabels(table.covariate)
    ax.set_xlim(-0.06, 2.72)
    ax.set_ylim(-0.85, len(table) - 0.35)
    ax.set_xlabel("|standardised mean difference| between treated and controls (unitless)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    titled(ax, "Weighting removes every imbalance the propensity model can see",
           f"CPS pool, Dehejia-Wahba propensity, IPW weights: worst |SMD| falls to "
           f"{table['|after|'].max():.3f}")

    figure.tight_layout()
    figure.savefig(out)
    plt.close(figure)
    return out


def overlap(out: Path) -> Path:
    """How much of each control pool is actually usable.

    The counts run from 449 to 15,992, so they want a log axis, and bars on a
    log axis are meaningless because the bar starts wherever the axis was cut.
    Dots carry the same numbers and claim nothing about length.
    """
    table = pd.read_csv(REPORTS / "lalonde_overlap.csv")
    table["above 0.01"] = table["n control"] - table["controls with PS < 0.01"]
    series = [("n control", GREY, "nominal control pool"),
              ("controls inside treated PS range", BLUE, "inside the treated propensity range"),
              ("above 0.01", TRUTH, "propensity above 0.01")]
    positions = np.arange(len(table))[::-1]

    figure, (left, right) = plt.subplots(
        1, 2, figsize=(13.6, 4.6), gridspec_kw={"width_ratios": [2.1, 1]}
    )

    left.hlines(positions, table[series[2][0]], table["n control"], color="#d5d5d5",
                lw=1.6, zorder=1)
    for column, colour, name in series:
        left.scatter(table[column], positions, s=95, color=colour, zorder=3, label=name)
        for row, value in zip(positions, table[column]):
            left.text(value, row - 0.17, f"{value:,}", ha="center", va="top",
                      fontsize=9, color=colour)
    left.set_xscale("log")
    left.set_yticks(positions)
    left.set_yticklabels([c.upper() for c in table.controls])
    left.set_xlim(250, 40_000)
    left.set_ylim(-0.7, 2.3)
    left.set_xlabel("controls remaining (count, log scale)")
    left.grid(axis="y", visible=False)
    left.legend(loc="upper left", ncol=1)
    titled(left, "Most of the control pool never enters the comparison",
           "each check is applied to the whole pool, so the two are not nested")

    weights = table["max control weight ps/(1-ps)"]
    bars = np.arange(len(table))
    right.bar(bars, weights, 0.5, color=[BROKEN if w > 50 else ORANGE for w in weights])
    for row, value in zip(bars, weights):
        right.text(row, value + 2.5, f"{value:.1f}", ha="center", fontsize=9.5,
                   fontweight="bold")
    right.set_xticks(bars)
    right.set_xticklabels([c.upper() for c in table.controls])
    right.set_xlim(-0.7, 1.7)
    right.set_ylim(0, 112)
    right.set_ylabel("largest single-control IPW weight (unitless)")
    right.grid(axis="x", visible=False)
    titled(right, "In PSID one control stands in for 94",
           "largest ps/(1-ps) weight in each pool")

    figure.tight_layout()
    figure.savefig(out)
    plt.close(figure)
    return out


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for path in (
        peeking(FIGURES / "peeking.png"),
        cuped(FIGURES / "cuped.png"),
        lalonde(FIGURES / "lalonde.png"),
        balance(FIGURES / "balance.png"),
        overlap(FIGURES / "overlap.png"),
    ):
        print(f"-> {path.relative_to(ROOT)} ({path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
