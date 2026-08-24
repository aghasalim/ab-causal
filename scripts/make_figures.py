"""Draw the README figures from reports/*.csv.

Reads the saved simulation and LaLonde output only -- nothing is re-run.

    python scripts/make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

ALPHA = 0.05


def peeking(out: Path) -> Path:
    """What repeated testing does to the false-positive rate, and what fixes it.

    Peeking daily and stopping at p<0.05 turns a nominal 5% test into a 22.3% one.
    Both corrections restore it, and both cost power -- Pocock keeps 0.601 and
    mSPRT 0.419 against the fixed-horizon 0.750. Nothing here is free.
    """
    table = pd.read_csv(REPORTS / "peeking.csv")
    labels = [r.replace("peek daily, ", "peek\n").replace(" (test once)", "\n(test once)")
              .replace(" (always-valid)", "\n(always-valid)") for r in table["rule"]]
    positions = np.arange(len(table))

    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    colours = ["#b2182b" if e > ALPHA * 1.5 else "#1a9850"
               for e in table["type-I error"]]
    axes[0].bar(positions, table["type-I error"] * 100, 0.6, color=colours,
                edgecolor="0.3", lw=0.5)
    axes[0].axhline(ALPHA * 100, color="0.2", ls="--", lw=1.5)
    axes[0].set_ylabel("type-I error (%)")
    axes[0].set_title("nominal 5% is the dashed line", fontsize=10)
    for index, value in enumerate(table["type-I error"]):
        axes[0].text(index, value * 100 + 0.5, f"{value:.1%}", ha="center",
                     fontsize=9, fontweight="bold")

    axes[1].bar(positions, table["power"] * 100, 0.6, color="#2166ac",
                edgecolor="0.3", lw=0.5)
    axes[1].set_ylabel("power (%)")
    axes[1].set_title("what each correction costs", fontsize=10)

    axes[2].bar(positions, table["avg n/arm at stop"], 0.6, color="#f4a582",
                edgecolor="0.3", lw=0.5)
    axes[2].set_ylabel("average n per arm at stop")
    axes[2].set_title("and what it saves in sample", fontsize=10)

    for ax in axes:
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=7, rotation=20, ha="right")
        ax.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def cuped(out: Path) -> Path:
    """Measured variance reduction against the value the theory predicts.

    CUPED should reduce variance by rho^2 for a pre-period covariate correlated
    rho with the outcome. Plotting the measured reduction against that prediction
    is the check that the implementation is doing what it claims.
    """
    table = pd.read_csv(REPORTS / "cuped.csv")
    correlation = table.iloc[:, 0]

    figure, (left, right) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    left.plot(correlation, table["predicted (rho^2)"] * 100, "s--", color="0.45",
              lw=1.8, label="theory: $\\rho^2$")
    left.plot(correlation, table["variance reduction"] * 100, "o-", color="#2166ac",
              lw=2, label="measured")
    left.set_xlabel("corr(pre-period covariate, outcome)")
    left.set_ylabel("variance reduction (%)")
    left.set_title("CUPED delivers what the theory predicts", fontsize=10)
    left.legend(frameon=False, fontsize=9)
    left.spines[["top", "right"]].set_visible(False)

    right.plot(correlation, table["bias (plain)"], "o-", color="#bdbdbd", lw=2,
               label="plain difference")
    right.plot(correlation, table["bias (CUPED)"], "s-", color="#2166ac", lw=2,
               label="CUPED")
    right.axhline(0, color="0.2", lw=1.1)
    right.set_xlabel("corr(pre-period covariate, outcome)")
    right.set_ylabel("bias")
    right.set_title("and does not buy it with bias", fontsize=10)
    right.legend(frameon=False, fontsize=9)
    right.spines[["top", "right"]].set_visible(False)

    figure.tight_layout()
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def lalonde(out: Path) -> Path:
    """Observational estimates against the experimental benchmark.

    LaLonde is the standard test because the randomised answer is known: 1794.
    The naive difference on observational controls gets -8498 on CPS and -15205 on
    PSID, wrong by more than five times the effect and in the wrong direction.
    """
    table = pd.read_csv(REPORTS / "lalonde.csv")
    truth = float(table[table.method == "randomised experiment"].att.iloc[0])
    table = table[table.method != "randomised experiment"].copy()
    table["label"] = table.controls + " · " + table.method
    table = table.sort_values("att")
    positions = np.arange(len(table))

    figure, ax = plt.subplots(figsize=(11, 7.5))
    colours = ["#b2182b" if "naive" in m else "#2166ac" for m in table.method]
    ax.barh(positions, table.att, color=colours, edgecolor="0.3", lw=0.4)
    ax.axvline(truth, color="#1a9850", ls="--", lw=2)
    ax.text(truth, -0.9, f"  randomised benchmark = {truth:,.0f}", fontsize=9,
            color="#1a9850", va="bottom")
    ax.set_yticks(positions)
    ax.set_yticklabels(table.label, fontsize=7.5)
    ax.set_xlabel("estimated ATT ($)")
    ax.set_title(
        "Red bars are the naive difference on observational controls: wrong by "
        "more than\nfive times the effect, and the wrong sign.",
        fontsize=10,
    )
    ax.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def balance(out: Path) -> Path:
    """Covariate imbalance before and after adjustment."""
    table = pd.read_csv(REPORTS / "lalonde_balance.csv")
    table = table.sort_values("|before|")
    positions = np.arange(len(table))

    figure, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.barh(positions - 0.2, table["|before|"], 0.4, label="before adjustment",
            color="#b2182b", edgecolor="0.3", lw=0.4)
    ax.barh(positions + 0.2, table["|after|"], 0.4, label="after adjustment",
            color="#1a9850", edgecolor="0.3", lw=0.4)
    ax.axvline(0.1, color="0.35", ls="--", lw=1.3)
    ax.text(0.1, len(table) - 0.4, "  0.1, the usual rule of thumb", fontsize=8,
            color="0.4")
    ax.set_yticks(positions)
    ax.set_yticklabels(table.covariate, fontsize=9)
    ax.set_xlabel("|standardised mean difference|")
    ax.set_title(
        "Balance is a precondition, not a result. All covariates land under 0.12.",
        fontsize=10,
    )
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def overlap(out: Path) -> Path:
    """How much of each control pool is actually usable.

    Overlap is the assumption that makes any of these estimators identified. PSID
    keeps 1,068 of 2,490 controls inside the treated propensity range, and 82% of
    them sit below 0.01 -- so most of the nominal sample contributes nothing.
    """
    table = pd.read_csv(REPORTS / "lalonde_overlap.csv")
    positions = np.arange(len(table))

    figure, (left, right) = plt.subplots(
        1, 2, figsize=(12.5, 4.4), gridspec_kw={"width_ratios": [2, 1]}
    )
    left.bar(positions - 0.25, table["n control"], 0.25, label="nominal controls",
             color="#bdbdbd", edgecolor="0.3", lw=0.4)
    left.bar(positions, table["controls inside treated PS range"], 0.25,
             label="inside the treated propensity range", color="#2166ac",
             edgecolor="0.3", lw=0.4)
    left.bar(positions + 0.25, table["n control"] - table["controls with PS < 0.01"],
             0.25, label="propensity above 0.01", color="#1a9850",
             edgecolor="0.3", lw=0.4)
    left.set_xticks(positions)
    left.set_xticklabels(table.controls)
    left.set_yscale("log")
    left.set_ylabel("controls (log scale)")
    left.set_title("most of the control pool is unusable", fontsize=10)
    left.legend(frameon=False, fontsize=8)
    left.spines[["top", "right"]].set_visible(False)

    right.bar(positions, table["max control weight ps/(1-ps)"], 0.5,
              color="#b2182b", edgecolor="0.3", lw=0.4)
    right.set_xticks(positions)
    right.set_xticklabels(table.controls)
    right.set_ylabel("max single-control IPW weight")
    right.set_title("and one unit can dominate", fontsize=10)
    for index, value in enumerate(table["max control weight ps/(1-ps)"]):
        right.text(index, value + 2, f"{value:.1f}", ha="center", fontsize=9)
    right.spines[["top", "right"]].set_visible(False)

    figure.tight_layout()
    figure.savefig(out, dpi=110, bbox_inches="tight")
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
        print(f"-> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
