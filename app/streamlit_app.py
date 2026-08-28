"""Experiment analyser: the checks worth running before and after a test."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.abcausal import config, diagnostics, sequential  # noqa: E402

st.set_page_config(page_title="A/B test analyser", page_icon="📊", layout="wide")


@st.cache_data(show_spinner="calibrating boundary…")
def boundary(n_looks: int) -> float:
    return sequential.calibrate_pocock(n_looks=n_looks, alpha=0.05, seed=4242, n_reps=20_000)


@st.cache_data
def report(name: str) -> pd.DataFrame | None:
    p = config.REPORTS / name
    return pd.read_csv(p) if p.exists() else None


st.title("A/B test analyser")
st.caption(
    "Built around one idea: a result is only as good as the procedure that "
    "produced it. Every rule here was scored against simulations where the true "
    "effect is known, see the Evidence tab."
)

analyse, plan, evidence = st.tabs(["Analyse a result", "Plan a test", "Evidence"])

# ---------------------------------------------------------------- analyse ---
with analyse:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Control")
        n_c = st.number_input("users", 1, 10_000_000, 10_000, key="nc")
        conv_c = st.number_input("conversions", 0, 10_000_000, 500, key="cc")
    with c2:
        st.subheader("Treatment")
        n_t = st.number_input("users", 1, 10_000_000, 10_000, key="nt")
        conv_t = st.number_input("conversions", 0, 10_000_000, 560, key="ct")

    looks = st.slider(
        "How many times have you checked results so far?", 1, 30, 1,
        help="Every check is another chance to cross the threshold by luck. "
             "One check = a standard fixed-horizon test.",
    )

    p_c, p_t = conv_c / n_c, conv_t / n_t
    lift = p_t - p_c
    se = np.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    z = lift / se if se > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    st.divider()
    srm = diagnostics.srm({"control": int(n_c), "treatment": int(n_t)})
    if srm["srm_detected"]:
        st.error(
            f"**Sample ratio mismatch** (p={srm['p_value']:.2e}). The split is not "
            "what you intended, so the arms are not comparable. Fix assignment "
            "before reading anything below. No analysis repairs a broken split."
        )
    else:
        st.success(f"Sample ratio check passed (p={srm['p_value']:.3f}).")

    m1, m2, m3 = st.columns(3)
    m1.metric("Control", f"{p_c:.2%}")
    m2.metric("Treatment", f"{p_t:.2%}", f"{lift:+.2%}")
    m3.metric("95% CI on lift", f"[{lift - 1.96 * se:+.2%}, {lift + 1.96 * se:+.2%}]")

    crit = 1.96 if looks == 1 else boundary(looks)
    significant = abs(z) > crit

    st.markdown(
        f"**z = {z:.2f}**, uncorrected p = {p_value:.4f}. "
        f"Threshold after {looks} look{'s' if looks > 1 else ''}: **|z| > {crit:.2f}**"
    )
    if significant:
        st.success("Significant, after correcting for how many times you looked.")
    elif abs(z) > 1.96:
        st.warning(
            f"This clears the uncorrected 1.96 bar but **not** the {crit:.2f} bar that "
            f"{looks} looks require. With {looks} checks, a true-null experiment "
            "crosses 1.96 far more than 5% of the time, which is what the Evidence "
            "tab measures. Keep running."
        )
    else:
        st.info("Not significant.")

# ------------------------------------------------------------------- plan ---
with plan:
    st.subheader("What can this experiment actually detect?")
    c1, c2 = st.columns(2)
    with c1:
        base = st.number_input("baseline conversion rate (%)", 0.01, 99.0, 5.0) / 100
        n_arm = st.number_input("users per arm", 100, 50_000_000, 10_000, step=1000)
    with c2:
        power = st.slider("power", 0.5, 0.99, 0.8)
        alpha = st.slider("alpha", 0.01, 0.10, 0.05)

    detectable = diagnostics.mde_proportion(int(n_arm), base, alpha, power)
    st.metric(
        "Minimum detectable effect",
        f"{detectable:.3%} absolute",
        f"{detectable / base:.1%} relative",
    )
    st.caption(
        "An experiment that cannot detect an effect you would act on is not "
        "inconclusive. It was unanswerable before it started."
    )

    target = st.number_input("effect you want to detect (relative %)", 0.1, 100.0, 5.0) / 100
    need = diagnostics.required_n(target * base, np.sqrt(base * (1 - base)), alpha, power)
    st.write(f"To detect a **{target:.1%} relative** change you need **{need:,} users per arm**.")

# --------------------------------------------------------------- evidence ---
with evidence:
    st.subheader("Peeking")
    df = report("peeking.csv")
    if df is not None:
        st.dataframe(df, hide_index=True, width="stretch")
        st.caption(
            "20,000 simulated experiments per rule. Read the error and power "
            "columns together: naive peeking shows higher power only because it "
            "declares significance more often whether or not anything is there."
        )
    st.subheader("CUPED on a covariate that treatment moved")
    df = report("cuped_post_treatment_bias.csv")
    if df is not None:
        st.dataframe(df, hide_index=True, width="stretch")
        st.caption("The standard error is identical in every row.")

    st.subheader("Observational estimators vs a randomised benchmark")
    df = report("lalonde.csv")
    if df is not None:
        st.dataframe(df, hide_index=True, width="stretch")
        st.caption("Truth is +$1,794, from the randomised sample.")
    if all(report(f) is None for f in ("peeking.csv", "lalonde.csv")):
        st.info("Run `make experiments` to generate these tables.")
