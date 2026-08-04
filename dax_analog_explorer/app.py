"""Streamlit MVP — DAX Analog Day Explorer.

Run with:   streamlit run dax_analog_explorer/app.py

Describe the current DAX session, choose an observation cutoff (e.g. 10:30
Berlin), and the tool finds historically similar days, shows their charts, and
reports what happened AFTER the cutoff -- as a historical conditional
distribution, never a prediction, always with sample size.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

# Streamlit executes this file as a script, so Python may only add this file's
# directory to sys.path.  Add the repository root so the package imports below
# work whether Streamlit is launched from the repository root or elsewhere.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from dax_analog_explorer.config import Config
from dax_analog_explorer import search as S
from dax_analog_explorer import charts as ch
from dax_analog_explorer import reports as rp
from dax_analog_explorer.similarity_engine import FilterSpec

st.set_page_config(page_title="DAX Analog Day Explorer", layout="wide")
cfg = Config()


# --------------------------------------------------------------------------- #
# data loading (cached)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=True)
def load_data():
    p = cfg.paths
    if not p.daily_features.exists():
        return None
    daily = pd.read_parquet(p.daily_features)
    opening = pd.read_parquet(p.opening_path_features)
    master = pd.read_parquet(p.master_5m)
    audit = p.audit_report_txt.read_text() if p.audit_report_txt.exists() else ""
    return daily, opening, master, audit


data = load_data()
if data is None:
    st.error("Processed data not found. Run:  `python -m dax_analog_explorer.preprocess`")
    st.stop()
daily, opening, master, audit_txt = data
available = pd.DatetimeIndex(sorted(daily["session_date"].unique()))


def snap_date(d) -> pd.Timestamp:
    d = pd.Timestamp(d)
    idx = available.searchsorted(d)
    idx = min(max(idx, 0), len(available) - 1)
    # nearest of the two neighbours
    cands = [available[max(idx - 1, 0)], available[idx]]
    return min(cands, key=lambda x: abs((x - d).days))


# --------------------------------------------------------------------------- #
# sidebar controls
# --------------------------------------------------------------------------- #
st.sidebar.header("1 · Data & reference")
period = st.sidebar.selectbox("Historical period",
    ["Last year", "Last 5 years", "Last 10 years", "Entire dataset"], index=2)

default_ref = snap_date(available[-1] - pd.Timedelta(days=5))
ref_in = st.sidebar.date_input("Reference date", value=default_ref.date(),
    min_value=available[0].date(), max_value=available[-1].date())
reference_date = snap_date(ref_in)
if reference_date.date() != ref_in:
    st.sidebar.caption(f"↳ snapped to trading day **{reference_date.date()}**")

cut_choice = st.sidebar.selectbox("Observation cutoff (Berlin)",
    ["09:15", "09:30", "10:00", "10:30", "custom"], index=3)
if cut_choice == "custom":
    ct = st.sidebar.time_input("custom cutoff", value=dt.time(10, 30))
    cutoff_str = ct.strftime("%H:%M")
else:
    cutoff_str = cut_choice
cutoff_min = cfg.minutes_since_open(cfg.cutoff_time(cutoff_str))

st.sidebar.header("2 · Search mode")
mode = st.sidebar.radio("Mode", [S.MODE_COMBINED, S.MODE_SIM, S.MODE_EXACT], index=0)
n_results = st.sidebar.slider("Number of matches", 5, 50, cfg.default_n_results, 1)

st.sidebar.header("3 · Context filters")
use_ath = st.sidebar.checkbox("Near ATH", value=True)
ath_tol = st.sidebar.select_slider("ATH tolerance (%)",
    options=list(cfg.ath_tolerances_pct), value=cfg.default_ath_tolerance_pct)
req_ath_open = st.sidebar.checkbox("Open above prior ATH (new ATH at open)", value=False)
req_ath_on = st.sidebar.checkbox("Overnight reached prior ATH", value=False)
gap_sign = st.sidebar.selectbox("Gap direction", ["any", "positive", "negative"], index=1)
open_pdh = st.sidebar.checkbox("Open above PDH", value=True)
open_pdl = st.sidebar.checkbox("Open below PDL", value=False)
regimes = st.sidebar.multiselect("Volatility regime", ["low", "normal", "high"], default=[])

st.sidebar.header("4 · Opening-pattern filters")
fb_bull = st.sidebar.checkbox("Bullish first bar", value=True)
fb_pctl = st.sidebar.slider("First-bar range ≥ percentile", 0, 99, int(cfg.first_bar_range_pctl))
fb_close = st.sidebar.slider("First-bar close location ≥", 0.0, 1.0, cfg.first_bar_close_top_frac, 0.05)
ext_max = st.sidebar.slider("Extension ratio <", 0.0, 3.0, cfg.weak_extension_ratio_max, 0.05)
ov_min = st.sidebar.slider("Overlap ratio >", 0.0, 1.0, cfg.weak_overlap_ratio_min, 0.05)
weak_dl = st.sidebar.selectbox("Bearish weakness before", ["none", "10:00", "10:30", "10:45", "11:00"], index=2)

st.sidebar.header("5 · Similarity weights")
w_ctx = st.sidebar.slider("Market context", 0, 100, int(cfg.weight_context * 100))
w_fb = st.sidebar.slider("First-bar structure", 0, 100, int(cfg.weight_first_bar * 100))
w_ps = st.sidebar.slider("Post-spike path", 0, 100, int(cfg.weight_post_spike * 100))
w_wk = st.sidebar.slider("Weakness structure", 0, 100, int(cfg.weight_weakness * 100))
weights = {"context": w_ctx, "first_bar": w_fb, "post_spike": w_ps, "weakness": w_wk}

path_method = st.sidebar.selectbox("Path distance", ["correlation", "euclidean", "cosine", "dtw"], index=0)
path_unit = st.sidebar.selectbox("Path units", ["atr", "pct", "points", "prior_range"], index=0)


def build_spec() -> FilterSpec:
    return FilterSpec(
        ath_within_pct=(ath_tol if use_ath else None),
        require_ath_at_open=req_ath_open, require_ath_overnight=req_ath_on,
        gap_sign=(None if gap_sign == "any" else gap_sign),
        open_above_pdh=open_pdh, open_below_pdl=open_pdl,
        vol_regimes=(tuple(regimes) if regimes else None),
        first_bar_bull=fb_bull,
        first_bar_range_pctl_min=(fb_pctl if fb_pctl > 0 else None),
        first_bar_close_top_frac=(fb_close if fb_close > 0 else None),
        extension_ratio_max=(ext_max if ext_max < 3.0 else None),
        overlap_ratio_min=(ov_min if ov_min > 0 else None),
        weakness_before_min=(None if weak_dl == "none"
                             else cfg.minutes_since_open(cfg.cutoff_time(weak_dl))))


# --------------------------------------------------------------------------- #
# header + run
# --------------------------------------------------------------------------- #
st.title("DAX Analog Day Explorer")
st.caption("Futures-only, roll back-adjusted (method: **%s**). Analogs are a "
           "historical conditional distribution — not a prediction." % cfg.roll_adjust_method)

c1, c2 = st.columns([1, 3])
run = c1.button("🔎 Find analogs", type="primary", use_container_width=True)
flagship = c2.button("Load the worked example (ATH gap-up → weak follow-through, 10:30)",
                     use_container_width=True)

if flagship:
    st.session_state["run"] = True
    st.session_state["flagship"] = True
if run:
    st.session_state["run"] = True
    st.session_state["flagship"] = False

if st.session_state.get("run"):
    spec = S.flagship_filter(cfg) if st.session_state.get("flagship") else build_spec()
    wts = None if st.session_state.get("flagship") else weights
    use_cut = 90 if st.session_state.get("flagship") else cutoff_min
    use_cut_str = "10:30" if st.session_state.get("flagship") else cutoff_str
    with st.spinner("Matching…"):
        res = S.run_search(daily, opening, master, reference_date=reference_date,
                           cutoff_min=use_cut, cutoff_str=use_cut_str,
                           mode=(S.MODE_COMBINED if st.session_state.get("flagship") else mode),
                           spec=spec, weights=wts, period=period, n=n_results,
                           path_method=path_method, path_unit=path_unit, cfg=cfg)

    tabs = st.tabs(["① Query & matches", "② Explanations", "③ Charts",
                    "④ Statistics", "⑤ Dates only", "⑥ Data audit"])

    # ---- ① query + matches ----
    with tabs[0]:
        st.subheader("Interpreted query")
        st.markdown(res.interpreted)
        st.info(f"**Reference {res.reference_date.date()}** — {res.reference_desc}")
        st.subheader("Progressive matching")
        st.markdown(rp.relaxation_note(res.tiers))
        st.caption(f"Candidate pool: {res.n_pool} · showing top {len(res.ranked_top)}")
        show_cols = ["session_date", "similarity_score", "distance_open_to_ath_pct",
                     "gap_pct", "open_vs_pdh_pct", "first_bar_range", "extension_ratio",
                     "post_spike_overlap_ratio", "post_spike_efficiency",
                     "first_bearish_expansion_min", "return_at_cash_close_pct",
                     "max_favorable_excursion_pct", "max_adverse_excursion_pct",
                     "day_classification"]
        show_cols = [c for c in show_cols if c in res.merged_top.columns]
        tbl = res.merged_top[show_cols].copy()
        tbl["session_date"] = tbl["session_date"].dt.date
        st.dataframe(tbl.round(3), use_container_width=True, height=430)

    # ---- ② explanations ----
    with tabs[1]:
        st.subheader("Why each session matched")
        for _, r in res.merged_top.iterrows():
            d = r["session_date"]
            exp = res.explanations.get(d)
            if not exp:
                continue
            with st.expander(f"{pd.Timestamp(d).date()} · score {r['similarity_score']:.1f} · "
                             f"{r.get('day_classification','')}"):
                st.markdown(rp.similarity_explanation_text(exp))
                if r.get("day_classification_reason"):
                    st.caption("Classification: " + str(r["day_classification_reason"]))

    # ---- ③ charts ----
    with tabs[2]:
        st.subheader("Reference day")
        rbars = S.day_bars(master, res.reference_date, cfg)
        st.plotly_chart(ch.day_candles(rbars, res.cutoff_min, cfg,
                        title=f"Reference {res.reference_date.date()}"),
                        use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(ch.analog_overlay(res.grid, res.path_mat, res.path_kept,
                            res.reference_date, res.cutoff_min, res.path_unit),
                            use_container_width=True)
        with c2:
            ref_path = None
            if res.reference_date in res.path_kept:
                ref_path = res.path_mat[res.path_kept.index(res.reference_date)]
            others = np.array([res.path_mat[i] for i, d in enumerate(res.path_kept)
                               if d != res.reference_date])
            st.plotly_chart(ch.median_band(res.grid, others, res.cutoff_min, ref_path,
                            res.path_unit), use_container_width=True)
        st.subheader("Top analog candlesticks")
        top_dates = res.merged_top["session_date"].head(6).tolist()
        cols = st.columns(2)
        for i, d in enumerate(top_dates):
            b = S.day_bars(master, d, cfg)
            if len(b):
                cols[i % 2].plotly_chart(
                    ch.day_candles(b, res.cutoff_min, cfg, title=str(pd.Timestamp(d).date())),
                    use_container_width=True)

    # ---- ④ statistics ----
    with tabs[3]:
        h = res.headline
        st.subheader(f"Outcome distribution — sample size {h['sample_size']}")
        if h["sample_size"] < 5:
            st.warning("Very small sample — treat as indicative only.")
        k = st.columns(4)
        k[0].metric("Median close vs cutoff", f"{h['median_return_at_close_pct']:.2f}%")
        k[1].metric("Closed above open", f"{h['positive_close_%']:.0f}%")
        k[2].metric("Median MFE", f"{h['median_MFE_pct']:.2f}%")
        k[3].metric("Median MAE", f"{h['median_MAE_pct']:.2f}%")
        k2 = st.columns(4)
        k2[0].metric("Full gap fill", f"{h['full_gap_fill_%']:.0f}%")
        k2[1].metric("Broke observed high", f"{h['break_observed_high_%']:.0f}%")
        k2[2].metric("Broke observed low", f"{h['break_observed_low_%']:.0f}%")
        st.dataframe(res.stats.round(3), use_container_width=True)
        cc = st.columns(2)
        if "return_at_cash_close_pct" in res.merged_top:
            cc[0].plotly_chart(ch.outcome_distribution(
                res.merged_top["return_at_cash_close_pct"], "Return to cash close",
                "% from cutoff"), use_container_width=True)
        if "max_adverse_excursion_pct" in res.merged_top:
            cc[1].plotly_chart(ch.outcome_distribution(
                res.merged_top["max_adverse_excursion_pct"], "Max adverse excursion",
                "% from cutoff"), use_container_width=True)

    # ---- ⑤ dates only ----
    with tabs[4]:
        st.subheader("Dates only")
        do = rp.dates_only(res.merged_top["session_date"])
        st.dataframe(do, use_container_width=True, height=430)
        st.download_button("Download dates (CSV)", do.to_csv(index=False),
                           file_name="analog_dates.csv")
        st.code("\n".join(do["date"].tolist()))

    # ---- ⑥ audit ----
    with tabs[5]:
        st.subheader("Data audit")
        st.text(audit_txt or "run preprocess to generate the audit report")
else:
    st.info("Set the reference date and cutoff, then **Find analogs** — or click the "
            "worked-example button to reproduce the ATH gap-up → weak-follow-through query.")
    with st.expander("Data audit report"):
        st.text(audit_txt)
