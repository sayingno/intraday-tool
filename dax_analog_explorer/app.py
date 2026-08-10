"""DAX Open Trend Continuation — the Streamlit front end.

Two things, and deliberately only two:

  1. **Read one session.**  Pick a day and a moment, get the OTC state as it
     stood then, every trigger counted with its stop / target / R:R, and the
     chart with the anatomy drawn on it.
  2. **Filter by context, then look at OTC inside that context.**  Choose
     categories — location, swing, prior day, gap, opening — and see how often
     the setup even appears there and what it did.

The analog-similarity search that used to live here is gone; its engines remain
in the package for the CLIs, but nothing in this app depends on them.

    streamlit run dax_analog_explorer/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):                      # `streamlit run app.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from dax_analog_explorer.config import Config
from dax_analog_explorer import charts, context as ctx, continuation as co
from dax_analog_explorer import otc, otc_report as orp

st.set_page_config(page_title="DAX · Open Trend Continuation", layout="wide")
cfg = Config()

CATEGORIES = {
    "location": ctx.LOCATION, "swing": ctx.SWING, "ma_state": ctx.MA_STATE,
    "prior_day_type": ctx.PRIOR_DAY, "gap_bucket": ctx.GAP,
    "open_loc": ctx.OPEN_LOC, "overnight_type": ctx.OVERNIGHT, "opening": ctx.OPENING,
}
LABELS = {"location": "Where we are", "swing": "Swing state",
          "ma_state": "Moving-average state", "prior_day_type": "Yesterday",
          "gap_bucket": "Gap", "open_loc": "Open vs yesterday's range",
          "overnight_type": "Overnight", "opening": "Opening 15 minutes"}


# --------------------------------------------------------------------------- #
# loading — the heavy work happens once per session, not once per widget change
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="loading bars …")
def load_bars():
    master = pd.read_parquet(cfg.paths.master_5m)
    return master, co.session_groups(master, cfg)


@st.cache_data(show_spinner="loading daily features …")
def load_daily() -> pd.DataFrame:
    return pd.read_parquet(cfg.paths.daily_features)


@st.cache_data(show_spinner="labelling context (first run ~25s) …")
def load_context(decision_min: int) -> pd.DataFrame:
    return ctx.load_context_table(cfg, decision_min)


@st.cache_data(show_spinner="walking every session (~20s) …")
def load_trades(deadline: int, max_entries: int, cost: float) -> pd.DataFrame:
    _, groups = load_bars()
    p = otc.OTCParams(trigger_deadline_min=deadline, max_entries=max_entries,
                      cost_points=cost)
    return orp.build_otc_trades(groups, load_daily(), p)


try:
    master, groups = load_bars()
    daily = load_daily()
except FileNotFoundError:
    st.error("No processed data yet. Run this once, from the repository root:\n\n"
             "`python -m dax_analog_explorer.preprocess --cutoff 10:30`")
    st.stop()

sessions = sorted(groups)
di = daily.set_index("session_date")

# The final session in the data can be a partial day (the newest FDAX file ends
# mid-session), which makes a confusing first impression -- every trade exits on
# "close" at whatever bar the data stopped.  Open on the last COMPLETE session.
_full = [d for d in sessions if groups[d]["mso"].max() >= 480]
DEFAULT_DAY = (_full or sessions)[-1]


# --------------------------------------------------------------------------- #
# sidebar — scope only.  Nothing here tunes the setup; see otc.OTCParams.
# --------------------------------------------------------------------------- #
st.sidebar.header("Session")
picked = st.sidebar.date_input("Date", value=DEFAULT_DAY.date(),
                               min_value=sessions[0].date(), max_value=sessions[-1].date())
sd = pd.Timestamp(picked).normalize()
if sd not in groups:                                # weekend / holiday
    earlier = [d for d in sessions if d <= sd]
    sd = earlier[-1] if earlier else sessions[0]
    st.sidebar.caption(f"↳ no session that day; showing **{sd.date()}**")

as_of = st.sidebar.slider("Read the session as of", min_value=30, max_value=510,
                          value=90, step=5,
                          format="%d min", help="minutes after the 09:00 open")
st.sidebar.caption(f"↳ **{otc.clock(as_of)}** Berlin — triggers after this are ignored")

st.sidebar.header("Scope")
max_entries = st.sidebar.slider("Max entries per session", 1, 4, 2)
cost = st.sidebar.number_input("Round-trip cost (points)", 0.0, 20.0, 2.0, 0.5)
st.sidebar.caption("The setup itself has no tunable values — every rule is a bar "
                   "count or a structural comparison. These three are scope only.")

P = otc.OTCParams(trigger_deadline_min=as_of, max_entries=max_entries, cost_points=cost)

st.title("DAX · Open Trend Continuation")
st.caption("Open develops direction → genuine follow-through → controlled pullback "
           "→ continuation trigger in the same direction. Nothing else is a trade.")

tab_day, tab_ctx = st.tabs(["① Read a session", "② Context filter"])


# --------------------------------------------------------------------------- #
# ① one session
# --------------------------------------------------------------------------- #
def evaluate(day: pd.Timestamp, p: otc.OTCParams):
    bars = groups[day].sort_values("mso")
    levels = orp._levels(di.loc[day])
    res = otc.evaluate_session(bars, levels, p)
    sims = [orp.simulate_signal(bars, s, p) if s.taken else {"status": "NOT_TAKEN"}
            for s in res.signals]
    return bars, levels, res, sims


def signal_table(res: otc.OTCResult, sims: list[dict]) -> pd.DataFrame:
    rows = []
    for s, sim in zip(res.signals, sims):
        rows.append({
            "signal": s.label, "at": otc.clock(s.signal_mso),
            "side": "buy" if s.side > 0 else "sell",
            "trigger": round(s.entry_px), "stop": round(s.stop_px),
            "target": None if s.target_px is None else round(s.target_px),
            "T1": s.target_name or "—", "R:R": s.rr,
            "risk (pts)": round(abs(s.entry_px - s.stop_px)),
            "outcome": sim.get("status", ""),
            "exit": otc.clock(sim["exit_mso"]) if sim.get("exit_mso") is not None else "—",
            "why": sim.get("reason", "—"),
            "R": None if sim.get("R_multiple") is None else round(sim["R_multiple"], 2)})
    return pd.DataFrame(rows)


with tab_day:
    bars, levels, res, sims = evaluate(sd, P)
    left, right = st.columns([2, 3])

    with left:
        st.subheader(f"{sd.date():%A %d %B %Y} · as of {otc.clock(as_of)}")
        state_note = {otc.ACTIVE: "success", otc.DEVELOPING: "warning",
                      otc.INVALID: "error"}[res.state]
        # markdown swallows single newlines; the sentence is written one fact per
        # line and has to stay that way to be readable
        getattr(st, state_note)(otc.live_status(res, P).replace("\n", "  \n"))

        if sd in load_context(30).set_index("session_date").index:
            row = load_context(30).set_index("session_date").loc[sd]
            st.markdown("**Context**")
            st.dataframe(pd.DataFrame(
                {"dimension": [LABELS[f] for f in CATEGORIES],
                 "label": [row[f] for f in CATEGORIES]}),
                hide_index=True, use_container_width=True)

    with right:
        title = f"{sd.date()} · reached {res.reached}"
        st.plotly_chart(
            charts.otc_session(bars, {**levels, "OPEN": di.loc[sd].get("cash_open_adj")},
                               orp.session_mark(bars, res, sims), as_of, title),
            use_container_width=True)

    if res.signals:
        st.markdown("**Triggers counted**")
        st.dataframe(signal_table(res, sims), hide_index=True, use_container_width=True)
    else:
        st.info("No trigger counted in this window — the setup never reached state C. "
                "Drag *Read the session as of* later to see whether one forms.")


# --------------------------------------------------------------------------- #
# ② context filter -> how OTC behaves inside it
# --------------------------------------------------------------------------- #
def performance_block(sub: pd.DataFrame, label: str) -> None:
    filled = sub[sub.status == "TRADED"]
    if len(filled) < 5:
        st.write(f"{label}: {len(sub)} signals, {len(filled)} filled — too few to quote.")
        return
    perf = co.performance(filled)
    c = st.columns(5)
    c[0].metric("filled trades", len(filled))
    c[1].metric("expectancy", f"{perf['expectancy_R']:+.3f}R")
    c[2].metric("t-stat", f"{perf['t_stat']:.2f}")
    c[3].metric("win rate", f"{perf['win_rate_%']:.1f}%")
    c[4].metric("profit factor", f"{perf['profit_factor']:.2f}")


def by_count(sub: pd.DataFrame) -> pd.DataFrame:
    """H1/L1 against H2/L2 side by side — the mandate's own claim, tested."""
    rows = []
    for k, name in ((1, "H1 / L1 — first attempt"), (2, "H2 / L2 — second attempt"),
                    (3, "H3+ / L3+ — wedge territory")):
        part = sub[sub["count"] == k] if k < 3 else sub[sub["count"] >= 3]
        filled = part[part.status == "TRADED"]
        if len(filled) < 5:
            rows.append({"attempt": name, "signals": len(part), "filled": len(filled),
                         "expectancy": None, "t": None, "win %": None, "profit factor": None})
            continue
        p = co.performance(filled)
        rows.append({"attempt": name, "signals": len(part), "filled": len(filled),
                     "expectancy": round(p["expectancy_R"], 3), "t": round(p["t_stat"], 2),
                     "win %": round(p["win_rate_%"], 1),
                     "profit factor": round(p["profit_factor"], 2)})
    return pd.DataFrame(rows)


with tab_ctx:
    st.subheader("Pick the context, then look at what the setup did inside it")
    st.caption("Categories only — no thresholds. Every cut is a rolling percentile of "
               "the instrument's own prior history, so a label means the same thing in "
               "2003 and 2026. Leave a box empty to ignore that dimension.")

    table = load_context(30)
    cols = st.columns(4)
    chosen = {}
    for i, (field, values) in enumerate(CATEGORIES.items()):
        with cols[i % 4]:
            got = st.multiselect(LABELS[field], list(values), default=[], key=f"ctx_{field}")
            if got:
                chosen[field] = tuple(got)

    spec = ctx.ContextSpec(**chosen)
    mask = ctx.apply_context(table, spec)
    picked_dates = table.loc[mask, "session_date"]
    st.markdown(f"**{int(mask.sum())} of {len(table)} sessions** match "
                f"({100 * mask.mean():.1f}%).")

    if chosen:
        with st.expander("What each condition costs in sample size"):
            st.dataframe(ctx.context_funnel(table, spec), hide_index=True,
                         use_container_width=True)

    if mask.sum() == 0:
        st.warning("No session matches that combination — loosen a dimension.")
        st.stop()

    trades = load_trades(as_of, max_entries, cost)
    sub = trades[trades.session_date.isin(set(picked_dates))] if len(trades) else trades

    st.markdown("### Does the setup even appear here?")
    reached = (trades[trades.session_date.isin(set(picked_dates))]
               .groupby("session_date").state.first())
    with_sig = len(reached)
    st.write(f"**{with_sig} of {int(mask.sum())}** matching sessions "
             f"({100 * with_sig / max(mask.sum(), 1):.1f}%) produced at least one trigger "
             f"by {otc.clock(as_of)}, against 47.3% across all history.")

    st.markdown("### And what did it do?")
    performance_block(sub, "all triggers")
    st.caption("H2/L2 is the mandate's primary — the second attempt at the pullback. "
               "A blank row means too few filled trades in this context to quote a rate.")
    st.dataframe(by_count(sub), hide_index=True, use_container_width=True)

    if len(sub[sub.status == "TRADED"]) >= 20:
        with st.expander("In-sample vs out-of-sample"):
            st.dataframe(co.split_performance(sub[sub.status == "TRADED"], "2017-01-01"),
                         hide_index=True, use_container_width=True)

    st.markdown("### The matching sessions")
    show = table.loc[mask, ["session_date"] + list(CATEGORIES)].copy()
    show["triggers"] = show.session_date.map(
        sub.groupby("session_date").label.apply(lambda s: " ".join(s)) if len(sub) else {})
    show["R"] = show.session_date.map(
        sub[sub.status == "TRADED"].groupby("session_date").R_multiple.sum().round(2)
        if len(sub) else {})
    st.dataframe(show.sort_values("session_date", ascending=False),
                 hide_index=True, use_container_width=True, height=320)
    st.download_button("Download these dates as CSV", show.to_csv(index=False),
                       file_name="context_matches.csv", mime="text/csv")

    st.markdown("### Chart one of them")
    pick = st.selectbox("Session", list(show.session_date.sort_values(ascending=False)),
                        format_func=lambda d: f"{d.date()} ({d.day_name()[:3]})")
    if pick is not None:
        b2, lv2, r2, s2 = evaluate(pd.Timestamp(pick).normalize(), P)
        st.plotly_chart(
            charts.otc_session(b2, {**lv2, "OPEN": di.loc[pick].get("cash_open_adj")},
                               orp.session_mark(b2, r2, s2), as_of,
                               f"{pd.Timestamp(pick).date()} · reached {r2.reached}"),
            use_container_width=True)
        if r2.signals:
            st.dataframe(signal_table(r2, s2), hide_index=True, use_container_width=True)
