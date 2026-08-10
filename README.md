# DAX · Open Trend Continuation

One setup, read off pure price action on 5-minute DAX futures bars:

> **Open develops direction → genuine follow-through → controlled pullback →
> continuation trigger in the same direction.**

The tool answers two questions. *What is this morning doing right now* — the OTC
state as of any moment, with every H1/H2 or L1/L2 trigger counted and its stop,
target and R:R read off the structure ([§7d](#7d-open-trend-continuation-otc--the-one-executable-setup)).
And *what has that done historically in this context* — filter by categorical
context (where we are, the swing, yesterday, the gap, the opening) and get the
conditional distribution with its sample size ([§7c](#7c-categorical-context-recognition)).

**No fitted thresholds and no indicators in the decision path.** Every condition
is a bar count or a structural comparison; the only numbers are scope choices
(bar size, deadline, entry cap).

The repository also still carries the earlier analog-similarity engine — describe
a session, find historically similar days, read what happened after the cutoff
(§6, §7, §7b). It is no longer what the app shows, and nothing in the OTC path
depends on it.

---

## Quick start

Everything runs from the repository root. The raw data is committed, so a fresh
clone has what it needs — only `data/processed/` has to be built.

```bash
pip install -r requirements.txt
python -m pytest                                          # 56 tests, ~2.4s
python -m dax_analog_explorer.preprocess --cutoff 10:30   # ~1m15s, once
python -m dax_analog_explorer.otc_report --date 2026-05-05   # the live sentence
streamlit run dax_analog_explorer/app.py                  # the app
```

Full command list in [§8](#8-running-it).

### On Windows

Same commands, three differences. Run them in **PowerShell**, not Command
Prompt:

```powershell
py -m pip install -r requirements.txt
py -m pytest
py -m dax_analog_explorer.preprocess --cutoff 10:30
py -m dax_analog_explorer.otc_report --date 2026-05-05
py -m streamlit run dax_analog_explorer\app.py
```

1. **`py -m pip`, never bare `pip`.** `pip.exe` is a launcher with the path to
   its interpreter baked in at install time. Move or uninstall that Python and it
   fails with `Fatal error in launcher: Unable to create process using
   "…\Python39\python.exe"` — the shim survives, the interpreter does not. `py`
   is the version launcher; it finds whatever is actually installed. `py -0p`
   lists them.
2. **`py`, not `python`.** A bare `python` on a machine with no python.org
   install hits the Microsoft Store alias and answers *"Python was not found; run
   without arguments to install from the Microsoft Store"*. If `py -0p` lists
   nothing, install **Python 3.12 from python.org** — not the Store — and tick
   **"Add python.exe to PATH"** in the installer.
3. **Never paste the `#` comments into Command Prompt.** `cmd.exe` has no comment
   syntax, so `pip install -r requirements.txt   # pandas, numpy` passes
   `#`, `pandas,` and `numpy` to pip as filenames. PowerShell and bash both
   understand `#`; `cmd.exe` does not.

Python 3.9 or newer works (every module carries `from __future__ import
annotations`); 3.11 is what the timings above were measured on.

---

## 1. What the data actually is (audited, not assumed)

Two raw sources ship in the repo. The pipeline **detects** their properties and
reports them (`data/processed/audit_report.txt`) rather than trusting labels:

| File(s) | Instrument | Bar | Timezone (verified) | Coverage |
|---|---|---|---|---|
| `dax-5m.zip` → `dax-5m.csv` | DAX **futures** (front-month, raw levels) | 5-min | **America/Chicago** → converted to Berlin | 2000-05-29 … 2024-09-18 |
| `FDAX_1min_*.csv` (7 files) | FDAX **futures**, raw quarterly contracts | 1-min | **Europe/Berlin** (Eurex) | 2024-11-01 … 2026-05-19 |

How we know:

* **Timezone** is inferred from the intraday **volume profile** — the DAX open
  (09:00) and close (17:30) auctions are unmistakable fingerprints. For
  `dax-5m` the 2019–24 volume peaks at **02:00 Chicago = 09:00 Berlin**; the
  Chicago→Berlin conversion is DST-aware, so the ~5–6 weeks/year when the US and
  EU daylight-saving calendars disagree get the correct **6 h** offset instead
  of 7 h. A naïve "+7h" would misplace the session by an hour for those weeks.
* **Both sources are raw futures, not back-adjusted**: `dax-5m` shows genuine
  historical levels (3621 at the Mar-2009 low, 2187 at the Mar-2003 bottom) and
  carries real volume; the FDAX files are 7 separate contracts with seam jumps
  up to ~3 %.
* There is a **42-day gap** between the two datasets (2024-09-19 … 2024-10-31).
  It is reported, never silently filled.

> Everything downstream treats price as **futures** and computes a single
> **roll-adjusted continuous** series (see §3).

---

## 2. Required file formats

**DAX 5-minute** — semicolon-separated, **no header**:

```
DD/MM/YYYY;HH:MM:SS;open;high;low;close;volume
29/05/2000;02:10:00;7018.5;7026;7015;7020.5;431
```

**FDAX 1-minute** — CSV with header (Interactive Brokers export):

```
Datetime,Instrument,IBSymbol,LocalSymbol,ContractMonth,Expiry,Exchange,Currency,
Multiplier,TradingClass,ConId,SecType,Open,High,Low,Close,Volume,Average,BarCount
```

Only `Datetime, Open, High, Low, Close, Volume, ContractMonth, Expiry, ConId,
LocalSymbol` are required. Contracts must not overlap in time (they don't).

To add more data, drop files matching those formats next to the existing ones
(`FDAX_1min_*.csv` glob; the DAX csv inside the zip) and re-run preprocessing.

---

## 3. Back-adjustment (how the futures ATH is made honest)

Raw quarterly contracts jump at each roll because the new front contract carries
more cost-of-carry (~**+0.6 %** per quarter). Stitching them raw would fabricate
false all-time highs. Because the raw contracts **do not overlap**, the pure
basis is not directly observable, so four methods are offered
(`config.roll_adjust_method`), newest contract always anchored to genuine price:

| method | what it removes | preserves | note |
|---|---|---|---|
| **`carry`** (default) | theoretical carry only (~0.6 %/roll) | % returns, **real moves** | most correct here; a rate assumption (`carry_annual_rate`, default 2.5 %) |
| `ratio` | measured seam jump (multiplicative) | % returns | **absorbs** the real weekend move into the factor |
| `difference` | measured seam jump (additive / Panama) | point changes | absorbs the real weekend move |
| `none` | nothing | — | raw stitch (contains carry steps) |

**Why `carry` is the default:** the measured seam gaps swing from −3.25 % to
+2.31 % because they contain **real weekend moves** (e.g. the Mar→Jun-2026 seam
is a real −882-point selloff, not a roll). The seam methods would bake those
into the "adjustment" and corrupt the historical curve. The carry model removes
only the ~0.6 % financing basis and leaves real price action intact.

The audit prints a per-seam breakdown (measured gap vs carry vs implied real
move) and a **verification block** (newest anchored raw, no negative prices, %
returns preserved, seam residuals). *Perfect* back-adjustment would need
overlapping contract data or a cash reference — neither is in the repo — so this
is the most-correct construction available and is clearly labelled.

`cleaned_fdax_1m.parquet` carries `adj_*` columns; `master_5m.parquet` carries
both raw OHLC (for display) and `adj_*` (for every cross-day comparison).

---

## 4. Session definitions (all configurable in `config.py`)

* **Cash session:** 09:00 → 17:30 Europe/Berlin.
* **Overnight session:** previous cash close (17:30) → current cash open (09:00),
  using whichever futures bars exist in between.
* **Observation cutoffs:** 09:15 / 09:30 / 10:00 / 10:30 / custom.
* **Opening windows:** first 5 / 15 / 30 / 60 minutes + the chosen cutoff.

---

## 5. How the ATH is defined (no leakage by construction)

```
prior_ath[D]              = max adjusted intraday high over sessions strictly < D
ath_at_open               = cash_open_adj > prior_ath[D]
distance_open_to_ath_pct  = (cash_open_adj - prior_ath) / prior_ath
ath_reached_overnight     = overnight_high_adj >= prior_ath[D]
ath_broken_first_{5,15,30}m = any cash bar in the window with adj_high > prior_ath
```

`prior_ath[D]` is an expanding max **shifted by one session**, so a session's own
high can never enter its own prior-ATH. This is unit-tested
(`tests/test_ath_and_leakage.py`).

---

## 6. How similarity is calculated

* **Robust scaling** (median / IQR) so outliers don't dominate.
* **Weighted blend of four feature blocks** (every weight configurable):
  * **Market context 30 %** — ATH proximity, overnight return/range, gap, open-vs-PDH
  * **First-bar structure 20 %** — return, size, body/range, close location, overnight-high break
  * **Post-spike path 30 %** — extension, pullback, overlap, efficiency, new-highs, **+ normalized path correlation** folded in
  * **Weakness structure 20 %** — lower-high timing, bearish-expansion timing, stall/level breaks, time-below-open
* **Path-shape matching**: paths are aligned at 09:00, rebased so the open is 0,
  expressed in points / % / prior-day-range / **ATR**, and compared by
  correlation / Euclidean / cosine / **DTW**. Only bars up to the cutoff are used.

**Three modes** — Mode 1 exact filter · Mode 2 pure similarity · **Mode 3
combined (default)**: broad *context* filter → rank the survivors by full opening-
path similarity (structure conditions become ranking preferences, not hard gates).

**Progressive matching** keeps sample size honest: it widens tolerances along a
recorded ladder (ATH tolerance → weakness timing → drop gap → drop regime) until
enough analogs exist, and **reports every relaxation** (Exact / Strong / Broad
counts). A warning fires when results rest on very few cases.

---

## 7. Outcomes (computed *after* the cutoff — never fed back into matching)

Returns at +15/30/60 min, 12:00, US open (15:30), cash close (17:30); MFE / MAE;
session-high/low times; breaks of the observed high/low; touches of ON high/low,
PDH, PDL, previous close; half / full gap fill; close vs open / prior-ATH; and an
explicit, editable **day classification** (`classifications.py`) into one of eight
labels (ATH gap-and-go, opening spike → bull flag, spike failure, partial fade,
full gap fill, complete reversal, high-level range, two-sided volatile).

---

## 7b. Price-action setups & conditional probabilities

`price_action.py` answers a different question from the analog search: *"this
exact setup — how often does the level actually come to me, given it hasn't
already?"* It uses **no indicators**. Every opening-pressure test is a
relationship between bar opens/highs/lows/closes, self-scaled by the bars' own
ranges, the way structure is read off a chart by eye.

```bash
python -m dax_analog_explorer.setup_report                          # default ATH-trap setup
python -m dax_analog_explorer.setup_report --close-beyond-prior-bar # add condition E
python -m dax_analog_explorer.setup_report --gap-max 0.9 --dates-only
python -m dax_analog_explorer.setup_report --direction up --below-pdl  # mirror it
```

The session table is cached under `data/processed/`, so the first run takes
~30 s and every later threshold tweak is instant (`--rebuild` to recompute).

### `PriceActionSpec` — the full parameter set

| group | parameter | default | meaning |
|---|---|---|---|
| **context** | `ath_tolerance_pct` | 0.50 | `\|open − prior_ath\| / prior_ath × 100 ≤` |
| | `require_new_ath_at_open` | False | `open > prior_ath` |
| | `require_open_above_pdh` | True | `open > previous_day_high` |
| | `require_open_below_pdl` | False | `open < previous_day_low` |
| | `gap_min_pct` | +0.30 | `(open − prev 17:30 close) / prev close × 100 ≥` |
| | `gap_max_pct` | +1.20 | upper bound — above ~0.6% the reversal branch disappears |
| | `on_location_min` / `_max` | 0.70 / — | `(open − ON_low) / (ON_high − ON_low)` |
| | `vol_regimes` | None | restrict to low / normal / high |
| **pressure** | `pressure_window_min` | 15 | bars 1–3 on a 5 m chart |
| | `direction` | "down" | "down" = selling pressure; "up" mirrors every test |
| | `require_close_beyond_open` | True | **(A)** close at +15 m below the open |
| | `close_location_max` | 0.33 | **(B)** `(close − low₁₅) / (high₁₅ − low₁₅) ≤` |
| | `require_close_beyond_prior_bar` | False | **(E)** a bar closed below the prior bar's low |
| | `require_monotonic_highs` | False | **(C)** `h1 > h2 > h3` |
| | `min_directional_bars` | None | **(D)** count of bear bars ≥ n |
| | `require_beyond_open_excursion` | False | **(F)** poked above the open, closed back below |
| **decision** | `decision_bar` / `bar_minutes` | 6 / 5 | bar 6 → 09:30 |
| **conditional** | `require_untested` | PDH, PDC, PDL | levels excluded from their own denominator once tested |
| **outcome** | `morning_end_min` | 180 | 12:00 Berlin |

Bar-1 direction is deliberately **not** a parameter: in this setup price
essentially always pokes above the open first, so requiring a bearish first bar
discards the "spike up then reject" days that belong to the pattern.

### Conditional probability

`conditional_report()` reports **P(level reached later | NOT reached by the
decision bar)** — a level already touched before bar 6 leaves its own
denominator, because you could not have entered ahead of it. That is what makes
the number conditional rather than unconditional, and it is usually the lower
(and more honest) figure.

`build_funnel()` shows how many sessions survive each parameter in order, so the
cost in sample size of every condition is visible before you trust a rate.

## 7c. Categorical context recognition

The ATH filter was a *threshold* ("within 0.25% of the high"). That is one value
of a more general question — **where are we, and how did we get here** — so
`context.py` replaces thresholds with named categories:

| dimension | categories |
|---|---|
| **location** | `AT_ATH` · `NEAR_ATH` · `BREAKOUT_UP` · `UPPER_RANGE` · `MID_RANGE` · `LOWER_RANGE` · `BREAKDOWN` · `AT_LOWS` |
| **swing** | `STRONG_BULL` · `BULL` · `RANGE` · `BEAR` · `STRONG_BEAR` |
| **ma_state** | `ABOVE_RISING` · `ABOVE_FALLING` · `BELOW_RISING` · `BELOW_FALLING` |
| **prior_day** | `TREND_UP` · `TREND_DOWN` · `REVERSAL_UP` · `REVERSAL_DOWN` · `BIG_RANGE` (news/shock) · `RANGE` · `INSIDE` · `OUTSIDE` |
| **gap** | `LARGE_UP` · `MODERATE_UP` · `SMALL_UP` · `FLAT` · `SMALL_DOWN` · `MODERATE_DOWN` · `LARGE_DOWN` |
| **open_loc** | `ABOVE_PDH` · `UPPER_PD` · `MID_PD` · `LOWER_PD` · `BELOW_PDL` |
| **overnight** | `TREND_UP` · `TREND_DOWN` · `RANGE` · `REVERSAL_UP` · `REVERSAL_DOWN` |
| **opening** | `DRIVE_UP` · `DRIVE_DOWN` · `REJECTION_UP` · `REJECTION_DOWN` · `RANGE_OPEN` |

Inputs are restricted to **OHLC and moving averages** — no oscillators, no
volume. Where a "big" or "small" judgement is needed the cut is a **rolling
percentile of the instrument's own prior history**, so `LARGE_UP` means the same
thing in 2003 and 2026. Everything is prior-only except `opening`, which is
clamped at the observation cutoff.

**Two definitions that matter.** A *rejection* requires a **meaningful
excursion** (≥35% of the opening window's own range) before price closes back
through the open — without that floor a one-tick poke counts and ~60% of all
sessions get mislabelled as rejections. The overnight classifier uses the same
give-back rule. `BIG_RANGE` (the news/shock footprint) is the top decile of the
prior-day range against its own trailing distribution.

```bash
python -m dax_analog_explorer.context_report --list      # every category
python -m dax_analog_explorer.context_report --profile   # label mix of the dataset
python -m dax_analog_explorer.context_report             # the ATH bull trap
python -m dax_analog_explorer.context_report --location AT_ATH NEAR_ATH --opening REJECTION_UP
python -m dax_analog_explorer.context_report --prior-day BIG_RANGE --gap SMALL_UP
```

The report prints the condition chain, a funnel showing the sample-size cost of
each category, the matching dates, and the same conditional-probability table as
`setup_report` — *P(level reached | not reached by the decision bar)*.

## 7d. Open Trend Continuation (OTC) — the one executable setup

`otc.py` is not another filter. `context.py` and `price_action.py` label a
session at **one fixed cutoff**; OTC is a **sequence** —

> open develops direction → genuine follow-through → controlled pullback →
> continuation trigger in the same direction

— so it is walked **bar by bar** and can fire at any time up to the deadline.

**No fitted thresholds and no indicators in the decision path.** Every condition
is a count or a structural comparison between bars. The only numbers are scope
choices: the bar interval, the trigger deadline (10:30 Berlin) and how many
entries a session may produce (2).

| # | Rule | Pure-price test |
|---|---|---|
| 1 | direction | ≥2 consecutive higher highs **and** higher lows from the cash open (mirrored for a bear) |
| 2 | follow-through | ≥2 **distinct** bars extend beyond bar 1's extreme — rejects "one big bar then nothing" without measuring bar size |
| 3 | not extended | *no test of its own* — a climax has not pulled back so it cannot trigger (rule 4), and an over-run leg fails on its own R:R arithmetic (rule 7) |
| 4 | pullback | the correction must not trade through the **leg origin** |
| 5 | trigger | **H1/H2** (bull) or **L1/L2** (bear) bar counting; entry is a stop order 1 tick beyond the signal bar |
| 6 | stop | 1 tick beyond the signal bar's extreme or the pullback extreme, whichever is structurally further |
| 7 | target | the **nearest untouched structural level** ahead: PDH / PDC / PDL / ONH / ONL / the session's own extreme |

R:R is **reported, not thresholded** — the TAKE/WAIT/PASS call on rules 1–5 is
mechanical, and the judgement stays with the trader.

### Bar counting

Bull: the pullback opens on a bar with a **lower high**; the first later bar whose
**high exceeds the prior bar's high** is **H1**; after another leg down, the next
one is **H2**. Bear mirrors it on lows. The count **resets** on a new leg extreme,
and that extreme is tracked **causally** — as the running max of the bars walked
so far, never from the whole window, or every post-reset H1 is mislabelled H2.

### Three states

| state | condition | output |
|---|---|---|
| **A** | rules 1–2 not met | `INVALID` → no trade |
| **B** | direction, no completed rotation | `DEVELOPING` → wait |
| **C** | rules 1–5 met | `ACTIVE` → take |

```bash
python -m dax_analog_explorer.otc_report --profile      # how many days reach A / B / C
python -m dax_analog_explorer.otc_report                # backtest, split by bar count
python -m dax_analog_explorer.otc_report --pdf otc.pdf  # annotated chart pack
python -m dax_analog_explorer.otc_report --date 2026-05-05   # the live sentence
```

### What the backtest found

Of 6,560 sessions the furthest state reached by 10:30 is **ACTIVE 47.3%**
(3,105 sessions), DEVELOPING 13.8%, INVALID 38.8%. Those 3,105 sessions produce
6,495 signals, of which 3,331 fill.

*Furthest reached* is the number that answers "does this setup show up". The
state on the **last bar** is a different question — it drops back to `DEVELOPING`
the bar after a trigger fires, because that bar counted nothing new — and
`OTCResult` carries both (`reached` and `state`). Reading a session as of an
earlier moment is what `--deadline` is for: `--date 2026-05-05 --deadline 60`
replays that day as it stood at 10:00 and returns `ACTIVE … Decision: TAKE`,
where the same day at the 10:30 default reports the trigger as already fired.

| cut | n filled | expectancy | t |
|---|---|---|---|
| all | 3,331 | **−0.180R** | −12.51 |
| in-sample (pre-2017) | 1,932 | −0.202R | −10.15 |
| out-of-sample (2017+) | 1,399 | −0.149R | −7.31 |
| H1/L1 | 2,271 | −0.192R | −10.75 |
| H2/L2 | 981 | **−0.157R** | −6.17 |

The mandate's premise — *the second attempt is the reliable one* — holds
**directionally**: H2/L2 beats H1/L1 on expectancy (−0.157R vs −0.192R), win rate
(54.2% vs 49.3%) and profit factor (0.61 vs 0.52). It is not enough to reach zero.

**Why it loses is structural, not directional.** The nearest untouched level
(rule 7) is the session's own extreme in 4,787 of 6,495 signals, so the median
target sits **7.3 points** away against **18.6 points** of structural risk —
median R:R **0.38**, and only 16% of signals offer 1.0 or better. Selecting for
better R:R makes it *worse*, monotonically:

| filter | n filled | expectancy | t |
|---|---|---|---|
| all | 3,276 | −0.191R | −14.68 |
| R:R ≥ 1.0 | 487 | −0.197R | −3.57 |
| R:R ≥ 1.5 | 180 | −0.237R | −2.28 |
| R:R ≥ 2.0 | 91 | −0.307R | −1.97 |
| R:R ≥ 3.0 | 34 | −0.440R | −1.71 |

A wide R:R is wide because the structure is stretched, and stretched structure is
exactly what fails. The best surviving cell (H2/L2 with R:R ≥ 1) is −0.184R
in-sample −0.110R, **out-of-sample −0.275R** — it gets worse out of sample, which
is the opposite of what a real edge does.

**Three corrections found by reading the output rather than trusting it:**

* the leg carried on each signal is the leg **as it stood when that signal
  fired** — a session can build a bear leg, fail, then build a bull one, so the
  leg standing at the deadline is not the leg the signal came from;
* an entry order is **cancelled** once price trades through the level where its
  stop would sit. Leaving it working and filling it later booked a result against
  a stop that had already been violated — **1,404 of 4,735 "trades"** were
  setups no trader would still have had on. Removing them moved expectancy from
  −0.216R to −0.180R and is the honest number;
* the state machine reported the **last bar's** state as the session's state, so
  a day that counted an H1 at 10:00 and then simply ran said `DEVELOPING → WAIT`
  at 10:30 and never mentioned its own trigger. `--profile` inherited the same
  field and was answering "did a trigger fire on the very final bar" — which is
  why it read 8.4% instead of 47.3%.

**The finding, plainly: OTC as specified does not have a positive expectancy on
this data.** That is a result, not a failure of the build — it tests the setup as
actually written, with structural stops and targets, rather than the 09:30
at-market proxy the mandate bans (which lost more, −0.128R with no pullback and
no trigger). The business rule *"if OTC does not develop, DAX gives me nothing
today"* remains the correct boundary; what the numbers argue against is the
**exit geometry**, not the pattern recognition.

## 8. Running it

Every command is run **from the repository root** (`config.Paths` resolves paths
from there) and every timing below was measured on this dataset, not estimated.

### Step 1 — build the data (once)

```bash
pip install -r requirements.txt
python -m dax_analog_explorer.preprocess --cutoff 10:30
#   options: --roll-method {carry,ratio,difference,none}   --no-audit
```

**~1m15s.** Converts DAX 5m Chicago→Berlin, merges the two sources, back-adjusts
the rolls, and writes `data/processed/`:

```
cleaned_dax_5m.parquet  cleaned_fdax_1m.parquet  master_5m.parquet
daily_features.parquet  opening_path_features.parquet  audit_report.{json,txt}
```

That directory is git-ignored — delete it and re-run any time. **Nothing else
works until this has run**, because every command below reads
`master_5m.parquet` + `daily_features.parquet`.

### Step 2 — everything else

| command | what it gives you | time |
|---|---|---|
| `otc_report --date 2026-05-05` | the live sentence for one session: state, direction, follow-through, every trigger counted with its stop / target / R:R, and the decision | 4s |
| `otc_report --date 2026-05-05 --deadline 60` | the same day **as it stood at 10:00** — how to read a morning bar by bar | 4s |
| `otc_report --profile` | furthest state reached across history: ACTIVE 47.3% / DEVELOPING 13.8% / INVALID 38.8% | 12s |
| `otc_report` | the full backtest, split by H1/H2 and L1/L2, in- vs out-of-sample | 18s |
| `otc_report --pdf otc.pdf --pdf-max 24` | the annotated chart pack: context above, the traded window zoomed below | 1m02s |
| `context_report --list` | every context category and the flag that selects it | 3s |
| `context_report --profile` | the label mix of the whole dataset | 26s |
| `context_report --location AT_ATH NEAR_ATH --opening REJECTION_UP` | a category scan: condition chain, funnel, dates, conditional probabilities | 41s |
| `setup_report` | the geometric setup scan + *P(level reached \| not reached by the decision bar)* | 44s |
| `continuation_report` | the opening-drive continuation backtest (the 09:30 at-market entry OTC bans) | 36s |
| `streamlit run dax_analog_explorer/app.py` | **the app** — read one session, or filter by context and see what OTC did inside it | — |

Prefix each with `python -m dax_analog_explorer.` — e.g.
`python -m dax_analog_explorer.otc_report --profile`.

`context_report` and `setup_report` cache their session tables into
`data/processed/` on first use, so a repeated scan with the same window is much
faster; `setup_report --rebuild` forces a fresh one.

Useful `otc_report` knobs: `--deadline 90` (minutes after the open — 90 = 10:30),
`--max-entries 2`, `--cost 2.0` (points round trip), `--split 2017-01-01`,
`--csv trades.csv`.

### The app

Two tabs, and deliberately only two.

**① Read a session.** Pick a date and drag *Read the session as of* to the moment
you want. You get the OTC state as it stood then, every trigger counted with its
stop / target / R:R and what it did, the session's context labels, and the chart
with the anatomy drawn on it — leg origin, leg extreme, the counted H1/H2 or
L1/L2 bar, trigger, stop, target and the exit. Moving the slider replays the
morning: the same day reads `ACTIVE · TAKE` at 10:00 and *"the trigger already
fired"* at 10:30.

**② Context filter.** Choose categories across the eight dimensions — leave a box
empty to ignore it — and the app answers two questions in order: *does the setup
even appear in this context* (share of matching sessions that produce a trigger,
against 47.3% across all history) and *what did it do* (expectancy, t-stat, win
rate, profit factor, split H1/L1 vs H2/L2 vs H3+, and in- vs out-of-sample). Then
the matching dates, downloadable as CSV, and a chart of any one of them.

The sidebar holds scope only — the date, the as-of time, the entry cap and the
cost. Nothing in it tunes the setup, because the setup has nothing to tune.

The analog-similarity search that used to be the app is gone. Its engines
(`search.py`, `similarity_engine.py`, `path_matching.py`) remain in the package
and are still covered by tests, but nothing in the app depends on them.

### The data ends 2026-05-19

There is no live feed. `--date` past the last session fails on purpose:

```
$ python -m dax_analog_explorer.otc_report --date 2026-06-01
no session on 2026-06-01 — the data covers 2000-05-29 .. 2026-05-19.
  (a later date needs its bars appended to a FDAX_1min_*.csv and
   'python -m dax_analog_explorer.preprocess' re-run)
```

To read a morning that is not in the data yet, append its 1-minute bars to a
`FDAX_1min_*.csv` in the same format (see §2), re-run `preprocess`, then call
`otc_report --date <today> --deadline <minutes since 09:00>`.

---

## 9. Project structure

```
dax_analog_explorer/
  config.py            single Config dataclass (paths, TZ, sessions, weights, thresholds)
  data_loader.py       load + standardize + Chicago→Berlin / Berlin localize + dedup
  data_audit.py        timezone detection, coverage, gaps, rolls, verification report
  session_builder.py   sessionize, FDAX 1m→5m, roll adjustment, continuous master, daily OHLC
  ath_engine.py        prior_ath (strict, no leakage), ATH flags, first-window breaks
  feature_engineering.py  daily_features table (prior-only rolling stats, gaps, regime)
  pattern_features.py  opening-path / first-bar / post-spike / weakness features (cutoff-clamped)
  similarity_engine.py robust scaling, weighted blocks, 3 modes, progressive matching
  path_matching.py     align-at-open paths, correlation/euclidean/cosine/DTW
  outcome_engine.py    post-cutoff outcomes, MFE/MAE, gap fills, touches
  classifications.py   explicit editable day-type rules
  charts.py            plotly: OTC session chart (leg/pullback/trigger/stop/target), candles, overlays
  reports.py           interpreted query, distribution stats, dates-only
  search.py            orchestrator (reference → analogs → outcomes → explanations)
  context.py           categorical context + opening-pattern recognition (OHLC + MAs)
  context_report.py    CLI for category-based scans
  price_action.py      pure-geometry setup filtering + conditional probabilities
  setup_report.py      CLI for setup scan + conditional-probability report
  otc.py               Open Trend Continuation: H1/H2-L1/L2 bar counting + state machine
  otc_report.py        CLI: OTC profile, event-driven backtest, annotated chart pack
  continuation.py      opening-drive continuation study + backtest harness (IS/OOS)
  continuation_report.py  CLI for the continuation grid
  chart_pdf.py         PDF chart packs: single session, multi-session Xetra view, OTC anatomy
  preprocess.py        command-line pipeline
  app.py               Streamlit app: read a session · context filter -> OTC performance
tests/                 pytest: ATH, leakage, DST, prev-day, cutoff, rolls, gap-fill
data/processed/        generated parquet artifacts (git-ignored)
```

---

## 10. Tests

```bash
python -m pytest        # 56 tests, ~2.4s
```

Covers ATH calculation, no-future-leakage, Europe/Berlin + Chicago DST,
previous-day levels, observation-cutoff enforcement, contract-roll / back-
adjustment handling, gap-fill (including "touching the current-day open is
**not** a full overnight gap fill"), categorical context labelling (a one-tick
poke is not a rejection), and the OTC engine: H1/H2 and L1/L2 counting on
hand-built bars, the count resetting on a new leg extreme, a climax reaching
`DEVELOPING` but never `ACTIVE`, a pullback through the leg origin invalidating,
the 3rd signal in a session being `NOT_TAKEN`, an untriggered entry not counting
as a loss, an entry cancelled when its stop level breaks first, and the stop
winning on a bar that contains both stop and target, and a fired trigger still
being reported when a later quiet bar drops the state back to `DEVELOPING`.

---

## 11. Research rules honoured

* Never uses future bars for matching (cutoff-clamped features **and** paths).
* Rolling / regime features use **prior sessions only**.
* Futures ATH runs on a **roll-adjusted continuous** series — raw contract levels
  are never mixed in a `max()`.
* Data-quality problems are **reported, not filled** (see the audit + warnings).
* Output is framed as a **historical conditional distribution** with sample size,
  not a claim that analogs predict the current session.
* Every threshold, weight, session definition and roll method is **configurable**.
* The OTC decision path contains **no fitted values at all** — only counts,
  structural comparisons, and the scope choices (bar size, deadline, entry cap).
* Negative results are reported as findings. OTC's −0.180R expectancy is stated
  with its t-statistic and its out-of-sample split rather than filtered until it
  looks profitable.

---

## 12. Known limitations

* No overlapping FDAX contract data → roll basis is estimated (carry model by
  default); the deep-2000s absolute levels are not required for recent-era ATH
  decisions and are left at genuine levels.
* The 42-day DAX↔FDAX gap means prior-ATH for the first FDAX sessions cannot see
  any high made inside that window (flagged in the audit).
* 1-minute FDAX precision is available 2024-11 onward; cross-era similarity uses
  the common 5-minute grid (the flagship "last 10 years" query spans both eras).
* OTC fills are simulated on 5-minute bars, so a bar containing both the stop and
  the target is resolved pessimistically (stop wins) rather than by sequence —
  1-minute resolution would settle those, but only for 2024-11 onward.
* The OTC target is the nearest untouched level from a fixed set (PDH/PDC/PDL/
  ONH/ONL/session extreme). A trader reading a chart would sometimes skip a level
  as insignificant; the engine never does, and that is why the session extreme
  dominates the target distribution.
