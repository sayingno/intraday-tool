# DAX Analog Day Explorer

Describe what today's DAX session is doing, pick an **observation cutoff**
(e.g. `10:30` Berlin), and the tool finds **historically similar trading days**,
shows their intraday charts, and reports **what happened after the cutoff** — as
a *historical conditional distribution*, never a prediction, always with the
sample size shown.

The first worked example is the pattern:

> **ATH gap-up → opening bullish spike → weak follow-through → high-level stall →
> first bearish weakness before ~10:30.**

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

## 8. Running it

```bash
pip install -r requirements.txt

# 1) one-time preprocessing  ->  data/processed/*.parquet + audit_report.txt
python -m dax_analog_explorer.preprocess --cutoff 10:30
#    options: --roll-method {carry,ratio,difference,none}  --no-audit

# 2) the app
streamlit run dax_analog_explorer/app.py
```

In the app: pick a **reference date** and **cutoff**, choose **Mode 3**, hit
**Find analogs** — or click **“Load the worked example”** to reproduce the ATH
gap-up → weak-follow-through query at 10:30 over the last 10 years and return the
20 most similar days. The **Dates only** tab answers *"just show me the dates."*

Artifacts written to `data/processed/` (git-ignored, regenerate any time):
`cleaned_dax_5m.parquet`, `cleaned_fdax_1m.parquet`, `master_5m.parquet`,
`daily_features.parquet`, `opening_path_features.parquet`, `audit_report.{json,txt}`.

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
  charts.py            plotly candles + aligned overlays + median band (observed vs outcome shaded)
  reports.py           interpreted query, distribution stats, dates-only
  search.py            orchestrator (reference → analogs → outcomes → explanations)
  price_action.py      pure-geometry setup filtering + conditional probabilities
  setup_report.py      CLI for setup scan + conditional-probability report
  preprocess.py        command-line pipeline
  app.py               Streamlit MVP
tests/                 pytest: ATH, leakage, DST, prev-day, cutoff, rolls, gap-fill
data/processed/        generated parquet artifacts (git-ignored)
```

---

## 10. Tests

```bash
python -m pytest        # 21 tests, ~0.4s
```

Covers ATH calculation, no-future-leakage, Europe/Berlin + Chicago DST,
previous-day levels, observation-cutoff enforcement, contract-roll / back-
adjustment handling, and gap-fill (including "touching the current-day open is
**not** a full overnight gap fill").

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

---

## 12. Known limitations

* No overlapping FDAX contract data → roll basis is estimated (carry model by
  default); the deep-2000s absolute levels are not required for recent-era ATH
  decisions and are left at genuine levels.
* The 42-day DAX↔FDAX gap means prior-ATH for the first FDAX sessions cannot see
  any high made inside that window (flagged in the audit).
* 1-minute FDAX precision is available 2024-11 onward; cross-era similarity uses
  the common 5-minute grid (the flagship "last 10 years" query spans both eras).
