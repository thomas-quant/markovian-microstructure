<!-- e4006855-646f-43bc-8f5d-f463db9d819a 85c1725e-e5f2-4b54-b60e-78788f6cdde5 -->
# PCX Model: Comprehensive Research Pipeline

## Overview

Build a multi-dimensional analysis framework to understand what makes the PCX model work, when it fails, and how to make it actionable for trading.

## Phase 1: Sweep vs Fail Factor Analysis

**Goal**: Identify what causes continuation (sweep) vs failure

### 1.1 Volume Analysis

- Extract volume characteristics for each prediction
- Compare breakout volume vs average volume (5, 10, 20-period)
- Volume ratio at state transitions vs consolidation
- Test if high-volume breakouts are more likely to sweep

### 1.2 Range & Volatility Analysis  

- Calculate ATR (14-period) for each candle
- Measure candle size relative to recent history (Z-score of ranges)
- Test if larger breakout candles → higher sweep probability
- Analyze body-to-range ratio impact

### 1.3 Session Context Analysis

- Leverage existing session tags in [`Research/pdx/data/nq_1m.parquet`](Research/pdx/data/nq_1m.parquet)
- Breakdown accuracy by session: ASIA, LONDON, NYAM, LUNCH, PM, OTHER
- Identify if sweeps are more common during high-liquidity sessions

### 1.4 Macro Event Analysis

- Load economic events from [`Misc/Forexfactory/economic_events.parquet`](Misc/Forexfactory/economic_events.parquet)
- Tag candles that occur within 1-hour window of high-impact events (NFP, FOMC, CPI)
- Test if model accuracy degrades around macro events (filter them out)

**Output**: `Research/pdx/sweep_factors.py` - comprehensive factor analysis script

---

## Phase 2: Multi-Timeframe Robustness Testing

**Goal**: Validate if PCX is timeframe-agnostic or just a daily phenomenon

### 2.1 Resample NQ Data

Generate datasets from [`Research/pdx/data/nq_1m.parquet`](Research/pdx/data/nq_1m.parquet):

- 5-minute
- 15-minute  
- 1-hour
- 4-hour
- Daily (existing)

### 2.2 Run PCX Backtest on Each Timeframe

- Apply identical logic from [`Research/pdx/main.py`](Research/pdx/main.py)
- Compare filtered accuracy across timeframes
- Identify if lower timeframes have more noise (lower accuracy)

**Output**: `Research/pdx/multi_timeframe.py` - timeframe comparison analysis

---

## Phase 3: Multi-Security Signal Validation  

**Goal**: Use ES + NQ alignment as a confirmation filter

### 3.1 Load ES Data

- Use [`Research/ict am/es_1m.parquet`](Research/ict am/es_1m.parquet)(Research/ict am/es_1m.parquet)(Research/ict am/es_1m.parquet)(Research/ict am/es_1m.parquet) (has session tags)
- Align ES and NQ on matching timestamps

### 3.2 Dual-Asset PCX Logic

- Run PCX state machine on both ES and NQ independently
- Flag "aligned signals": both predict same direction (High or Low target)
- Flag "divergent signals": ES and NQ predict opposite directions

### 3.3 Measure Alignment Edge

- Compare accuracy of:
  - NQ-only signals
  - ES-only signals  
  - Aligned signals (both agree)
  - Divergent signals (filter these out)

**Output**: `Research/pdx/multi_security.py` - ES+NQ alignment analysis

---

## Phase 4: Timing Analysis (When Do Sweeps Occur?)

**Goal**: Discover the average time to sweep after a prediction

### 4.1 Time-to-Target Measurement

For each successful prediction, calculate:

- Number of candles until target hit
- Actual time duration (minutes) until sweep
- Distribution: median, mean, P25, P75, P90

### 4.2 Failure Timing

For failed predictions:

- When was the opposite extreme hit instead?
- Distribution of "time until invalidation"

### 4.3 Intraday Timing Patterns

- Break down time-to-sweep by session
- Identify if NYAM sweeps faster than ASIA

**Output**: `Research/pdx/timing_analysis.py` - timing distribution analysis

---

## Phase 5: Discovery of Related Markov Models

**Goal**: Generalize the pattern and discover similar state-based models

### 5.1 Swing-Based PCX

- Adapt PCX logic to use swing highs/lows instead of sequential candles
- Leverage existing swing detection from [`Research/SMT research/timing_analysis.py`](Research/SMT research/timing_analysis.py)(Research/SMT research/timing_analysis.py)(Research/SMT research/timing_analysis.py)(Research/SMT research/timing_analysis.py) (lines 144-165)
- Test: "If close > previous swing high → expect current high to be taken"

### 5.2 Inside Bar Expansion Model

- Detect inside bars (current range fully within previous range)
- Rule: First breakout direction predicts continuation to sweep that extreme

### 5.3 Engulfing Candle Model

- Detect engulfing candles (current range fully encompasses previous)
- Rule: Engulfing direction predicts next candle continues that direction

### 5.4 Three-Candle State Model

- Extend PCX to 3-state Markov chain instead of 2-state
- States: Strong Bull, Weak Bull, Neutral, Weak Bear, Strong Bear
- Transitions based on consecutive breakouts

**Output**: `Research/pdx/markov_models.py` - collection of related pattern models

---

## Phase 6: Synthesis & Dashboard

**Goal**: Consolidate findings into actionable insights

### 6.1 Build Factor Model

- Combine strongest factors from Phase 1 into a scoring system
- Score = f(volume_ratio, ATR_zscore, session, alignment, time_of_day)
- Test if high-score predictions have 80%+ accuracy

### 6.2 Create Results Dashboard

- Generate summary markdown: `Research/pdx/results.md`
- Include:
  - Best timeframe for PCX
  - Best session for PCX  
  - ES+NQ alignment edge
  - Timing distributions
  - Factor importance rankings

### 6.3 Actionable Trading Rules

Document final ruleset:

```
ENTRY: PCX signal with score > threshold
CONFIRMATION: ES+NQ aligned  
TIMING: Expect sweep within X candles (P75)
INVALIDATION: Opposite extreme hit
```

**Output**: `Research/pdx/results.md` + `Research/pdx/factor_model.py`

---

## Implementation Files Structure

```
Research/pdx/
├── main.py (existing - core PCX logic)
├── pcx.md (existing - documentation)
├── data/
│   └── nq_1m.parquet (existing)
├── sweep_factors.py (NEW - Phase 1)
├── multi_timeframe.py (NEW - Phase 2)  
├── multi_security.py (NEW - Phase 3)
├── timing_analysis.py (NEW - Phase 4)
├── markov_models.py (NEW - Phase 5)
├── factor_model.py (NEW - Phase 6)
└── results.md (NEW - Phase 6)
```

---

## Key Reusable Components

**Session Tagging**: Already exists in NQ/ES parquet data

**Swing Detection**: Reuse from [`Research/SMT research/timing_analysis.py:144-165`](Research/SMT research/timing_analysis.py)(Research/SMT research/timing_analysis.py)(Research/SMT research/timing_analysis.py)(Research/SMT research/timing_analysis.py)

**Economic Events**: Load from [`Misc/Forexfactory/economic_events.parquet`](Misc/Forexfactory/economic_events.parquet)

**ES Data**: Available at [`Research/ict am/es_1m.parquet`](Research/ict am/es_1m.parquet)(Research/ict am/es_1m.parquet)(Research/ict am/es_1m.parquet)(Research/ict am/es_1m.parquet)

### To-dos

- [ ] Implement sweep vs fail factor analysis (volume, ATR, session, macro)
- [ ] Test PCX robustness across 5m, 15m, 1h, 4h timeframes
- [ ] Build ES+NQ alignment filter and measure accuracy improvement
- [ ] Measure time-to-sweep distributions and failure timing
- [ ] Discover and test related candle-based Markov models
- [ ] Build factor scoring model and create results dashboard