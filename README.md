# PCX & ICT Daily Bias Research

Statistical analysis of two independent daily-timeframe models on **NQ (Nasdaq 100 Futures)** using 1-minute data resampled to daily bars.

- **Instrument:** NQ Futures
- **Data:** 1,627 daily bars (2020-08-31 to 2025-11-21)
- **Source:** `data/nq_1m.parquet` (1-minute bars, resampled to daily)

---

## Models

### PCX Expansion Model (`main.py`)

A state machine that tracks whether the prior day closed above or below the previous day's high/low, then predicts the next day's directional expansion target.

### ICT Daily Bias (`ict_daily_bias.py`)

A swing-point state machine implementing the ICT daily bias methodology. Identifies bullish/bearish active states through swing high/low structure and tests whether the bias predicts previous-day liquidity runs.

### Combined Test (`combined_test.py`)

Tests whether the two models are redundant or stackable by measuring signal correlation, agreement accuracy, and whether combining them improves hit rates.

---

## Results

### PCX Expansion Model

| Metric | Value |
|---|---|
| Total predictions | 1,624 |
| Overall accuracy | 63.85% |
| Filtered accuracy (no abnormal wicks) | 71.19% |
| High target accuracy | 68.30% (n=839) |
| Low target accuracy | 59.11% (n=785) |
| Bullish state accuracy | 65.30% (n=928) |
| Bearish state accuracy | 61.93% (n=696) |

### ICT Daily Bias

**State distribution:**

| State | Days | % |
|---|---|---|
| BULLISH_ACTIVE | 312 | 19.2% |
| BULLISH_WATCH | 429 | 26.4% |
| BEARISH_ACTIVE | 236 | 14.5% |
| BEARISH_WATCH | 474 | 29.1% |
| NEUTRAL | 176 | 10.8% |

**Baseline (unconditional, all days):**

| Metric | Rate |
|---|---|
| P(today's high > yesterday's high) | 54.06% |
| P(today's low < yesterday's low) | 45.57% |

**Hypothesis test (alpha = 0.05):**

| Signal | N | Hits | Rate | p (vs 50%) | p (vs baseline) | Significant |
|---|---|---|---|---|---|---|
| Bullish active | 312 | 199 | 63.78% | 0.000001 | 0.000318 | Yes |
| Bearish active | 236 | 145 | 61.44% | 0.000267 | 0.000001 | Yes |

**Verdict:** Both signals beat baseline -- methodology **not falsified**.

### Combined Test

**Signal correlation:**

- Days where both models signal: 548 / 1,624 (33.7%)
- Agreement rate: 48.0%
- Pearson r = -0.0452, p = 0.2913 (no significant correlation)
- Chi-squared independence test: p = 0.3316, Cramer's V = 0.0415 -- **signals are independent**

**Accuracy by signal combination:**

| ICT | PCX | N | Ran high rate | Ran low rate |
|---|---|---|---|---|
| LONG | HIGH | 155 | 73.5% | 24.5% |
| LONG | LOW | 157 | 54.1% | 51.0% |
| SHORT | HIGH | 128 | 57.0% | 52.3% |
| SHORT | LOW | 108 | 25.0% | 72.2% |

**Standalone vs combined hit rates:**

| Configuration | Long/High Rate | N | Short/Low Rate | N |
|---|---|---|---|---|
| ICT alone | 63.8% | 312 | 61.4% | 236 |
| PCX alone (filtered) | 74.8% | 497 | 66.7% | 402 |
| **Combined (agree + filtered)** | **79.0%** | 105 | **77.5%** | 80 |

**Statistical tests (combined vs standalone):**

| Comparison | p-value | Significant |
|---|---|---|
| Combined LONG vs ICT alone | 0.000533 | Yes |
| Combined LONG vs PCX alone | 0.190814 | No |
| Combined SHORT vs ICT alone | 0.001704 | Yes |
| Combined SHORT vs PCX alone | 0.023684 | Yes |

---

## Key Takeaways

1. **Both models independently beat baseline** at statistically significant levels.
2. **The signals are independent** (Chi-squared p = 0.33) -- they are not measuring the same thing.
3. **Combining both models when they agree boosts accuracy** to ~79% long and ~78% short, but at the cost of fewer trading days (105 and 80 respectively).
4. **PCX is the stronger standalone model** (74.8% filtered high target) compared to ICT (63.8% bullish active).
5. The combined signal significantly beats ICT alone on both sides, and significantly beats PCX alone on the short side (p = 0.024).

---

## Usage

```bash
# PCX Expansion backtest
python main.py

# ICT Daily Bias falsifiable test
python ict_daily_bias.py

# Combined model comparison
python combined_test.py
```

## Requirements

- Python 3.10+
- pandas
- numpy
- scipy
