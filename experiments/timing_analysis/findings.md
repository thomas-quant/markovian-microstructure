# PCX + ICT Aligned Signal: Timing Analysis Findings

**Instrument:** NQ (Nasdaq 100 Futures)
**Period:** 2020-08-31 to 2025-11-21 (1,627 daily bars)
**Dataset:** 185 aligned days where both PCX and ICT agree on direction, PCX wick filter passed

---

## What Was Tested

When the PCX expansion model and the ICT daily bias state machine both signal the same direction on the same day — and the PCX signal candle passes the abnormal wick filter — we asked: **at what time of day does the predicted target actually get taken?**

For LONG signals, the target is the prior day's high. For SHORT signals, the prior day's low. Time is measured from the futures session open at 6pm ET the prior calendar day.

---

## Hit Rates

| Direction | Hits | Total | Rate |
|-----------|------|-------|------|
| LONG      | 83   | 105   | 79.0% |
| SHORT     | 62   | 80    | 77.5% |
| **Combined** | **145** | **185** | **78.4%** |

These match the results from `combined_test.py` exactly, validating the alignment logic.

---

## When Does the Target Get Hit?

### Time-to-Target Distribution (from 6pm ET session open)

| Percentile | Minutes | Clock Time ET |
|------------|---------|---------------|
| P25        | 485     | 02:05 AM      |
| **Median** | **630** | **04:30 AM**  |
| Mean       | 742     | 06:22 AM      |
| P75        | 931     | 09:31 AM      |
| P90        | 1,135   | 12:54 PM      |

**The median target hit occurs at 4:30 AM ET** — squarely in the London session. Half of all targets are taken before New York even opens.

### By Direction

| Direction | Mean (min) | Mean Clock | Median (min) | Median Clock |
|-----------|-----------|------------|-------------|-------------|
| LONG      | 726       | 06:06 AM   | 621         | 04:21 AM    |
| SHORT     | 764       | 06:44 AM   | 664         | 05:04 AM    |

Longs resolve slightly faster than shorts — median difference of ~43 minutes. Both are firmly in the pre-NY window.

### Session Breakdown

| Session | Hits | % of All Hits |
|---------|------|---------------|
| OTHER   | 75   | 51.7%         |
| LONDON  | 42   | 29.0%         |
| NYAM    | 20   | 13.8%         |
| ASIA    | 4    | 2.8%          |
| LUNCH   | 2    | 1.4%          |
| PM      | 2    | 1.4%          |

**Key insight:** "OTHER" dominates because the named session windows in the data are narrow (e.g. London is tagged as 02:00-04:59 ET only). Many targets that get hit at 05:00-09:29 ET — the gap between London close and NYAM open — fall into OTHER. Combined with the actual London window, **~81% of targets are hit before 9:30 AM ET**.

NYAM captures 13.8%, and the afternoon sessions are negligible. **By lunch, 97.2% of targets that will be hit have already been hit.**

---

## Overshoot

When the target is hit, how far past it does price run?

| Metric | Value |
|--------|-------|
| Mean overshoot | 143.33 points |
| Mean overshoot (% of prior range) | 63.5% |
| Median overshoot (% of prior range) | 36.1% |

The distribution is right-skewed — most overshoots are moderate (median 36% of prior day's range), but some blow-through days push the mean to 63.5%. This means the typical day runs about a third of yesterday's range past the target level.

---

## Failure Analysis

40 out of 185 aligned days (21.6%) failed to hit the target.

| Failure Type | Count | % of Failures |
|-------------|-------|---------------|
| Inside bar (no breakout either way) | 28 | 70.0% |
| Opposite target hit instead | 12 | 30.0% |
| Broke toward target but fell short | 0 | 0.0% |

**The dominant failure mode is the inside bar** — the market simply doesn't expand beyond the prior day's range in either direction. This accounts for 70% of all failures.

When the model does fail directionally (30% of failures, or 6.5% of all aligned days), it's because the opposite side ran instead — a complete reversal.

**Zero cases** of "it moved in the right direction but didn't quite make it." The target is either clearly hit or clearly missed via consolidation or reversal.

---

## Range & Volatility: Hits vs Misses

| Metric | Hits | Misses |
|--------|------|--------|
| Mean day range (% of close) | 1.864% | 1.404% |
| Mean range/ATR ratio | 1.070 | 0.813 |

Hit days have measurably larger ranges (1.86% vs 1.40%) and above-average ATR ratios (1.07x vs 0.81x). **Misses cluster on below-average range days** — the market doesn't expand enough to reach the target.

This is a strong hint for the momentum analysis: candle strength / volatility on the signal day may be a useful pre-filter.

---

## Implications

1. **Timing edge is pre-NY**: If trading this signal, the window is 6pm ET to ~9:30 AM ET. Most targets resolve overnight or in London. Sitting through the NY session waiting for a target hit is rarely productive.

2. **Inside bars are the enemy**: 70% of failures are consolidation days. Any filter that identifies low-expansion days in advance (low ATR, tight prior ranges, macro event lulls) could cut false signals.

3. **Binary outcomes**: The target either gets clearly hit (with 36% median overshoot) or the day consolidates. There's no "almost got there" category — which is actually clean from a trading perspective.

4. **Volatility predicts success**: Hit days run 1.07x ATR vs 0.81x for misses. This sets up the momentum experiment: does the signal candle's strength predict whether the target day will have enough range?

---

## Next Steps

- **Momentum / candle strength analysis**: Measure body-to-range ratio, close position, range-vs-ATR of the signal candle. Can we filter out the inside-bar failures before they happen?
- **Regime analysis**: Does the 79% hit rate hold across bull markets, bear markets, and chop? Rolling accuracy over time.
- **Refined session timing**: Re-tag the "OTHER" hits with more granular windows (pre-London, London-NY gap, etc.) to get a sharper timing picture.
