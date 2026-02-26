# PCX Model: Previous Candle Expansion (Symmetric State Machine)

The PCX model is a state-based Markov model that predicts the directional bias of the next candle by tracking the **Expansion State** of the market. It recognizes both bullish and bearish momentum shifts and their subsequent failures.

## Core Mechanics: The Expansion State

The model maintains a persistent **State** (Bullish or Bearish) based on the most recent breakout of a previous candle's range.

| Event | State Transition |
|-------|------------------|
| Close > Previous High | → **Bullish** |
| Close < Previous Low  | → **Bearish** |
| Close inside range    | → *State unchanged* |

---

## Prediction Rules

Given the **current state** (from prior candle's outcome), predict the next candle's target based on the current close.

### In Bullish State (compare to Previous High):

```
if close[0] > high[-1]:
    expect high[1] > high[0]    # Target High (continuation)
    
elif close[0] < high[-1]:
    expect low[1] < low[0]      # Target Low (failure to expand)
```

### In Bearish State (compare to Previous Low):

```
if close[0] < low[-1]:
    expect low[1] < low[0]      # Target Low (continuation)
    
elif close[0] > low[-1]:
    expect high[1] > high[0]    # Target High (failure to expand)
```

**Key Insight**: The reference level changes based on state:
- Bullish → compare close to **previous high**
- Bearish → compare close to **previous low**

---

## Execution Flow

```
1. ENTER candle with prior state
2. PREDICT target using state's rules
3. UPDATE state based on this candle's close (for next iteration)
```

This order is critical: the state used for prediction is the state we **entered** with, not the state after the current candle's outcome.

---

## Exceptions & Filters

### The "Abnormal Wick" Exception
If the current candle has an **abnormal wick** (defined as either the upper or lower wick exceeding 40% of the total candle range), the principle is considered unreliable. Large wicks signify rejection or high volatility that disrupts the expansion/failure logic.

---

## Backtest Results (NQ Daily Data)

| Metric | Result |
|--------|--------|
| **Total Candles with Predictions** | 1,624 |
| **Overall Accuracy** | 63.85% |
| **Filtered Accuracy (No Abnormal Wicks)** | **71.19%** |
| **High Target Accuracy** | 68.30% (n=839) |
| **Low Target Accuracy** | 59.11% (n=785) |
| **Bullish State Accuracy** | 65.30% (n=928) |
| **Bearish State Accuracy** | 61.93% (n=696) |

### Key Findings
- **Filtered Accuracy of 71%+** makes this a robust directional bias indicator.
- **Bullish state dominance** (928 vs 696 candles) reflects NQ's inherent upward bias.
- **Wick filtering** improves accuracy by ~7%, confirming the model's sensitivity to rejection candles.
