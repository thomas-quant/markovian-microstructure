"""
ICT Daily Bias — State Machine Implementation + Falsifiable Test

The claim under test:
    When the bias state machine is BULLISH_ACTIVE, the next trading day's
    high will exceed the previous day's high at a rate significantly
    greater than chance (>50%).

    When BEARISH_ACTIVE, the next trading day's low will undercut the
    previous day's low at a rate significantly greater than chance (>50%).

Null hypothesis (H0):
    The bias signal has no predictive power. The rate of "previous day
    high/low violated" is <= 50% regardless of bias state.

Alternative hypothesis (H1):
    The bias signal predicts the direction of the liquidity run at a
    rate > 50%.

The test is falsifiable: if the hit rates are not statistically
significantly above 50%, the methodology fails on this instrument.
"""

import pandas as pd
import numpy as np
from enum import Enum, auto
from dataclasses import dataclass, field
from scipy import stats


# ─── Daily Bar Construction ────────────────────────────────────────────────

def build_daily_bars(path: str) -> pd.DataFrame:
    """Resample 1-minute ET data to daily OHLCV bars (midnight-to-midnight ET)."""
    df = pd.read_parquet(path)
    df = df.set_index("DateTime_ET").sort_index()

    daily = df.resample("1D").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }).dropna()

    daily.index.name = "date"
    return daily


# ─── State Machine ─────────────────────────────────────────────────────────

class State(Enum):
    NEUTRAL = auto()
    BULLISH_WATCH = auto()
    BULLISH_ACTIVE = auto()
    BEARISH_WATCH = auto()
    BEARISH_ACTIVE = auto()


class Bias(Enum):
    NONE = 0
    LONG = 1
    SHORT = -1


@dataclass
class SwingPoint:
    index: int      # bar index in the daily array
    price: float


@dataclass
class BiasEngine:
    """
    Processes daily bars one at a time, maintaining the ICT daily bias
    state machine. Call `update(bar_index, row)` for each new daily bar.
    """
    state: State = State.NEUTRAL
    bias: Bias = Bias.NONE

    last_SH: SwingPoint | None = None
    last_SL: SwingPoint | None = None

    # The swing being validated during WATCH states
    pending_c3_high: float | None = None   # bullish watch: need price > this
    pending_c3_low: float | None = None    # bearish watch: need price < this

    # Rolling window of last 3 bars for swing detection
    _window: list = field(default_factory=list)
    _bar_idx: int = -1

    # Log of state transitions for analysis
    log: list = field(default_factory=list)

    def _detect_swings(self):
        """Check if the middle bar of the 3-bar window is a swing point."""
        if len(self._window) < 3:
            return None, None

        left, mid, right = self._window[-3], self._window[-2], self._window[-1]
        new_sh = None
        new_sl = None

        if mid["High"] > left["High"] and mid["High"] > right["High"]:
            new_sh = SwingPoint(index=self._bar_idx - 1, price=mid["High"])

        if mid["Low"] < left["Low"] and mid["Low"] < right["Low"]:
            new_sl = SwingPoint(index=self._bar_idx - 1, price=mid["Low"])

        return new_sh, new_sl

    def update(self, bar_idx: int, bar: dict) -> tuple[State, Bias]:
        """
        Process one completed daily bar.
        bar: dict with keys Open, High, Low, Close
        Returns (state, bias) after processing.
        """
        self._bar_idx = bar_idx
        self._window.append(bar)
        if len(self._window) > 3:
            self._window.pop(0)

        new_sh, new_sl = self._detect_swings()
        prev_state = self.state

        if self.state == State.NEUTRAL:
            self._handle_neutral(bar, new_sh, new_sl)

        elif self.state == State.BULLISH_WATCH:
            self._handle_bullish_watch(bar, new_sh, new_sl)

        elif self.state == State.BULLISH_ACTIVE:
            self._handle_bullish_active(bar, new_sh, new_sl)

        elif self.state == State.BEARISH_WATCH:
            self._handle_bearish_watch(bar, new_sh, new_sl)

        elif self.state == State.BEARISH_ACTIVE:
            self._handle_bearish_active(bar, new_sh, new_sl)

        # Update reference swing points
        if new_sh:
            self.last_SH = new_sh
        if new_sl:
            self.last_SL = new_sl

        if self.state != prev_state:
            self.log.append({
                "bar_idx": bar_idx,
                "from": prev_state.name,
                "to": self.state.name,
            })

        return self.state, self.bias

    def _handle_neutral(self, bar, new_sh, new_sl):
        if self.last_SH and bar["High"] > self.last_SH.price:
            self.state = State.BULLISH_WATCH
            self.bias = Bias.NONE
            self.pending_c3_high = None
        elif self.last_SL and bar["Low"] < self.last_SL.price:
            self.state = State.BEARISH_WATCH
            self.bias = Bias.NONE
            self.pending_c3_low = None

    def _handle_bullish_watch(self, bar, new_sh, new_sl):
        # Check if a new swing low just confirmed
        if new_sl:
            if self.last_SL is None or new_sl.price >= self.last_SL.price:
                # Valid higher swing low — C3 is the rightmost bar (current bar)
                self.pending_c3_high = self._window[-1]["High"]
            else:
                # Swing low broke previous swing low — invalidated
                self.state = State.NEUTRAL
                self.bias = Bias.NONE
                self.pending_c3_high = None
                return

        # Check if C3 high has been broken by current bar
        if self.pending_c3_high is not None and bar["High"] > self.pending_c3_high:
            self.state = State.BULLISH_ACTIVE
            self.bias = Bias.LONG
            self.pending_c3_high = None

    def _handle_bullish_active(self, bar, new_sh, new_sl):
        self.bias = Bias.LONG
        # Invalidation: price breaks a confirmed swing low
        if self.last_SL and bar["Low"] < self.last_SL.price:
            # Check if this also qualifies as a bearish break
            self.state = State.BEARISH_WATCH
            self.bias = Bias.NONE
            self.pending_c3_low = None

    def _handle_bearish_watch(self, bar, new_sh, new_sl):
        if new_sh:
            if self.last_SH is None or new_sh.price <= self.last_SH.price:
                # Valid lower swing high — C3 is the rightmost bar
                self.pending_c3_low = self._window[-1]["Low"]
            else:
                # Swing high broke previous swing high — invalidated
                self.state = State.NEUTRAL
                self.bias = Bias.NONE
                self.pending_c3_low = None
                return

        if self.pending_c3_low is not None and bar["Low"] < self.pending_c3_low:
            self.state = State.BEARISH_ACTIVE
            self.bias = Bias.SHORT
            self.pending_c3_low = None

    def _handle_bearish_active(self, bar, new_sh, new_sl):
        self.bias = Bias.SHORT
        if self.last_SH and bar["High"] > self.last_SH.price:
            self.state = State.BULLISH_WATCH
            self.bias = Bias.NONE
            self.pending_c3_high = None


# ─── Run the Bias Engine ───────────────────────────────────────────────────

def compute_bias_series(daily: pd.DataFrame) -> pd.DataFrame:
    """Run the state machine over all daily bars, return annotated DataFrame."""
    engine = BiasEngine()
    states = []
    biases = []

    for i in range(len(daily)):
        row = daily.iloc[i]
        bar = {
            "Open": row["Open"],
            "High": row["High"],
            "Low": row["Low"],
            "Close": row["Close"],
        }
        state, bias = engine.update(i, bar)
        states.append(state.name)
        biases.append(bias.value)

    result = daily.copy()
    result["state"] = states
    result["bias"] = biases
    result["prev_day_high"] = result["High"].shift(1)
    result["prev_day_low"] = result["Low"].shift(1)
    result["high_2d_ago"] = result["High"].shift(2)
    result["low_2d_ago"] = result["Low"].shift(2)

    # Outcome columns: did today's price run the target liquidity?
    result["ran_prev_high"] = result["High"] > result["prev_day_high"]
    result["ran_prev_low"] = result["Low"] < result["prev_day_low"]

    return result


# ─── Falsifiable Test ──────────────────────────────────────────────────────

def run_test(daily_bias: pd.DataFrame, alpha: float = 0.05):
    """
    Test whether the bias signal predicts previous-day liquidity runs
    at a rate significantly above 50%.

    Uses a one-sided binomial test.
    H0: hit_rate <= 0.50
    H1: hit_rate >  0.50
    """
    results = {}

    # ── BULLISH: when bias == LONG, does today run previous day's high?
    bullish = daily_bias[daily_bias["bias"] == Bias.LONG.value].copy()
    bullish = bullish.dropna(subset=["prev_day_high"])
    n_bull = len(bullish)
    hits_bull = bullish["ran_prev_high"].sum()
    rate_bull = hits_bull / n_bull if n_bull > 0 else 0
    pval_bull = stats.binomtest(hits_bull, n_bull, 0.5, alternative="greater").pvalue if n_bull > 0 else 1.0

    results["bullish"] = {
        "n": n_bull,
        "hits": int(hits_bull),
        "rate": rate_bull,
        "p_value": pval_bull,
        "significant": pval_bull < alpha,
    }

    # ── BEARISH: when bias == SHORT, does today run previous day's low?
    bearish = daily_bias[daily_bias["bias"] == Bias.SHORT.value].copy()
    bearish = bearish.dropna(subset=["prev_day_low"])
    n_bear = len(bearish)
    hits_bear = bearish["ran_prev_low"].sum()
    rate_bear = hits_bear / n_bear if n_bear > 0 else 0
    pval_bear = stats.binomtest(hits_bear, n_bear, 0.5, alternative="greater").pvalue if n_bear > 0 else 1.0

    results["bearish"] = {
        "n": n_bear,
        "hits": int(hits_bear),
        "rate": rate_bear,
        "p_value": pval_bear,
        "significant": pval_bear < alpha,
    }

    # ── BASELINE: unconditional rate (no bias filter)
    base = daily_bias.dropna(subset=["prev_day_high", "prev_day_low"])
    base_high_rate = base["ran_prev_high"].mean()
    base_low_rate = base["ran_prev_low"].mean()

    results["baseline"] = {
        "n": len(base),
        "unconditional_high_run_rate": base_high_rate,
        "unconditional_low_run_rate": base_low_rate,
    }

    # ── COMBINED: does the signal beat the unconditional base rate?
    # Chi-squared test: is the bullish hit rate different from the base rate?
    if n_bull > 0:
        pval_bull_vs_base = stats.binomtest(
            hits_bull, n_bull, base_high_rate, alternative="greater"
        ).pvalue if base_high_rate < 1.0 else 1.0
        results["bullish"]["vs_baseline_p"] = pval_bull_vs_base
        results["bullish"]["vs_baseline_sig"] = pval_bull_vs_base < alpha

    if n_bear > 0:
        pval_bear_vs_base = stats.binomtest(
            hits_bear, n_bear, base_low_rate, alternative="greater"
        ).pvalue if base_low_rate < 1.0 else 1.0
        results["bearish"]["vs_baseline_p"] = pval_bear_vs_base
        results["bearish"]["vs_baseline_sig"] = pval_bear_vs_base < alpha

    return results


# ── State Distribution ─────────────────────────────────────────────────────

def state_distribution(daily_bias: pd.DataFrame) -> dict:
    """How much time does the engine spend in each state?"""
    counts = daily_bias["state"].value_counts()
    total = len(daily_bias)
    return {state: {"count": int(c), "pct": round(c / total * 100, 1)}
            for state, c in counts.items()}


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("ICT DAILY BIAS — FALSIFIABLE TEST")
    print("Instrument: NQ (Nasdaq 100 Futures), 1m → Daily")
    print("=" * 70)

    # Build daily bars
    daily = build_daily_bars("data/nq_1m.parquet")
    print(f"\nDaily bars: {len(daily)}")
    print(f"Date range: {daily.index[0].date()} to {daily.index[-1].date()}")

    # Run bias engine
    daily_bias = compute_bias_series(daily)

    # State distribution
    print("\n── State Distribution ────────────────────────────────────")
    dist = state_distribution(daily_bias)
    for state, info in sorted(dist.items()):
        print(f"  {state:<20s}  {info['count']:>5d} days  ({info['pct']}%)")

    # Run the test
    print("\n── Hypothesis Test (alpha = 0.05) ────────────────────────")
    results = run_test(daily_bias)

    print("\nBASELINE (unconditional, all days):")
    b = results["baseline"]
    print(f"  N = {b['n']}")
    print(f"  P(today's high > yesterday's high) = {b['unconditional_high_run_rate']:.4f}")
    print(f"  P(today's low  < yesterday's low)  = {b['unconditional_low_run_rate']:.4f}")

    print("\nBULLISH ACTIVE days:")
    bu = results["bullish"]
    print(f"  N = {bu['n']}, Hits = {bu['hits']}, Rate = {bu['rate']:.4f}")
    print(f"  H0: rate <= 0.50  →  p = {bu['p_value']:.6f}  {'REJECT' if bu['significant'] else 'FAIL TO REJECT'}")
    if "vs_baseline_p" in bu:
        print(f"  H0: rate <= baseline ({b['unconditional_high_run_rate']:.4f})"
              f"  →  p = {bu['vs_baseline_p']:.6f}  {'REJECT' if bu['vs_baseline_sig'] else 'FAIL TO REJECT'}")

    print("\nBEARISH ACTIVE days:")
    be = results["bearish"]
    print(f"  N = {be['n']}, Hits = {be['hits']}, Rate = {be['rate']:.4f}")
    print(f"  H0: rate <= 0.50  →  p = {be['p_value']:.6f}  {'REJECT' if be['significant'] else 'FAIL TO REJECT'}")
    if "vs_baseline_p" in be:
        print(f"  H0: rate <= baseline ({b['unconditional_low_run_rate']:.4f})"
              f"  →  p = {be['vs_baseline_p']:.6f}  {'REJECT' if be['vs_baseline_sig'] else 'FAIL TO REJECT'}")

    # Verdict
    print("\n── Verdict ──────────────────────────────────────────────")
    bull_pass = bu.get("vs_baseline_sig", False)
    bear_pass = be.get("vs_baseline_sig", False)

    if bull_pass and bear_pass:
        print("  BOTH signals beat baseline — methodology NOT falsified")
    elif bull_pass or bear_pass:
        side = "Bullish" if bull_pass else "Bearish"
        fail = "Bearish" if bull_pass else "Bullish"
        print(f"  PARTIAL: {side} signal beats baseline, {fail} does not")
    else:
        print("  NEITHER signal beats baseline — methodology FALSIFIED on this data")

    print()
    return daily_bias, results


if __name__ == "__main__":
    daily_bias, results = main()
