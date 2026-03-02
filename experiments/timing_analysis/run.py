"""
Timing Analysis: When do aligned PCX+ICT targets get hit?

For each day where both models agree on direction (with PCX wick filter),
walk the 1-minute bars to find the exact time the target was breached.
"""

import sys
import io
import contextlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from ict_daily_bias import build_daily_bars, BiasEngine, Bias
from combined_test import compute_pcx, compute_ict_bias

DATA_PATH = PROJECT_ROOT / "data" / "nq_1m.parquet"
LOG_DIR = Path(__file__).resolve().parent / "logs"


# ── Step 1: Identify aligned days ─────────────────────────────────────────

def identify_aligned_days() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run both models, return (daily_with_signals, aligned_subset).
    aligned_subset has columns: date, direction, prev_high, prev_low, target_level, etc.
    """
    daily = build_daily_bars(str(DATA_PATH))
    daily = compute_ict_bias(daily)
    daily = compute_pcx(daily)

    daily["prev_day_high"] = daily["High"].shift(1)
    daily["prev_day_low"] = daily["Low"].shift(1)

    df = daily.iloc[3:].copy()

    long_mask = (
        (df["ict_bias"] == 1)
        & (df["pcx_prediction_for_today"] == 1)
        & (~df["pcx_prev_abnormal"].fillna(True))
    )
    short_mask = (
        (df["ict_bias"] == -1)
        & (df["pcx_prediction_for_today"] == -1)
        & (~df["pcx_prev_abnormal"].fillna(True))
    )

    rows = []
    for idx in df[long_mask].index:
        rows.append({
            "date": idx,
            "direction": "LONG",
            "prev_high": df.loc[idx, "prev_day_high"],
            "prev_low": df.loc[idx, "prev_day_low"],
            "target_level": df.loc[idx, "prev_day_high"],
            "day_high": df.loc[idx, "High"],
            "day_low": df.loc[idx, "Low"],
            "day_close": df.loc[idx, "Close"],
        })
    for idx in df[short_mask].index:
        rows.append({
            "date": idx,
            "direction": "SHORT",
            "prev_high": df.loc[idx, "prev_day_high"],
            "prev_low": df.loc[idx, "prev_day_low"],
            "target_level": df.loc[idx, "prev_day_low"],
            "day_high": df.loc[idx, "High"],
            "day_low": df.loc[idx, "Low"],
            "day_close": df.loc[idx, "Close"],
        })

    aligned = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return daily, aligned


# ── Step 2: Load and walk 1-minute bars ───────────────────────────────────

def load_minute_data() -> pd.DataFrame:
    """Load 1-min data with DateTime_ET as index."""
    df = pd.read_parquet(str(DATA_PATH))
    df["DateTime_ET"] = pd.to_datetime(df["DateTime_ET"])
    df = df.set_index("DateTime_ET").sort_index()
    return df


def compute_atr_series(daily: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute ATR as percentage of close on the daily DataFrame."""
    tr = pd.concat([
        daily["High"] - daily["Low"],
        (daily["High"] - daily["Close"].shift(1)).abs(),
        (daily["Low"] - daily["Close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr / daily["Close"] * 100


def walk_day(minute_df: pd.DataFrame, date: pd.Timestamp,
             direction: str, target_level: float,
             prev_high: float, prev_low: float,
             day_high: float, day_low: float, day_close: float,
             atr_pct: float) -> dict:
    """
    Walk 1-min bars for a single aligned day.
    Session starts at 6pm ET the prior calendar day.
    """
    cal_date = date.date() if hasattr(date, "date") else date
    prev_cal = cal_date - pd.Timedelta(days=1)

    session_start = pd.Timestamp(f"{prev_cal} 18:00:00")
    session_end = pd.Timestamp(f"{cal_date} 23:59:59")

    bars = minute_df.loc[session_start:session_end]

    prev_range = prev_high - prev_low
    day_range_pct = (day_high - day_low) / day_close * 100 if day_close != 0 else 0

    result = {
        "date": cal_date,
        "direction": direction,
        "prev_high": prev_high,
        "prev_low": prev_low,
        "hit": False,
        "hit_time_et": None,
        "hit_session": None,
        "minutes_from_open": None,
        "overshoot_points": None,
        "overshoot_pct": None,
        "opposite_hit": False,
        "day_range_pct": round(day_range_pct, 4),
        "atr_14_pct": round(atr_pct, 4) if not np.isnan(atr_pct) else None,
        "range_vs_atr": round(day_range_pct / atr_pct, 4) if (atr_pct and not np.isnan(atr_pct) and atr_pct != 0) else None,
        "inside_bar": (day_high <= prev_high) and (day_low >= prev_low),
    }

    if len(bars) == 0:
        return result

    for ts, bar in bars.iterrows():
        if direction == "LONG" and bar["High"] > target_level:
            result["hit"] = True
            result["hit_time_et"] = str(ts)
            result["hit_session"] = bar["session"]
            result["minutes_from_open"] = int((ts - session_start).total_seconds() / 60)
            result["overshoot_points"] = round(day_high - target_level, 2)
            result["overshoot_pct"] = round((day_high - target_level) / prev_range * 100, 2) if prev_range > 0 else 0
            break
        elif direction == "SHORT" and bar["Low"] < target_level:
            result["hit"] = True
            result["hit_time_et"] = str(ts)
            result["hit_session"] = bar["session"]
            result["minutes_from_open"] = int((ts - session_start).total_seconds() / 60)
            result["overshoot_points"] = round(target_level - day_low, 2)
            result["overshoot_pct"] = round((target_level - day_low) / prev_range * 100, 2) if prev_range > 0 else 0
            break

    if not result["hit"]:
        if direction == "LONG":
            result["opposite_hit"] = day_low < prev_low
        else:
            result["opposite_hit"] = day_high > prev_high

    return result


# ── Step 3: Summary statistics ────────────────────────────────────────────

def minutes_to_clock(minutes_from_6pm: float) -> str:
    """Convert minutes-from-6pm-ET to a clock time string."""
    base = pd.Timestamp("2000-01-01 18:00:00")
    t = base + pd.Timedelta(minutes=minutes_from_6pm)
    return t.strftime("%H:%M")


def print_summary(results_df: pd.DataFrame):
    """Print aggregate statistics."""
    hits = results_df[results_df["hit"]]
    misses = results_df[~results_df["hit"]]

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for direction in ["LONG", "SHORT"]:
        sub = results_df[results_df["direction"] == direction]
        sub_hits = sub[sub["hit"]]
        rate = sub_hits.shape[0] / sub.shape[0] * 100 if len(sub) > 0 else 0
        print(f"\n  {direction}: {sub_hits.shape[0]}/{sub.shape[0]} hit ({rate:.1f}%)")

    if len(hits) > 0:
        mins = hits["minutes_from_open"].dropna()
        print(f"\n── Time to Target (all hits, n={len(mins)}) ──")
        for label, val in [
            ("Mean", mins.mean()),
            ("Median", mins.median()),
            ("P25", mins.quantile(0.25)),
            ("P75", mins.quantile(0.75)),
            ("P90", mins.quantile(0.90)),
        ]:
            print(f"  {label:8s}: {val:6.0f} min  ({minutes_to_clock(val)})")

        for direction in ["LONG", "SHORT"]:
            sub = hits[hits["direction"] == direction]["minutes_from_open"].dropna()
            if len(sub) > 0:
                print(f"\n  {direction} hits (n={len(sub)}):")
                print(f"    Mean:   {sub.mean():6.0f} min  ({minutes_to_clock(sub.mean())})")
                print(f"    Median: {sub.median():6.0f} min  ({minutes_to_clock(sub.median())})")

    if len(hits) > 0:
        print(f"\n── Session Breakdown (hits) ──")
        sess_counts = hits["hit_session"].value_counts()
        for sess, count in sess_counts.items():
            pct = count / len(hits) * 100
            print(f"  {sess:8s}: {count:4d}  ({pct:5.1f}%)")

    if len(hits) > 0:
        os_pts = hits["overshoot_points"].dropna()
        os_pct = hits["overshoot_pct"].dropna()
        print(f"\n── Overshoot (hits) ──")
        print(f"  Mean points:  {os_pts.mean():.2f}")
        print(f"  Mean % of prior range: {os_pct.mean():.1f}%")

    if len(misses) > 0:
        print(f"\n── Failures (n={len(misses)}) ──")
        inside = misses["inside_bar"].sum()
        opposite = misses["opposite_hit"].sum()
        other = len(misses) - inside - opposite
        print(f"  Inside bars (no breakout either way): {inside}")
        print(f"  Opposite target hit instead:          {opposite}")
        print(f"  Other (broke out but didn't reach):   {other}")

    print()


# ── Main ──────────────────────────────────────────────────────────────────

def run_timing_analysis():
    """Main entry point."""
    print("=" * 70)
    print("TIMING ANALYSIS: ALIGNED PCX + ICT CASES")
    print("=" * 70)

    daily, aligned = identify_aligned_days()
    n_long = (aligned["direction"] == "LONG").sum()
    n_short = (aligned["direction"] == "SHORT").sum()
    print(f"\nAligned days: {len(aligned)} (LONG={n_long}, SHORT={n_short})")

    print("Loading 1-minute data...")
    minute_df = load_minute_data()
    print(f"  {len(minute_df)} bars loaded")

    atr_series = compute_atr_series(daily)

    print("Walking 1-minute bars for each aligned day...")
    results = []
    for _, row in aligned.iterrows():
        date = row["date"]
        atr_val = atr_series.loc[date] if date in atr_series.index else np.nan
        result = walk_day(
            minute_df, date, row["direction"], row["target_level"],
            row["prev_high"], row["prev_low"],
            row["day_high"], row["day_low"], row["day_close"],
            atr_val,
        )
        results.append(result)

    results_df = pd.DataFrame(results)

    long_df = results_df[results_df["direction"] == "LONG"]
    short_df = results_df[results_df["direction"] == "SHORT"]

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(LOG_DIR / "aligned_long_timing.csv", index=False)
    short_df.to_csv(LOG_DIR / "aligned_short_timing.csv", index=False)
    print(f"\nCSVs saved to {LOG_DIR}/")

    # Print summary and tee to file
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_summary(results_df)
    summary_text = buf.getvalue()

    print(summary_text, end="")

    with open(LOG_DIR / "summary.txt", "w") as f:
        f.write(summary_text)
    print(f"Summary saved to {LOG_DIR / 'summary.txt'}")

    return results_df


if __name__ == "__main__":
    results_df = run_timing_analysis()
