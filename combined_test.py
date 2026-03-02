"""
Combined Test: ICT Daily Bias + PCX Expansion Model

Question: Do these models measure the same thing, or can they be stacked?

Key alignment issue:
  - ICT bias on bar[i] predicts bar[i]'s outcome (will today run prev day's high/low?)
  - PCX target on bar[i] predicts bar[i+1]'s outcome (made at bar[i]'s close)
  - So for the SAME day's outcome (bar i): ICT uses bias[i], PCX uses target[i-1]

We test:
  1. Signal correlation (are they saying the same thing?)
  2. Agreement accuracy (both point same direction → better?)
  3. Disagreement accuracy (which one dominates?)
  4. Independence test (chi-squared on the joint confusion matrix)
"""

import pandas as pd
import numpy as np
from scipy import stats
from ict_daily_bias import build_daily_bars, BiasEngine, Bias, State


def compute_pcx(daily: pd.DataFrame) -> pd.DataFrame:
    """Run the PCX model, return predictions aligned to the bar they predict FOR."""
    df = daily.copy()
    df["prev_high"] = df["High"].shift(1)
    df["prev_low"] = df["Low"].shift(1)

    states = np.zeros(len(df))
    targets = np.zeros(len(df))  # prediction for NEXT bar
    current_state = 0

    for i in range(1, len(df)):
        close = df["Close"].iloc[i]
        p_high = df["prev_high"].iloc[i]
        p_low = df["prev_low"].iloc[i]

        states[i] = current_state

        if current_state == 1:
            if close >= p_high:
                targets[i] = 1
            else:
                targets[i] = -1
        elif current_state == -1:
            if close <= p_low:
                targets[i] = -1
            else:
                targets[i] = 1
        else:
            targets[i] = 0

        if close >= p_high:
            current_state = 1
        elif close <= p_low:
            current_state = -1

    df["pcx_state"] = states
    df["pcx_target"] = targets

    # Shift target forward: target[i] predicts bar i+1, so pcx_for_tomorrow[i] = target for bar i+1
    # To align with bar i's outcome, we need target from bar i-1
    df["pcx_prediction_for_today"] = df["pcx_target"].shift(1)

    # Abnormal wick filter
    df["range"] = df["High"] - df["Low"]
    df["upper_wick"] = df["High"] - df[["Open", "Close"]].max(axis=1)
    df["lower_wick"] = df[["Open", "Close"]].min(axis=1) - df["Low"]
    df["pcx_abnormal"] = (df["upper_wick"] > 0.4 * df["range"]) | (
        df["lower_wick"] > 0.4 * df["range"]
    )
    # The wick filter applies to the bar that MADE the prediction (bar i-1)
    df["pcx_prev_abnormal"] = df["pcx_abnormal"].shift(1)

    return df


def compute_ict_bias(daily: pd.DataFrame) -> pd.DataFrame:
    """Run the ICT bias engine, return bias per bar."""
    engine = BiasEngine()
    biases = []

    for i in range(len(daily)):
        row = daily.iloc[i]
        bar = {
            "Open": row["Open"],
            "High": row["High"],
            "Low": row["Low"],
            "Close": row["Close"],
        }
        _, bias = engine.update(i, bar)
        biases.append(bias.value)

    daily = daily.copy()
    daily["ict_bias"] = biases
    return daily


def main():
    print("=" * 70)
    print("COMBINED TEST: ICT DAILY BIAS + PCX EXPANSION")
    print("=" * 70)

    daily = build_daily_bars("data/nq_1m.parquet")
    daily = compute_ict_bias(daily)
    daily = compute_pcx(daily)

    # Outcome columns
    daily["prev_day_high"] = daily["High"].shift(1)
    daily["prev_day_low"] = daily["Low"].shift(1)
    daily["ran_high"] = daily["High"] > daily["prev_day_high"]
    daily["ran_low"] = daily["Low"] < daily["prev_day_low"]

    # ── Normalize both signals to same encoding ──
    # ICT: 1 = LONG (expect ran_high), -1 = SHORT (expect ran_low), 0 = no signal
    # PCX: 1 = target high (expect ran_high), -1 = target low (expect ran_low), 0 = no signal
    # Both are already aligned.

    # Drop rows without valid predictions from both
    df = daily.iloc[3:].copy()  # need a few bars for both models to initialize

    # ── 1. SIGNAL CORRELATION ──────────────────────────────────────────
    print("\n── 1. Signal Correlation ─────────────────────────────────")

    # Where both have a signal
    both_active = df[(df["ict_bias"] != 0) & (df["pcx_prediction_for_today"] != 0)].copy()
    print(f"Days where both models signal: {len(both_active)} / {len(df)}")

    if len(both_active) > 0:
        agree = (both_active["ict_bias"] == both_active["pcx_prediction_for_today"]).sum()
        disagree = len(both_active) - agree
        agreement_rate = agree / len(both_active)
        print(f"Agreement:    {agree} ({agreement_rate:.1%})")
        print(f"Disagreement: {disagree} ({1 - agreement_rate:.1%})")

        # Pearson correlation between the two signal vectors
        corr, corr_p = stats.pearsonr(
            both_active["ict_bias"], both_active["pcx_prediction_for_today"]
        )
        print(f"Pearson r = {corr:.4f}, p = {corr_p:.6f}")

    # ── 2. JOINT ACCURACY TABLE ────────────────────────────────────────
    print("\n── 2. Accuracy by Signal Combination ────────────────────")

    def accuracy_for_subset(subset, label):
        if len(subset) == 0:
            print(f"  {label}: n=0")
            return None
        # For bullish predictions, success = ran_high. For bearish, success = ran_low.
        # Use the "consensus" direction. If signals disagree, use each separately.
        # Actually, we need a unified target. Let's define success based on what the
        # combined signal says.
        # When both agree on LONG: success = ran_high
        # When both agree on SHORT: success = ran_low
        # When they disagree: not applicable for agreement analysis
        pass

    # Build a clean joint table
    combos = []
    for ict_val, ict_label in [(1, "LONG"), (-1, "SHORT"), (0, "NONE")]:
        for pcx_val, pcx_label in [(1, "HIGH"), (-1, "LOW"), (0, "NONE")]:
            mask = (df["ict_bias"] == ict_val) & (df["pcx_prediction_for_today"] == pcx_val)
            sub = df[mask]
            n = len(sub)
            if n == 0:
                combos.append({
                    "ICT": ict_label, "PCX": pcx_label, "n": 0,
                    "ran_high_rate": np.nan, "ran_low_rate": np.nan,
                })
                continue
            combos.append({
                "ICT": ict_label,
                "PCX": pcx_label,
                "n": n,
                "ran_high_rate": sub["ran_high"].mean(),
                "ran_low_rate": sub["ran_low"].mean(),
            })

    combo_df = pd.DataFrame(combos)
    print(combo_df.to_string(index=False, float_format="%.3f"))

    # ── 3. AGREEMENT vs DISAGREEMENT ───────────────────────────────────
    print("\n── 3. Agreement vs Disagreement Accuracy ────────────────")

    # Both say LONG
    both_long = df[(df["ict_bias"] == 1) & (df["pcx_prediction_for_today"] == 1)]
    both_long_acc = both_long["ran_high"].mean() if len(both_long) > 0 else np.nan
    print(f"Both LONG  → ran_high rate: {both_long_acc:.4f}  (n={len(both_long)})")

    # Both say SHORT
    both_short = df[(df["ict_bias"] == -1) & (df["pcx_prediction_for_today"] == -1)]
    both_short_acc = both_short["ran_low"].mean() if len(both_short) > 0 else np.nan
    print(f"Both SHORT → ran_low  rate: {both_short_acc:.4f}  (n={len(both_short)})")

    # ICT LONG, PCX LOW (disagree)
    ict_long_pcx_low = df[(df["ict_bias"] == 1) & (df["pcx_prediction_for_today"] == -1)]
    if len(ict_long_pcx_low) > 0:
        print(f"ICT LONG + PCX LOW  → ran_high: {ict_long_pcx_low['ran_high'].mean():.4f}, "
              f"ran_low: {ict_long_pcx_low['ran_low'].mean():.4f}  (n={len(ict_long_pcx_low)})")

    # ICT SHORT, PCX HIGH (disagree)
    ict_short_pcx_high = df[(df["ict_bias"] == -1) & (df["pcx_prediction_for_today"] == 1)]
    if len(ict_short_pcx_high) > 0:
        print(f"ICT SHORT + PCX HIGH → ran_high: {ict_short_pcx_high['ran_high'].mean():.4f}, "
              f"ran_low: {ict_short_pcx_high['ran_low'].mean():.4f}  (n={len(ict_short_pcx_high)})")

    # ── 4. STANDALONE vs COMBINED COMPARISON ───────────────────────────
    print("\n── 4. Standalone vs Combined Hit Rates ──────────────────")

    # ICT alone (when active)
    ict_long_all = df[df["ict_bias"] == 1]
    ict_short_all = df[df["ict_bias"] == -1]
    ict_long_rate = ict_long_all["ran_high"].mean()
    ict_short_rate = ict_short_all["ran_low"].mean()
    print(f"ICT LONG alone:     {ict_long_rate:.4f}  (n={len(ict_long_all)})")
    print(f"ICT SHORT alone:    {ict_short_rate:.4f}  (n={len(ict_short_all)})")

    # PCX alone (when active, excluding abnormal wicks)
    pcx_high_all = df[(df["pcx_prediction_for_today"] == 1) & (~df["pcx_prev_abnormal"].fillna(True))]
    pcx_low_all = df[(df["pcx_prediction_for_today"] == -1) & (~df["pcx_prev_abnormal"].fillna(True))]
    pcx_high_rate = pcx_high_all["ran_high"].mean() if len(pcx_high_all) > 0 else np.nan
    pcx_low_rate = pcx_low_all["ran_low"].mean() if len(pcx_low_all) > 0 else np.nan
    print(f"PCX HIGH alone (filtered): {pcx_high_rate:.4f}  (n={len(pcx_high_all)})")
    print(f"PCX LOW alone (filtered):  {pcx_low_rate:.4f}  (n={len(pcx_low_all)})")

    # Combined: both agree, PCX not abnormal
    both_long_clean = df[
        (df["ict_bias"] == 1)
        & (df["pcx_prediction_for_today"] == 1)
        & (~df["pcx_prev_abnormal"].fillna(True))
    ]
    both_short_clean = df[
        (df["ict_bias"] == -1)
        & (df["pcx_prediction_for_today"] == -1)
        & (~df["pcx_prev_abnormal"].fillna(True))
    ]
    combined_long_rate = both_long_clean["ran_high"].mean() if len(both_long_clean) > 0 else np.nan
    combined_short_rate = both_short_clean["ran_low"].mean() if len(both_short_clean) > 0 else np.nan
    print(f"COMBINED LONG (agree + filtered):  {combined_long_rate:.4f}  (n={len(both_long_clean)})")
    print(f"COMBINED SHORT (agree + filtered): {combined_short_rate:.4f}  (n={len(both_short_clean)})")

    # ── 5. STATISTICAL TEST: DOES COMBINING BEAT EITHER ALONE? ─────────
    print("\n── 5. Does Combining Improve Over Best Standalone? ──────")

    # Bullish: combined vs ICT alone
    if len(both_long_clean) > 10:
        hits = both_long_clean["ran_high"].sum()
        n = len(both_long_clean)
        # Test against ICT standalone rate
        p_vs_ict = stats.binomtest(int(hits), n, ict_long_rate, alternative="greater").pvalue
        # Test against PCX standalone rate
        p_vs_pcx = stats.binomtest(int(hits), n, pcx_high_rate, alternative="greater").pvalue if not np.isnan(pcx_high_rate) else 1.0
        print(f"Combined LONG vs ICT alone ({ict_long_rate:.4f}): p = {p_vs_ict:.6f}")
        print(f"Combined LONG vs PCX alone ({pcx_high_rate:.4f}): p = {p_vs_pcx:.6f}")

    if len(both_short_clean) > 10:
        hits = both_short_clean["ran_low"].sum()
        n = len(both_short_clean)
        p_vs_ict = stats.binomtest(int(hits), n, ict_short_rate, alternative="greater").pvalue
        p_vs_pcx = stats.binomtest(int(hits), n, pcx_low_rate, alternative="greater").pvalue if not np.isnan(pcx_low_rate) else 1.0
        print(f"Combined SHORT vs ICT alone ({ict_short_rate:.4f}): p = {p_vs_ict:.6f}")
        print(f"Combined SHORT vs PCX alone ({pcx_low_rate:.4f}): p = {p_vs_pcx:.6f}")

    # ── 6. INDEPENDENCE TEST ───────────────────────────────────────────
    print("\n── 6. Signal Independence (Chi-Squared) ─────────────────")

    # Contingency table: ICT signal vs PCX signal (excluding NONE from both)
    active = df[(df["ict_bias"] != 0) & (df["pcx_prediction_for_today"] != 0)].copy()
    if len(active) > 0:
        ct = pd.crosstab(
            active["ict_bias"].map({1: "ICT_LONG", -1: "ICT_SHORT"}),
            active["pcx_prediction_for_today"].map({1: "PCX_HIGH", -1: "PCX_LOW"}),
        )
        print(ct)
        chi2, chi_p, dof, expected = stats.chi2_contingency(ct)
        cramers_v = np.sqrt(chi2 / (len(active) * (min(ct.shape) - 1)))
        print(f"\nChi-squared = {chi2:.2f}, p = {chi_p:.6f}, Cramer's V = {cramers_v:.4f}")
        if chi_p < 0.05:
            print("Signals are NOT independent (significant association)")
        else:
            print("Signals ARE independent (no significant association)")

    # ── VERDICT ────────────────────────────────────────────────────────
    print("\n── Verdict ──────────────────────────────────────────────")
    print(f"  ICT alone LONG:       {ict_long_rate:.1%} over {len(ict_long_all)} days")
    print(f"  PCX alone HIGH (flt): {pcx_high_rate:.1%} over {len(pcx_high_all)} days")
    print(f"  Combined LONG (flt):  {combined_long_rate:.1%} over {len(both_long_clean)} days")
    print()
    print(f"  ICT alone SHORT:      {ict_short_rate:.1%} over {len(ict_short_all)} days")
    print(f"  PCX alone LOW (flt):  {pcx_low_rate:.1%} over {len(pcx_low_all)} days")
    print(f"  Combined SHORT (flt): {combined_short_rate:.1%} over {len(both_short_clean)} days")

    overlap = len(both_active) / len(df) * 100 if len(df) > 0 else 0
    print(f"\n  Signal overlap: {overlap:.1f}% of days have both signals active")
    print(f"  Agreement rate: {agreement_rate:.1%} when both active")


if __name__ == "__main__":
    main()
