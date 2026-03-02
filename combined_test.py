"""
Combined Test: ICT Daily Bias + PCX Expansion Model
====================================================
Enhanced with four additional filter/analysis tracks:

  Track 1 — PCX Proximity Filter
      Relaxes the binary state transition threshold to allow near-miss closes.
      Sweeps PROXIMITY_THRESHOLD across [0.01, 0.02, 0.05, 0.10].

  Track 2 — Adverse Candle Filter
      Suppresses signals when the signal bar moved strongly against the
      predicted direction (potential conviction reversal).

  Track 3 — Inside Day Prediction (Logistic Regression)
      Uses InsideDayPredictor to pre-screen qualifying days.
      Days predicted to be inside bars are filtered out.

  Track 4 — CISD Integration (Triple-Stack)
      Adds CISD daily signal as a third confirming filter.
      Reports sample sizes and hit rates for PCX+ICT+CISD.

Baseline (PCX + ICT agree + no abnormal wicks): ~79% LONG, ~78% SHORT.

Key alignment:
  - ICT bias on bar[i] predicts bar[i]'s outcome (will today run prev day's high/low?)
  - PCX target on bar[i] predicts bar[i+1]'s outcome (made at bar[i]'s close)
  - So for the SAME day's outcome (bar i): ICT uses bias[i], PCX uses target[i-1]
"""

import pandas as pd
import numpy as np
from scipy import stats
from ict_daily_bias import build_daily_bars, BiasEngine, Bias, State
from cisd_analysis import get_daily_cisd_direction
from inside_day_predictor import InsideDayPredictor


# ── Configuration ──────────────────────────────────────────────────────────────

# Track 1: PCX Proximity
PROXIMITY_THRESHOLDS = [0.01, 0.02, 0.05, 0.10]   # sweep values
PROXIMITY_DEFAULT    = 0.05                         # default for combined runs

# Track 2: Adverse Candle
ADVERSE_RANGE_RATIO  = 0.75   # today's range > 75% of prior → conviction move
ADVERSE_FILTER       = True   # toggle adverse candle filter in combined runs

# Track 3: Inside Day
INSIDE_DAY_FILTER    = True   # toggle inside day predictor in combined runs

# Track 4: CISD
CISD_FILTER          = True   # toggle CISD triple-stack in combined runs


# ── Model Computation ──────────────────────────────────────────────────────────

def compute_pcx(daily: pd.DataFrame, proximity_threshold: float = 0.0) -> pd.DataFrame:
    """
    Run the PCX model with optional proximity near-miss state transitions.

    proximity_threshold=0.0 → standard binary (original behaviour)
    proximity_threshold>0.0 → near-miss close within threshold% of prior range
                               counts as a valid state transition.

    Returns the dataframe with columns:
      pcx_state, pcx_target, pcx_prediction_for_today
      pcx_abnormal, pcx_prev_abnormal
      adverse_long, adverse_short, adverse_for_today_long, adverse_for_today_short
    """
    df = daily.copy()
    df["prev_high"] = df["High"].shift(1)
    df["prev_low"]  = df["Low"].shift(1)

    states  = np.zeros(len(df))
    targets = np.zeros(len(df))
    current_state = 0

    for i in range(1, len(df)):
        close   = df["Close"].iloc[i]
        p_high  = df["prev_high"].iloc[i]
        p_low   = df["prev_low"].iloc[i]
        p_range = p_high - p_low

        states[i] = current_state

        # ── Helpers ────────────────────────────────────────────────────────
        def bullish_break():
            if close >= p_high:
                return True
            if proximity_threshold > 0 and p_range > 0:
                return (p_high - close) / p_range <= proximity_threshold
            return False

        def bearish_break():
            if close <= p_low:
                return True
            if proximity_threshold > 0 and p_range > 0:
                return (close - p_low) / p_range <= proximity_threshold
            return False

        # ── Target based on current (entering) state ───────────────────────
        if current_state == 1:
            targets[i] = 1 if bullish_break() else -1
        elif current_state == -1:
            targets[i] = -1 if bearish_break() else 1
        else:
            targets[i] = 0

        # ── Update state for next bar (priority: full break > near-miss) ──
        if close >= p_high:
            current_state = 1
        elif close <= p_low:
            current_state = -1
        elif proximity_threshold > 0 and p_range > 0:
            if (p_high - close) / p_range <= proximity_threshold:
                current_state = 1
            elif (close - p_low) / p_range <= proximity_threshold:
                current_state = -1
        # else: state unchanged (inside bar)

    df["pcx_state"]  = states
    df["pcx_target"] = targets

    # Align: target[i] predicts bar i+1 → prediction_for_today[i] = target[i-1]
    df["pcx_prediction_for_today"] = df["pcx_target"].shift(1)

    # ── Abnormal wick filter ────────────────────────────────────────────────
    df["range"]      = df["High"] - df["Low"]
    df["upper_wick"] = df["High"] - df[["Open", "Close"]].max(axis=1)
    df["lower_wick"] = df[["Open", "Close"]].min(axis=1) - df["Low"]
    df["pcx_abnormal"] = (
        (df["upper_wick"] > 0.4 * df["range"]) |
        (df["lower_wick"] > 0.4 * df["range"])
    )
    df["pcx_prev_abnormal"] = df["pcx_abnormal"].shift(1)

    # ── Track 2: Adverse candle flag ───────────────────────────────────────
    # Adverse for BEARISH signal (expecting tomorrow to sweep low):
    #   today is bullish AND close > prior midpoint AND conviction move
    # Adverse for BULLISH signal (expecting tomorrow to sweep high):
    #   today is bearish AND close < prior midpoint AND conviction move

    prior_mid = (df["prev_high"] + df["prev_low"]) / 2
    conv_move = df["range"] > (ADVERSE_RANGE_RATIO * df["range"].shift(1))

    df["adverse_short"] = (
        (df["Close"] > df["Open"]) &          # bullish bar (goes against short signal)
        (df["Close"] > prior_mid) &
        conv_move
    )
    df["adverse_long"] = (
        (df["Close"] < df["Open"]) &          # bearish bar (goes against long signal)
        (df["Close"] < prior_mid) &
        conv_move
    )

    # Shift forward: flag applies to NEXT bar's prediction
    df["adverse_for_today_short"] = df["adverse_short"].shift(1)
    df["adverse_for_today_long"]  = df["adverse_long"].shift(1)

    return df


def compute_ict_bias(daily: pd.DataFrame) -> pd.DataFrame:
    """Run the ICT bias engine, return bias per bar."""
    engine = BiasEngine()
    biases = []

    for i in range(len(daily)):
        row = daily.iloc[i]
        bar = {
            "Open":  row["Open"],
            "High":  row["High"],
            "Low":   row["Low"],
            "Close": row["Close"],
        }
        _, bias = engine.update(i, bar)
        biases.append(bias.value)

    result = daily.copy()
    result["ict_bias"] = biases
    return result


# ── Hit-Rate Helper ─────────────────────────────────────────────────────────────

def hit_rate_block(df: pd.DataFrame, label: str,
                   pcx_col: str = "pcx_prediction_for_today") -> dict:
    """
    Compute combined hit rate for a filtered subset.
    Requires df already filtered for valid rows (both active, wick OK, etc.).
    pcx_col: column to use for PCX direction (default = standard PCX).
    Returns dict with long/short/combined counts and rates.
    """
    long_mask  = (df["ict_bias"] == 1)  & (df[pcx_col] == 1)
    short_mask = (df["ict_bias"] == -1) & (df[pcx_col] == -1)

    long_sub  = df[long_mask]
    short_sub = df[short_mask]

    n_long  = len(long_sub)
    n_short = len(short_sub)

    long_rate  = long_sub["ran_high"].mean()  if n_long  > 0 else np.nan
    short_rate = short_sub["ran_low"].mean()  if n_short > 0 else np.nan

    # Combined (pool both directions)
    n_total = n_long + n_short
    if n_total > 0:
        hits = (long_sub["ran_high"].sum() if n_long > 0 else 0) + \
               (short_sub["ran_low"].sum() if n_short > 0 else 0)
        combined_rate = hits / n_total
    else:
        combined_rate = np.nan

    return {
        "label":          label,
        "n_long":         n_long,
        "long_rate":      long_rate,
        "n_short":        n_short,
        "short_rate":     short_rate,
        "n_combined":     n_total,
        "combined_rate":  combined_rate,
    }


def print_hit_block(r: dict) -> None:
    lr  = f"{r['long_rate']:.1%}"  if not np.isnan(r['long_rate'])      else "  N/A "
    sr  = f"{r['short_rate']:.1%}" if not np.isnan(r['short_rate'])     else "  N/A "
    cr  = f"{r['combined_rate']:.1%}" if not np.isnan(r['combined_rate']) else "  N/A "
    print(f"  {r['label']:<45s}  LONG {lr} (n={r['n_long']:>3d})  "
          f"SHORT {sr} (n={r['n_short']:>3d})  COMBINED {cr} (n={r['n_combined']:>3d})")


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("COMBINED TEST: ICT DAILY BIAS + PCX EXPANSION  (Enhanced — 4 Tracks)")
    print("=" * 80)

    # ── Load & compute all signals ────────────────────────────────────────────
    daily = build_daily_bars("data/nq_1m.parquet")
    daily = compute_ict_bias(daily)
    daily = compute_pcx(daily, proximity_threshold=0.0)  # standard PCX

    # Outcome columns (did today's bar run the target?)
    daily["ran_high"] = daily["High"] > daily["High"].shift(1)
    daily["ran_low"]  = daily["Low"]  < daily["Low"].shift(1)

    # CISD direction (Track 4)
    # cisd_direction[i] = bullish/bearish CISD event ON bar[i], using bar[i-1] as prior.
    # For filtering bar[i]'s PCX+ICT prediction (made at bar[i-1]), we want to know
    # whether bar[i-1] itself showed a CISD — i.e. the signal bar's momentum confirmation.
    # shift(1) aligns: cisd_on_signal_bar[i] = cisd_direction[i-1].
    daily["cisd_direction_raw"]    = get_daily_cisd_direction(daily)
    daily["cisd_direction"]        = daily["cisd_direction_raw"].shift(1)  # signal-bar CISD

    # Drop first few rows (model warm-up)
    df = daily.iloc[3:].copy()

    # ── 1. SIGNAL CORRELATION ─────────────────────────────────────────────────
    print("\n── 1. Signal Correlation ─────────────────────────────────")
    both_active = df[(df["ict_bias"] != 0) & (df["pcx_prediction_for_today"] != 0)].copy()
    print(f"Days where both models signal: {len(both_active)} / {len(df)}")

    agreement_rate = np.nan
    if len(both_active) > 0:
        agree = (both_active["ict_bias"] == both_active["pcx_prediction_for_today"]).sum()
        disagree = len(both_active) - agree
        agreement_rate = agree / len(both_active)
        print(f"Agreement:    {agree} ({agreement_rate:.1%})")
        print(f"Disagreement: {disagree} ({1 - agreement_rate:.1%})")

        corr, corr_p = stats.pearsonr(
            both_active["ict_bias"], both_active["pcx_prediction_for_today"]
        )
        print(f"Pearson r = {corr:.4f}, p = {corr_p:.6f}")

    # ── 2. JOINT ACCURACY TABLE ──────────────────────────────────────────────
    print("\n── 2. Accuracy by Signal Combination ────────────────────")
    combos = []
    for ict_val, ict_label in [(1, "LONG"), (-1, "SHORT"), (0, "NONE")]:
        for pcx_val, pcx_label in [(1, "HIGH"), (-1, "LOW"), (0, "NONE")]:
            mask = (df["ict_bias"] == ict_val) & (df["pcx_prediction_for_today"] == pcx_val)
            sub  = df[mask]
            n    = len(sub)
            combos.append({
                "ICT": ict_label, "PCX": pcx_label, "n": n,
                "ran_high_rate": sub["ran_high"].mean() if n > 0 else np.nan,
                "ran_low_rate":  sub["ran_low"].mean()  if n > 0 else np.nan,
            })
    combo_df = pd.DataFrame(combos)
    print(combo_df.to_string(index=False, float_format="%.3f"))

    # ── 3. AGREEMENT vs DISAGREEMENT ─────────────────────────────────────────
    print("\n── 3. Agreement vs Disagreement Accuracy ────────────────")
    both_long  = df[(df["ict_bias"] == 1)  & (df["pcx_prediction_for_today"] == 1)]
    both_short = df[(df["ict_bias"] == -1) & (df["pcx_prediction_for_today"] == -1)]
    print(f"Both LONG  → ran_high rate: {both_long['ran_high'].mean():.4f}  (n={len(both_long)})")
    print(f"Both SHORT → ran_low  rate: {both_short['ran_low'].mean():.4f}  (n={len(both_short)})")

    for ict_v, ict_l, pcx_v, pcx_l in [
        (1, "ICT LONG",  -1, "PCX LOW"),
        (-1, "ICT SHORT", 1, "PCX HIGH"),
    ]:
        sub = df[(df["ict_bias"] == ict_v) & (df["pcx_prediction_for_today"] == pcx_v)]
        if len(sub) > 0:
            print(f"{ict_l} + {pcx_l} → ran_high: {sub['ran_high'].mean():.4f}, "
                  f"ran_low: {sub['ran_low'].mean():.4f}  (n={len(sub)})")

    # ── 4. STANDALONE vs COMBINED ─────────────────────────────────────────────
    print("\n── 4. Standalone vs Combined Hit Rates ──────────────────")

    ict_long_all  = df[df["ict_bias"] == 1]
    ict_short_all = df[df["ict_bias"] == -1]
    ict_long_rate  = ict_long_all["ran_high"].mean()
    ict_short_rate = ict_short_all["ran_low"].mean()
    print(f"ICT LONG alone:     {ict_long_rate:.4f}  (n={len(ict_long_all)})")
    print(f"ICT SHORT alone:    {ict_short_rate:.4f}  (n={len(ict_short_all)})")

    pcx_high_all = df[
        (df["pcx_prediction_for_today"] == 1) &
        (~df["pcx_prev_abnormal"].fillna(True))
    ]
    pcx_low_all  = df[
        (df["pcx_prediction_for_today"] == -1) &
        (~df["pcx_prev_abnormal"].fillna(True))
    ]
    pcx_high_rate = pcx_high_all["ran_high"].mean() if len(pcx_high_all) > 0 else np.nan
    pcx_low_rate  = pcx_low_all["ran_low"].mean()   if len(pcx_low_all)  > 0 else np.nan
    print(f"PCX HIGH alone (wick-filtered): {pcx_high_rate:.4f}  (n={len(pcx_high_all)})")
    print(f"PCX LOW  alone (wick-filtered): {pcx_low_rate:.4f}  (n={len(pcx_low_all)})")

    # BASELINE: combined agree + wick filter
    both_long_clean = df[
        (df["ict_bias"] == 1) &
        (df["pcx_prediction_for_today"] == 1) &
        (~df["pcx_prev_abnormal"].fillna(True))
    ]
    both_short_clean = df[
        (df["ict_bias"] == -1) &
        (df["pcx_prediction_for_today"] == -1) &
        (~df["pcx_prev_abnormal"].fillna(True))
    ]
    combined_long_rate  = both_long_clean["ran_high"].mean()  if len(both_long_clean)  > 0 else np.nan
    combined_short_rate = both_short_clean["ran_low"].mean()  if len(both_short_clean) > 0 else np.nan
    print(f"BASELINE COMBINED LONG  (agree + wick): {combined_long_rate:.4f}  (n={len(both_long_clean)})")
    print(f"BASELINE COMBINED SHORT (agree + wick): {combined_short_rate:.4f}  (n={len(both_short_clean)})")

    # ── 5. STATISTICAL TEST ───────────────────────────────────────────────────
    print("\n── 5. Does Combining Improve Over Best Standalone? ──────")
    for hits, n, rate_ict, rate_pcx, label_ict, label_pcx, side in [
        (
            int(both_long_clean["ran_high"].sum()), len(both_long_clean),
            ict_long_rate, pcx_high_rate,
            f"ICT alone ({ict_long_rate:.4f})", f"PCX alone ({pcx_high_rate:.4f})",
            "LONG",
        ),
        (
            int(both_short_clean["ran_low"].sum()), len(both_short_clean),
            ict_short_rate, pcx_low_rate,
            f"ICT alone ({ict_short_rate:.4f})", f"PCX alone ({pcx_low_rate:.4f})",
            "SHORT",
        ),
    ]:
        if n > 10:
            p1 = stats.binomtest(hits, n, rate_ict, alternative="greater").pvalue
            p2 = stats.binomtest(hits, n, rate_pcx, alternative="greater").pvalue if not np.isnan(rate_pcx) else 1.0
            print(f"Combined {side} vs {label_ict}: p = {p1:.6f}")
            print(f"Combined {side} vs {label_pcx}: p = {p2:.6f}")

    # ── 6. SIGNAL INDEPENDENCE (CHI-SQUARED) ─────────────────────────────────
    print("\n── 6. Signal Independence (Chi-Squared) ─────────────────")
    active = df[(df["ict_bias"] != 0) & (df["pcx_prediction_for_today"] != 0)].copy()
    if len(active) > 0:
        ct = pd.crosstab(
            active["ict_bias"].map({1: "ICT_LONG", -1: "ICT_SHORT"}),
            active["pcx_prediction_for_today"].map({1: "PCX_HIGH", -1: "PCX_LOW"}),
        )
        print(ct)
        chi2, chi_p, dof, _ = stats.chi2_contingency(ct)
        cramers_v = np.sqrt(chi2 / (len(active) * (min(ct.shape) - 1)))
        print(f"\nChi-squared = {chi2:.2f}, p = {chi_p:.6f}, Cramer's V = {cramers_v:.4f}")
        if chi_p < 0.05:
            print("Signals are NOT independent (significant association)")
        else:
            print("Signals ARE independent (no significant association)")

    # ══════════════════════════════════════════════════════════════════════════
    # TRACK 1 — PCX PROXIMITY FILTER
    # ══════════════════════════════════════════════════════════════════════════
    print("\n")
    print("=" * 80)
    print("TRACK 1 — PCX PROXIMITY FILTER (near-miss state transitions)")
    print("=" * 80)
    print("Baseline (exact-break, wick-filtered, agree with ICT):")
    print(f"  LONG  {combined_long_rate:.1%} (n={len(both_long_clean)})   "
          f"SHORT {combined_short_rate:.1%} (n={len(both_short_clean)})")
    print()
    print(f"{'Threshold':>12}  {'LONG rate':>10}  {'LONG n':>6}  {'SHORT rate':>10}  {'SHORT n':>7}  {'COMB rate':>10}  {'COMB n':>7}")
    print("-" * 80)

    # Standard (threshold=0) as row 0
    r0 = hit_rate_block(
        df[~df["pcx_prev_abnormal"].fillna(True)],
        "exact (0.00)",
    )
    print(f"  {'0.00 (exact)':>10}  {r0['long_rate']:>10.1%}  {r0['n_long']:>6}  "
          f"{r0['short_rate']:>10.1%}  {r0['n_short']:>7}  "
          f"{r0['combined_rate']:>10.1%}  {r0['n_combined']:>7}")

    prox_results = {}
    for thresh in PROXIMITY_THRESHOLDS:
        # Recompute PCX with this proximity threshold
        daily_p = build_daily_bars("data/nq_1m.parquet")
        daily_p = compute_ict_bias(daily_p)
        daily_p = compute_pcx(daily_p, proximity_threshold=thresh)
        daily_p["ran_high"] = daily_p["High"] > daily_p["High"].shift(1)
        daily_p["ran_low"]  = daily_p["Low"]  < daily_p["Low"].shift(1)
        df_p    = daily_p.iloc[3:].copy()

        r = hit_rate_block(
            df_p[~df_p["pcx_prev_abnormal"].fillna(True)],
            f"prox_{thresh:.2f}",
        )
        prox_results[thresh] = r
        print(f"  {thresh:>10.2f}  {r['long_rate']:>10.1%}  {r['n_long']:>6}  "
              f"{r['short_rate']:>10.1%}  {r['n_short']:>7}  "
              f"{r['combined_rate']:>10.1%}  {r['n_combined']:>7}")

    # Pick best threshold by combined hit rate (for use in combined run below)
    best_thresh = max(prox_results, key=lambda t: (
        prox_results[t]["combined_rate"] if not np.isnan(prox_results[t]["combined_rate"]) else -1
    ))
    print(f"\n  Best proximity threshold by combined hit rate: {best_thresh:.2f}")

    # ══════════════════════════════════════════════════════════════════════════
    # TRACK 2 — ADVERSE CANDLE FILTER
    # ══════════════════════════════════════════════════════════════════════════
    print("\n")
    print("=" * 80)
    print(f"TRACK 2 — ADVERSE CANDLE FILTER (ADVERSE_RANGE_RATIO = {ADVERSE_RANGE_RATIO})")
    print("=" * 80)

    # Base mask: agree + wick filter
    base_mask = (
        (~df["pcx_prev_abnormal"].fillna(True))
    )
    # Long base
    long_base  = df[base_mask & (df["ict_bias"] == 1)  & (df["pcx_prediction_for_today"] == 1)]
    short_base = df[base_mask & (df["ict_bias"] == -1) & (df["pcx_prediction_for_today"] == -1)]

    # Adverse filter removes rows where signal bar was adversely directional
    long_adv_removed  = df[base_mask & (df["ict_bias"] == 1)  & (df["pcx_prediction_for_today"] == 1)
                           & (df["adverse_for_today_long"].fillna(False))]
    short_adv_removed = df[base_mask & (df["ict_bias"] == -1) & (df["pcx_prediction_for_today"] == -1)
                           & (df["adverse_for_today_short"].fillna(False))]

    long_clean  = df[base_mask & (df["ict_bias"] == 1)  & (df["pcx_prediction_for_today"] == 1)
                     & (~df["adverse_for_today_long"].fillna(False))]
    short_clean = df[base_mask & (df["ict_bias"] == -1) & (df["pcx_prediction_for_today"] == -1)
                     & (~df["adverse_for_today_short"].fillna(False))]

    print(f"\n  LONG  — before adverse filter: {len(long_base):>3d} days, "
          f"rate = {long_base['ran_high'].mean():.1%}")
    print(f"  LONG  — adverse candles removed: {len(long_adv_removed):>3d} "
          f"(hit rate within: {long_adv_removed['ran_high'].mean():.1%})")
    print(f"  LONG  — after adverse filter:  {len(long_clean):>3d} days, "
          f"rate = {long_clean['ran_high'].mean():.1%}")

    print(f"\n  SHORT — before adverse filter: {len(short_base):>3d} days, "
          f"rate = {short_base['ran_low'].mean():.1%}")
    print(f"  SHORT — adverse candles removed: {len(short_adv_removed):>3d} "
          f"(hit rate within: {short_adv_removed['ran_low'].mean():.1%})")
    print(f"  SHORT — after adverse filter:  {len(short_clean):>3d} days, "
          f"rate = {short_clean['ran_low'].mean():.1%}")

    # Chi-squared: does removing adverse candles significantly improve hit rate?
    for side_label, pre_df, post_df, outcome_col in [
        ("LONG",  long_base,  long_clean,  "ran_high"),
        ("SHORT", short_base, short_clean, "ran_low"),
    ]:
        n_pre, hits_pre = len(pre_df),  int(pre_df[outcome_col].sum())
        n_post, hits_post = len(post_df), int(post_df[outcome_col].sum())
        if n_post > 0 and n_pre > 0:
            rate_pre = hits_pre / n_pre
            p = stats.binomtest(hits_post, n_post, rate_pre, alternative="greater").pvalue
            print(f"\n  {side_label} adverse filter: p (post > pre rate {rate_pre:.4f}) = {p:.6f} "
                  f"{'*' if p < 0.05 else ''}")

    # ══════════════════════════════════════════════════════════════════════════
    # TRACK 3 — INSIDE DAY PREDICTION
    # ══════════════════════════════════════════════════════════════════════════
    print("\n")
    print("=" * 80)
    print("TRACK 3 — INSIDE DAY PREDICTION (Logistic Regression Pre-Filter)")
    print("=" * 80)

    predictor = InsideDayPredictor()
    try:
        metrics = predictor.fit(daily)
        predictor.print_report(metrics)

        if metrics["val_auc"] < 0.60:
            print("\n  Warning: Val AUC < 0.60. Filter may not be meaningful.")

        # Apply to qualifying days
        inside_proba = predictor.predict_proba(daily)
        inside_flag  = predictor.predict(daily)     # 1 = likely inside, suppress signal

        daily["inside_day_prob"]  = inside_proba
        daily["inside_day_flag"]  = inside_flag

        # Re-slice df after adding inside columns
        df = daily.iloc[3:].copy()

        # Filter: only keep days NOT predicted as inside
        base_long_mask  = (
            (~df["pcx_prev_abnormal"].fillna(True)) &
            (df["ict_bias"] == 1) & (df["pcx_prediction_for_today"] == 1)
        )
        base_short_mask = (
            (~df["pcx_prev_abnormal"].fillna(True)) &
            (df["ict_bias"] == -1) & (df["pcx_prediction_for_today"] == -1)
        )

        inside_filtered_long  = df[base_long_mask  & (df["inside_day_flag"] == 0)]
        inside_filtered_short = df[base_short_mask & (df["inside_day_flag"] == 0)]
        inside_removed_long   = df[base_long_mask  & (df["inside_day_flag"] == 1)]
        inside_removed_short  = df[base_short_mask & (df["inside_day_flag"] == 1)]

        print(f"\n  LONG  — before inside filter: {base_long_mask.sum():>3d} days, "
              f"rate = {df[base_long_mask]['ran_high'].mean():.1%}")
        print(f"  LONG  — inside-predicted removed: {len(inside_removed_long):>3d} "
              f"(hit rate within removed: {inside_removed_long['ran_high'].mean():.1%})")
        print(f"  LONG  — after inside filter:  {len(inside_filtered_long):>3d} days, "
              f"rate = {inside_filtered_long['ran_high'].mean():.1%}")

        print(f"\n  SHORT — before inside filter: {base_short_mask.sum():>3d} days, "
              f"rate = {df[base_short_mask]['ran_low'].mean():.1%}")
        print(f"  SHORT — inside-predicted removed: {len(inside_removed_short):>3d} "
              f"(hit rate within removed: {inside_removed_short['ran_low'].mean():.1%})")
        print(f"  SHORT — after inside filter:  {len(inside_filtered_short):>3d} days, "
              f"rate = {inside_filtered_short['ran_low'].mean():.1%}")

    except Exception as e:
        print(f"\n  InsideDayPredictor error: {e}")
        print("  Skipping Track 3.")
        inside_flag = np.zeros(len(daily))
        daily["inside_day_flag"] = inside_flag
        df = daily.iloc[3:].copy()

    # ══════════════════════════════════════════════════════════════════════════
    # TRACK 4 — CISD INTEGRATION (TRIPLE-STACK)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n")
    print("=" * 80)
    print("TRACK 4 — CISD INTEGRATION (PCX + ICT + CISD Triple-Stack)")
    print("=" * 80)

    # CISD direction must match the PCX+ICT signal direction to be confirming
    # For a LONG signal: cisd_direction == 1 (bullish CISD)
    # For a SHORT signal: cisd_direction == -1 (bearish CISD)

    base_long_clean  = df[
        (~df["pcx_prev_abnormal"].fillna(True)) &
        (df["ict_bias"] == 1) & (df["pcx_prediction_for_today"] == 1)
    ]
    base_short_clean = df[
        (~df["pcx_prev_abnormal"].fillna(True)) &
        (df["ict_bias"] == -1) & (df["pcx_prediction_for_today"] == -1)
    ]

    cisd_long  = base_long_clean[base_long_clean["cisd_direction"] == 1]
    cisd_short = base_short_clean[base_short_clean["cisd_direction"] == -1]

    n_cisd_long  = len(cisd_long)
    n_cisd_short = len(cisd_short)

    cisd_long_rate  = cisd_long["ran_high"].mean()  if n_cisd_long  > 0 else np.nan
    cisd_short_rate = cisd_short["ran_low"].mean()  if n_cisd_short > 0 else np.nan
    n_cisd_total    = n_cisd_long + n_cisd_short

    if n_cisd_total > 0:
        cisd_hits = (
            (int(cisd_long["ran_high"].sum())  if n_cisd_long  > 0 else 0) +
            (int(cisd_short["ran_low"].sum())  if n_cisd_short > 0 else 0)
        )
        cisd_combined_rate = cisd_hits / n_cisd_total
    else:
        cisd_combined_rate = np.nan

    print(f"\n  PCX + ICT baseline (agree + wick-filtered):")
    print(f"    LONG  {combined_long_rate:.1%} (n={len(both_long_clean)})   "
          f"SHORT {combined_short_rate:.1%} (n={len(both_short_clean)})")

    print(f"\n  PCX + ICT + CISD (triple-stack, CISD must confirm direction):")
    print(f"    LONG  {'N/A' if np.isnan(cisd_long_rate)  else f'{cisd_long_rate:.1%}':>6}  (n={n_cisd_long})")
    print(f"    SHORT {'N/A' if np.isnan(cisd_short_rate) else f'{cisd_short_rate:.1%}':>6}  (n={n_cisd_short})")
    print(f"    COMBINED: {'N/A' if np.isnan(cisd_combined_rate) else f'{cisd_combined_rate:.1%}'} (n={n_cisd_total})")

    # CISD coverage: what % of double-stack days have a confirming CISD?
    if len(base_long_clean) > 0:
        cisd_coverage_long = n_cisd_long / len(base_long_clean)
        print(f"\n  CISD coverage — LONG  side: {cisd_coverage_long:.1%} "
              f"({n_cisd_long}/{len(base_long_clean)} double-stack days have confirming CISD)")
    if len(base_short_clean) > 0:
        cisd_coverage_short = n_cisd_short / len(base_short_clean)
        print(f"  CISD coverage — SHORT side: {cisd_coverage_short:.1%} "
              f"({n_cisd_short}/{len(base_short_clean)} double-stack days have confirming CISD)")

    # Statistical test: does CISD triple-stack beat double-stack?
    if n_cisd_total >= 50:
        print(f"\n  Statistical test (triple-stack vs double-stack):")
        for hits_ts, n_ts, rate_ds, side in [
            (int(cisd_long["ran_high"].sum()),  n_cisd_long,  combined_long_rate,  "LONG"),
            (int(cisd_short["ran_low"].sum()),  n_cisd_short, combined_short_rate, "SHORT"),
        ]:
            if n_ts > 10 and not np.isnan(rate_ds):
                p = stats.binomtest(hits_ts, n_ts, rate_ds, alternative="greater").pvalue
                print(f"    {side}: triple vs double ({rate_ds:.4f}): p = {p:.6f} "
                      f"{'*' if p < 0.05 else '(not significant)'}")
    else:
        print(f"\n  N={n_cisd_total} — below minimum of 50 for meaningful statistics.")

    # ══════════════════════════════════════════════════════════════════════════
    # COMBINED: ALL FILTERS ACTIVE
    # ══════════════════════════════════════════════════════════════════════════
    print("\n")
    print("=" * 80)
    print("ALL FILTERS COMBINED (Proximity + Adverse Candle + Inside Day + CISD)")
    print("=" * 80)

    summary_rows = []

    def add_summary(label, df_long, df_short, outcome_long="ran_high", outcome_short="ran_low"):
        n_l  = len(df_long)
        n_s  = len(df_short)
        r_l  = df_long[outcome_long].mean()   if n_l > 0 else np.nan
        r_s  = df_short[outcome_short].mean() if n_s > 0 else np.nan
        n_c  = n_l + n_s
        if n_c > 0:
            hits = ((int(df_long[outcome_long].sum()) if n_l > 0 else 0) +
                    (int(df_short[outcome_short].sum()) if n_s > 0 else 0))
            r_c = hits / n_c
        else:
            r_c = np.nan
        summary_rows.append({
            "Configuration": label,
            "LONG_rate": r_l, "LONG_n": n_l,
            "SHORT_rate": r_s, "SHORT_n": n_s,
            "COMB_rate": r_c, "COMB_n": n_c,
        })

    # Baseline
    add_summary(
        "Baseline (agree + wick)",
        df[(~df["pcx_prev_abnormal"].fillna(True)) & (df["ict_bias"] == 1) & (df["pcx_prediction_for_today"] == 1)],
        df[(~df["pcx_prev_abnormal"].fillna(True)) & (df["ict_bias"] == -1) & (df["pcx_prediction_for_today"] == -1)],
    )

    # + Adverse candle only
    add_summary(
        "+ Adverse candle filter",
        df[(~df["pcx_prev_abnormal"].fillna(True)) & (df["ict_bias"] == 1)  & (df["pcx_prediction_for_today"] == 1)
           & (~df["adverse_for_today_long"].fillna(False))],
        df[(~df["pcx_prev_abnormal"].fillna(True)) & (df["ict_bias"] == -1) & (df["pcx_prediction_for_today"] == -1)
           & (~df["adverse_for_today_short"].fillna(False))],
    )

    # + Inside day only
    add_summary(
        "+ Inside day filter",
        df[(~df["pcx_prev_abnormal"].fillna(True)) & (df["ict_bias"] == 1)  & (df["pcx_prediction_for_today"] == 1)
           & (df["inside_day_flag"] == 0)],
        df[(~df["pcx_prev_abnormal"].fillna(True)) & (df["ict_bias"] == -1) & (df["pcx_prediction_for_today"] == -1)
           & (df["inside_day_flag"] == 0)],
    )

    # + CISD only
    add_summary(
        "+ CISD triple-stack",
        cisd_long,
        cisd_short,
    )

    # Adverse + Inside
    adv_inside_long  = df[(~df["pcx_prev_abnormal"].fillna(True)) & (df["ict_bias"] == 1)  & (df["pcx_prediction_for_today"] == 1)
                          & (~df["adverse_for_today_long"].fillna(False)) & (df["inside_day_flag"] == 0)]
    adv_inside_short = df[(~df["pcx_prev_abnormal"].fillna(True)) & (df["ict_bias"] == -1) & (df["pcx_prediction_for_today"] == -1)
                          & (~df["adverse_for_today_short"].fillna(False)) & (df["inside_day_flag"] == 0)]
    add_summary("+ Adverse + Inside", adv_inside_long, adv_inside_short)

    # Adverse + CISD
    adv_cisd_long  = cisd_long[~cisd_long["adverse_for_today_long"].fillna(False)]
    adv_cisd_short = cisd_short[~cisd_short["adverse_for_today_short"].fillna(False)]
    add_summary("+ Adverse + CISD", adv_cisd_long, adv_cisd_short)

    # Inside + CISD
    ins_cisd_long  = cisd_long[cisd_long["inside_day_flag"] == 0]
    ins_cisd_short = cisd_short[cisd_short["inside_day_flag"] == 0]
    add_summary("+ Inside + CISD", ins_cisd_long, ins_cisd_short)

    # All three additional filters
    all_long  = cisd_long[~cisd_long["adverse_for_today_long"].fillna(False) & (cisd_long["inside_day_flag"] == 0)]
    all_short = cisd_short[~cisd_short["adverse_for_today_short"].fillna(False) & (cisd_short["inside_day_flag"] == 0)]
    add_summary("+ Adverse + Inside + CISD (ALL)", all_long, all_short)

    # Print summary table
    print()
    hdr = f"  {'Configuration':<45}  {'LONG':>7}  {'n_L':>4}  {'SHORT':>7}  {'n_S':>4}  {'COMB':>7}  {'n_C':>4}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for row in summary_rows:
        lr = f"{row['LONG_rate']:.1%}"  if not np.isnan(row['LONG_rate'])  else "  N/A "
        sr = f"{row['SHORT_rate']:.1%}" if not np.isnan(row['SHORT_rate']) else "  N/A "
        cr = f"{row['COMB_rate']:.1%}"  if not np.isnan(row['COMB_rate'])  else "  N/A "
        print(f"  {row['Configuration']:<45}  {lr:>7}  {row['LONG_n']:>4}  "
              f"{sr:>7}  {row['SHORT_n']:>4}  {cr:>7}  {row['COMB_n']:>4}")

    # ── Acceptance criteria ────────────────────────────────────────────────
    print("\n── Acceptance Criteria Check ─────────────────────────────")
    baseline_combined = (
        (combined_long_rate * len(both_long_clean) + combined_short_rate * len(both_short_clean))
        / (len(both_long_clean) + len(both_short_clean))
        if (len(both_long_clean) + len(both_short_clean)) > 0 else np.nan
    )
    print(f"  Baseline combined hit rate:     {baseline_combined:.1%}  (target: ≥79%)")

    all_comb_n    = summary_rows[-1]["COMB_n"]
    all_comb_rate = summary_rows[-1]["COMB_rate"]
    if not np.isnan(all_comb_rate):
        beats = all_comb_rate >= baseline_combined if not np.isnan(baseline_combined) else False
        print(f"  All-filters combined hit rate:  {all_comb_rate:.1%}  n={all_comb_n}  "
              f"{'✓ beats baseline' if beats else '✗ below baseline'}")
    else:
        print("  All-filters combined hit rate:  N/A")

    # Inside day AUC check
    if "val_auc" in locals().get("metrics", {}):
        auc_ok = metrics.get("val_auc", 0) > 0.60
        print(f"  Inside day predictor val AUC:   {metrics.get('val_auc', 'N/A')}  "
              f"{'✓ >0.60' if auc_ok else '✗ ≤0.60'}")

    # CISD sample size check
    print(f"  CISD triple-stack sample size:  {n_cisd_total}  "
          f"{'✓ ≥50' if n_cisd_total >= 50 else '✗ <50'}")

    # ── Final verdict ──────────────────────────────────────────────────────
    print("\n── Final Verdict ────────────────────────────────────────")
    print(f"  ICT alone LONG:       {ict_long_rate:.1%} over {len(ict_long_all)} days")
    print(f"  PCX alone HIGH (flt): {pcx_high_rate:.1%} over {len(pcx_high_all)} days")
    print(f"  Baseline LONG (flt):  {combined_long_rate:.1%} over {len(both_long_clean)} days")
    print()
    print(f"  ICT alone SHORT:      {ict_short_rate:.1%} over {len(ict_short_all)} days")
    print(f"  PCX alone LOW  (flt): {pcx_low_rate:.1%} over {len(pcx_low_all)} days")
    print(f"  Baseline SHORT (flt): {combined_short_rate:.1%} over {len(both_short_clean)} days")
    if len(both_active) > 0:
        overlap = len(both_active) / len(df) * 100
        print(f"\n  Signal overlap: {overlap:.1f}% of days have both signals active")
        print(f"  Agreement rate: {agreement_rate:.1%} when both active")


if __name__ == "__main__":
    main()
