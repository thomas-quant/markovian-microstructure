import pandas as pd
import numpy as np

def run_backtest(data_path):
    # Load data
    df = pd.read_parquet(data_path)
    df['DateTime_ET'] = pd.to_datetime(df['DateTime_ET'])
    df.set_index('DateTime_ET', inplace=True)
    
    # Resample to Daily
    daily = df.resample('D').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    }).dropna()
    
    # Previous day's levels
    daily['prev_high'] = daily['High'].shift(1)
    daily['prev_low'] = daily['Low'].shift(1)
    
    # Determine the "State" of the expansion
    # State is Bullish (1) if last breakout was above high, Bearish (-1) if below low
    states = np.zeros(len(daily))
    targets = np.zeros(len(daily)) # 1 for High, -1 for Low
    
    current_state = 0 # 0 = Unknown/Neutral until first breakout
    
    for i in range(1, len(daily)):
        close = daily['Close'].iloc[i]
        p_high = daily['prev_high'].iloc[i]
        p_low = daily['prev_low'].iloc[i]
        
        # 1. Store the state we ENTERED with (for this prediction)
        states[i] = current_state
        
        # 2. Determine Target based on PRIOR state
        if current_state == 1:
            # Bullish Expansion Rules - compare to prev HIGH
            if close >= p_high:
                targets[i] = 1  # Target High (continuation)
            else:
                targets[i] = -1 # Target Low (failure to close above)
        elif current_state == -1:
            # Bearish Expansion Rules - compare to prev LOW
            if close <= p_low:
                targets[i] = -1 # Target Low (continuation)
            else:
                targets[i] = 1  # Target High (failure to close below)
        else:
            # Neutral state (no prior breakout) - no prediction
            targets[i] = 0
        
        # 3. Update State for NEXT candle based on this candle's breakout
        if close >= p_high:
            current_state = 1
        elif close <= p_low:
            current_state = -1
        # else: state remains unchanged (inside bar)
                
    daily['state'] = states
    daily['target'] = targets
    
    # Success check (next day)
    daily['next_high'] = daily['High'].shift(-1)
    daily['next_low'] = daily['Low'].shift(-1)
    
    # Success is True if predicted target is hit
    daily['success'] = np.where(
        daily['target'] == 1,
        daily['next_high'] > daily['High'], # Predicted High target
        daily['next_low'] < daily['Low']    # Predicted Low target
    )
    
    # Abnormal Wicks Filter
    daily['range'] = daily['High'] - daily['Low']
    daily['upper_wick'] = daily['High'] - daily[['Open', 'Close']].max(axis=1)
    daily['lower_wick'] = daily[['Open', 'Close']].min(axis=1) - daily['Low']
    daily['is_abnormal'] = (daily['upper_wick'] > 0.4 * daily['range']) | (daily['lower_wick'] > 0.4 * daily['range'])
    
    # Calculate performance - exclude neutral state (no prediction)
    results = daily.dropna().copy()
    results = results[results['target'] != 0]  # Only include candles with predictions
    
    total_acc = results['success'].mean()
    
    filtered_results = results[~results['is_abnormal']]
    filtered_acc = filtered_results['success'].mean()
    
    print(f"Total Daily Candles with Predictions: {len(results)}")
    print(f"Overall Accuracy: {total_acc:.2%}")
    print(f"Filtered Accuracy (no abnormal wicks): {filtered_acc:.2%}")
    
    # Break down by target type
    high_targets = results[results['target'] == 1]
    low_targets = results[results['target'] == -1]
    
    print(f"\nHigh Target Accuracy: {high_targets['success'].mean():.2%} (n={len(high_targets)})")
    print(f"Low Target Accuracy: {low_targets['success'].mean():.2%} (n={len(low_targets)})")
    
    # Break down by state (state we ENTERED with when making prediction)
    bull_state = results[results['state'] == 1]
    bear_state = results[results['state'] == -1]
    
    print(f"\nBullish State Accuracy: {bull_state['success'].mean():.2%} (n={len(bull_state)})")
    print(f"Bearish State Accuracy: {bear_state['success'].mean():.2%} (n={len(bear_state)})")
    
    return results

if __name__ == "__main__":
    run_backtest('data/nq_1m.parquet')
