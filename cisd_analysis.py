import pandas as pd
import numpy as np


def get_daily_cisd_direction(daily: pd.DataFrame) -> pd.Series:
    """
    Compute CISD (Change in State of Delivery) signal for daily bars.

    Rules (match the `prepare()` logic used in this module):
      Bullish CISD: previous daily candle was bearish,  current close > previous close
      Bearish CISD: previous daily candle was bullish,  current close < previous close

    Returns a pd.Series of int (1 = bullish, -1 = bearish, 0 = no signal)
    aligned to the same index as `daily`.

    Works with both capitalised (OHLCV) and lower-case (ohlcv) column names.
    """
    col = {c.lower(): c for c in daily.columns}
    C = col.get("close", "Close")
    O = col.get("open",  "Open")

    close = daily[C]
    open_ = daily[O]

    direction = np.where(close > open_, "bullish",
                np.where(close < open_, "bearish", "neutral"))
    prev_direction = pd.Series(direction, index=daily.index).shift(1)
    prev_close     = close.shift(1)

    cisd = np.select(
        [
            (prev_direction == "bearish") & (close > prev_close),
            (prev_direction == "bullish") & (close < prev_close),
        ],
        [1, -1],
        default=0,
    )
    return pd.Series(cisd.astype(int), index=daily.index, name="cisd_direction")


