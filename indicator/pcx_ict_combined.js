//@version=1
// PCX + ICT Combined Daily Bias Indicator
// Implements both the PCX Expansion state machine and ICT Daily Bias swing-point engine.
// When both models agree direction (and PCX bar passes wick filter), a high-confidence
// combined signal fires (~79% accuracy on NQ daily backtest).

// ─── STATE VARIABLES ────────────────────────────────────────────────────────
// Module-level persistent state (survives across onTick calls)

let pcxState = 0;      // 0=neutral, 1=bullish, -1=bearish
let pcxTarget = 0;     // 1=target high, -1=target low

let ictState = 0;      // 0=NEUTRAL, 1=BULL_WATCH, 2=BULL_ACTIVE, 3=BEAR_WATCH, 4=BEAR_ACTIVE
let ictBias = 0;       // 0=NONE, 1=LONG, -1=SHORT

let lastSH_price = null;
let lastSH_time = null;
let lastSL_price = null;
let lastSL_time = null;

let pendingC3H = null;  // bullish watch: need price > this
let pendingC3L = null;  // bearish watch: need price < this

let prevPcxTarget = 0;
let prevIsAbnormal = false;

let prevIctState = 0;

let barCount = 0;

// Drawing IDs for target lines (so we can update/delete them)
let targetHighLineId = null;
let targetLowLineId = null;

// ─── INIT ───────────────────────────────────────────────────────────────────

init = () => {
    indicator({
        onMainPanel: true,
        format: 'inherit'
    });

    // Model toggles
    input.bool('Show PCX', true, 'showPCX');
    input.bool('Show ICT Bias', true, 'showICT');
    input.bool('Show Combined Signal', true, 'showCombined');

    // Visual toggles
    input.bool('Bar Coloring', true, 'showBarColor');
    input.bool('Target Lines', true, 'showTargetLines');
    input.bool('Swing Points', true, 'showSwingPoints');
    input.bool('State Switch Labels', true, 'showSwitchLabels');
    input.bool('Combined Signal Arrows', true, 'showSignalArrows');

    // Filter
    input.float('Wick Filter Threshold', 0.4, 'wickThreshold', 0.1, 0.9, 0.05);

    // Colors
    input.color('Combined Long Color', color.rgba(0, 200, 83, 1), 'colCombLong');
    input.color('Combined Short Color', color.rgba(255, 23, 68, 1), 'colCombShort');
    input.color('Swing High Color', color.rgba(255, 152, 0, 1), 'colSwingHigh');
    input.color('Swing Low Color', color.rgba(33, 150, 243, 1), 'colSwingLow');
};

// ─── ON TICK ────────────────────────────────────────────────────────────────

onTick = (length, _moment, _, ta, inputs) => {
    barCount++;

    // Need at least 3 bars for swing detection
    if (barCount < 3) return;

    const showPCX = inputs.showPCX;
    const showICT = inputs.showICT;
    const showCombined = inputs.showCombined;
    const showBarColor = inputs.showBarColor;
    const showTargetLines = inputs.showTargetLines;
    const showSwingPoints = inputs.showSwingPoints;
    const showSwitchLabels = inputs.showSwitchLabels;
    const showSignalArrows = inputs.showSignalArrows;
    const wickThreshold = inputs.wickThreshold;
    const colCombLong = inputs.colCombLong;
    const colCombShort = inputs.colCombShort;
    const colSwingHigh = inputs.colSwingHigh;
    const colSwingLow = inputs.colSwingLow;

    // ─── Current and previous bar data ──────────────────────────────
    const curHigh = high(0);
    const curLow = low(0);
    const curClose = closeC(0);
    const curOpen = openC(0);
    const curTime = time(0);

    const prevHigh = high(1);
    const prevLow = low(1);
    const prevClose = closeC(1);

    const prev2High = high(2);
    const prev2Low = low(2);

    // ─── PCX EXPANSION MODEL ────────────────────────────────────────
    // Save previous target for combined signal alignment
    prevPcxTarget = pcxTarget;

    // Abnormal wick filter on current bar
    const barRange = curHigh - curLow;
    const upperWick = curHigh - Math.max(curOpen, curClose);
    const lowerWick = Math.min(curOpen, curClose) - curLow;
    const curIsAbnormal = barRange > 0
        ? (upperWick > wickThreshold * barRange) || (lowerWick > wickThreshold * barRange)
        : false;

    // Calculate target based on state we ENTERED with
    if (pcxState === 1) {
        pcxTarget = curClose >= prevHigh ? 1 : -1;
    } else if (pcxState === -1) {
        pcxTarget = curClose <= prevLow ? -1 : 1;
    } else {
        pcxTarget = 0;
    }

    // Update state for NEXT bar
    if (curClose >= prevHigh) {
        pcxState = 1;
    } else if (curClose <= prevLow) {
        pcxState = -1;
    }
    // else state unchanged

    // PCX signal (filtered)
    const pcxSignal = curIsAbnormal ? 0 : pcxTarget;

    // Store for next bar's combined signal
    const savedPrevAbnormal = prevIsAbnormal;
    prevIsAbnormal = curIsAbnormal;

    // ─── ICT DAILY BIAS ENGINE ──────────────────────────────────────
    // Swing detection: middle bar of last 3 is a pivot
    const newSwingHigh = prevHigh > prev2High && prevHigh > curHigh;
    const newSwingLow = prevLow < prev2Low && prevLow < curLow;

    const newSH_price = newSwingHigh ? prevHigh : null;
    const newSL_price = newSwingLow ? prevLow : null;
    const newSH_time = newSwingHigh ? time(1) : null;
    const newSL_time = newSwingLow ? time(1) : null;

    prevIctState = ictState;

    // State machine transitions
    if (ictState === 0) { // NEUTRAL
        if (lastSH_price !== null && curHigh > lastSH_price) {
            ictState = 1;  // → BULLISH_WATCH
            ictBias = 0;
            pendingC3H = null;
        } else if (lastSL_price !== null && curLow < lastSL_price) {
            ictState = 3;  // → BEARISH_WATCH
            ictBias = 0;
            pendingC3L = null;
        }
    } else if (ictState === 1) { // BULLISH_WATCH
        if (newSwingLow) {
            if (lastSL_price === null || newSL_price >= lastSL_price) {
                pendingC3H = curHigh;
            } else {
                ictState = 0;
                ictBias = 0;
                pendingC3H = null;
            }
        }
        if (pendingC3H !== null && curHigh > pendingC3H) {
            ictState = 2;  // → BULLISH_ACTIVE
            ictBias = 1;
            pendingC3H = null;
        }
    } else if (ictState === 2) { // BULLISH_ACTIVE
        ictBias = 1;
        if (lastSL_price !== null && curLow < lastSL_price) {
            ictState = 3;  // → BEARISH_WATCH
            ictBias = 0;
            pendingC3L = null;
        }
    } else if (ictState === 3) { // BEARISH_WATCH
        if (newSwingHigh) {
            if (lastSH_price === null || newSH_price <= lastSH_price) {
                pendingC3L = curLow;
            } else {
                ictState = 0;
                ictBias = 0;
                pendingC3L = null;
            }
        }
        if (pendingC3L !== null && curLow < pendingC3L) {
            ictState = 4;  // → BEARISH_ACTIVE
            ictBias = -1;
            pendingC3L = null;
        }
    } else if (ictState === 4) { // BEARISH_ACTIVE
        ictBias = -1;
        if (lastSH_price !== null && curHigh > lastSH_price) {
            ictState = 1;  // → BULLISH_WATCH
            ictBias = 0;
            pendingC3H = null;
        }
    }

    // Update swing points AFTER transitions
    if (newSwingHigh) {
        lastSH_price = newSH_price;
        lastSH_time = newSH_time;
    }
    if (newSwingLow) {
        lastSL_price = newSL_price;
        lastSL_time = newSL_time;
    }

    // ─── COMBINED SIGNAL ────────────────────────────────────────────
    // PCX target from previous bar predicts today | ICT bias is for today
    const pcxPredForToday = prevPcxTarget;
    const pcxFilteredForToday = !savedPrevAbnormal;

    let combinedSignal = 0;
    if (pcxPredForToday === 1 && ictBias === 1 && pcxFilteredForToday) {
        combinedSignal = 1;
    } else if (pcxPredForToday === -1 && ictBias === -1 && pcxFilteredForToday) {
        combinedSignal = -1;
    }

    // ─── VISUALS ────────────────────────────────────────────────────
    // plot.line must be called unconditionally every tick (registers a series)

    // Target lines: prev day high/low
    const plotHigh = showTargetLines ? prevHigh : null;
    const plotLow = showTargetLines ? prevLow : null;
    plot.line('Prev High', plotHigh, color.green);
    plot.line('Prev Low', plotLow, color.red);

    // Bar coloring
    let barColorIndex = 0;
    if (showBarColor) {
        if (combinedSignal === 1 && showCombined) {
            barColorIndex = 1;
        } else if (combinedSignal === -1 && showCombined) {
            barColorIndex = 2;
        } else if (ictBias === 1 && showICT) {
            barColorIndex = 3;
        } else if (ictBias === -1 && showICT) {
            barColorIndex = 4;
        } else if (pcxSignal === 1 && showPCX) {
            barColorIndex = 5;
        } else if (pcxSignal === -1 && showPCX) {
            barColorIndex = 6;
        }
    }
    plot.barColorer('Bias', barColorIndex, [
        { color: 'green', name: 'Combined Long' },
        { color: 'red', name: 'Combined Short' },
        { color: 'blue', name: 'ICT Long' },
        { color: 'orange', name: 'ICT Short' },
        { color: 'lime', name: 'PCX High' },
        { color: 'maroon', name: 'PCX Low' },
    ]);
};
