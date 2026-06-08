Analyze a single BEI ticker from the local database. Usage: /analyze-ticker BBCA [days=60]

Spawn the data-analyst agent with this task:

"Analyze $TICKER using the last $DAYS days of data from bei_stocks.db. Produce:

1. **Price summary**: current close, 52-week high/low (or max available), % from each
2. **Trend**: MA5 vs MA20 vs MA50 — direction (uptrend / downtrend / sideways)
3. **Momentum**: RSI(14) — overbought (>70) / neutral / oversold (<30)
4. **Volatility**: ATR(14) in IDR and as % of close; Bollinger Band width
5. **Volume**: today vs 20-day avg volume; flag if > 2x average
6. **Recent returns**: last 5 trading days with date, close, and DayReturn_Pct
7. **Signal summary**: one-line verdict (e.g. 'Oversold with rising volume — watch for reversal')

Show the date range used. Warn if fewer than 20 rows are available."

If $TICKER is not provided, ask the user which ticker to analyze.
If $DAYS is not provided, use 60.
