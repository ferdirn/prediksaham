Screen all tickers in bei_stocks.db against a set of criteria and rank them. Usage: /screen [preset]

Available presets: momentum | oversold | breakout | volatile | default

Spawn the data-analyst agent with this task:

"Screen all tickers in bei_stocks.db using the last 30 days of data. For each ticker compute:
- MA5, MA20, MA50
- RSI(14)
- ATR(14) as % of close (volatility score)
- 5-day return %
- Volume ratio: last close volume / 20-day avg volume

Apply the '$PRESET' filter:
- **momentum**: MA5 > MA20 > MA50 AND RSI between 50–70 AND 5-day return > 0
- **oversold**: RSI < 35 AND MA5 < MA20 (potential reversal candidates)
- **breakout**: volume ratio > 2.0 AND DayReturn_Pct > 1.5% on most recent day
- **volatile**: ATR% in top 3 of all tickers
- **default**: show all tickers ranked by 5-day return descending

Output a ranked markdown table with columns: Rank | Ticker | Close | MA5>MA20 | RSI | 5D Return% | Vol Ratio | ATR%
Add a one-line note for each ticker that matches a signal.
Exclude tickers with fewer than 20 rows."

If $PRESET is not provided, use 'default'.
