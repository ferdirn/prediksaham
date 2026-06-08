Refresh recent data for all watchlist tickers (last 7 days by default). Usage: /refresh [days]

This is a quick update command — use it to fill in the most recent trading days.

Steps:
1. Show the most recent date in bei_stocks.db before refresh:
   sqlite3 bei_stocks.db "SELECT Ticker, MAX(Date) as latest FROM daily_prices GROUP BY Ticker ORDER BY Ticker"

2. Run the downloader:
   source .venv/bin/activate && python bei_stock_downloader.py --file watchlist.txt --days $DAYS

3. Show the most recent date after refresh and count of new rows added per ticker

4. If any ticker shows no new rows and its latest date is more than 3 days old, flag it as potentially delisted or having a data gap

If $DAYS is not provided, use 7.
