Download all tickers in watchlist.txt for the last $DAYS days (default: 30).

Usage: /download-watchlist [days]

Steps:
1. Read watchlist.txt and show which tickers will be downloaded
2. Run: source .venv/bin/activate && python bei_stock_downloader.py --file watchlist.txt --days $DAYS
3. After completion, show a summary table of rows saved per ticker by querying bei_stocks.db:
   sqlite3 bei_stocks.db "SELECT Ticker, COUNT(*) as rows, MAX(Date) as latest FROM daily_prices GROUP BY Ticker ORDER BY Ticker"
4. Highlight any tickers that returned 0 rows (possible delisting or wrong symbol)

If $DAYS is not provided, use 30.
