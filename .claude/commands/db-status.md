Show the current state of bei_stocks.db — how many rows per ticker, date coverage, and data freshness.

Steps:
1. Run this query:
   sqlite3 bei_stocks.db "
     SELECT
       Ticker,
       COUNT(*)         AS rows,
       MIN(Date)        AS oldest,
       MAX(Date)        AS latest,
       MAX(UpdatedAt)   AS last_downloaded,
       ROUND(AVG(Volume)/1000000.0, 2) AS avg_vol_M
     FROM daily_prices
     GROUP BY Ticker
     ORDER BY Ticker;
   "

2. Format the output as a markdown table with columns: Ticker | Rows | Oldest | Latest | Last Downloaded | Avg Vol (M)

3. Flag any ticker where:
   - `latest` date is more than 5 calendar days ago → mark as STALE
   - `rows` < 10 → mark as SPARSE

4. Show total row count:
   sqlite3 bei_stocks.db "SELECT COUNT(*) as total_rows FROM daily_prices;"
