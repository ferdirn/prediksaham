# Database Rules

## Schema facts

- Table: `daily_prices` in `bei_stocks.db`
- Primary key: `id INTEGER AUTOINCREMENT`
- Uniqueness constraint: `UNIQUE(Ticker, Date)` — this is the logical primary key
- `Frequency` is always `NULL` — Yahoo Finance free tier does not expose trade count per day
- `Date` is stored as `TEXT` in `YYYY-MM-DD` format — use string comparison for date ranges

## Safe query patterns

```sql
-- Date range lookup (correct)
SELECT * FROM daily_prices
WHERE Ticker = ? AND Date BETWEEN ? AND ?
ORDER BY Date ASC;

-- Latest N rows for a ticker
SELECT * FROM daily_prices
WHERE Ticker = ?
ORDER BY Date DESC
LIMIT ?;

-- All tickers with row counts
SELECT Ticker, COUNT(*), MIN(Date), MAX(Date)
FROM daily_prices
GROUP BY Ticker
ORDER BY Ticker;
```

## Upsert pattern

Always use `INSERT OR REPLACE` — never `INSERT OR IGNORE` (which would silently skip updated prices):
```sql
INSERT OR REPLACE INTO daily_prices (Ticker, Date, Open, ...) VALUES (?, ?, ?, ...)
```

## Safety rules

- Never run `DELETE FROM daily_prices` without a `WHERE Ticker = ?` clause
- Never drop or truncate the table
- Never modify the schema unless explicitly asked — the `init_db()` function is the single source of truth
- `bei_stocks.db` is not committed to git — treat it as a local artifact

## Performance

- The index `idx_ticker_date ON daily_prices (Ticker, Date)` covers all primary lookup patterns
- For full-table scans (e.g., screening all tickers), read into pandas and compute in Python — avoid complex SQL with window functions
