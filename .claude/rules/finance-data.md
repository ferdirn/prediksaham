# Finance Data Rules

## IDX ticker conventions

- BEI tickers are bare 4-letter IDX codes: `BBCA`, `TLKM`, `GOTO`
- Yahoo Finance requires `.JK` suffix: `BBCA.JK` — the `to_yf_ticker()` function handles this
- Never pass `BBCA.JK` directly to the downloader CLI or `download_stock()` — strip `.JK` first
- Prices are in IDR (Indonesian Rupiah). Do not assume USD unless explicitly stated.

## Yahoo Finance quirks

- `auto_adjust=True` is always used — prices are adjusted for splits and dividends
- `progress=False` suppresses the download progress bar
- yfinance >= 0.2 returns MultiIndex columns for single-ticker downloads — always flatten
- The free tier does not provide intraday data, tick data, or trade frequency
- Rate limiting: add a short sleep between ticker downloads if batch-downloading > 20 tickers

## Market calendar

- BEI trading hours: 09:00–11:30 and 13:30–15:00 WIB (UTC+7)
- BEI is closed on Indonesian national holidays — gaps in date sequences are normal
- Weekends always have no data — do not flag Saturday/Sunday gaps as errors
- If the latest date in the DB is a Friday and today is Monday, the data is current

## Derived metrics interpretation

- `GainLoss_IDR`: intraday open-to-close move (NOT day-over-day)
- `GainLoss_Pct`: intraday % move based on open price
- `DayReturn_Pct`: day-over-day % change based on previous close — this is the standard "daily return"
- `IntraDay_Range`: High - Low; a proxy for intraday volatility
- `DayReturn_Pct` for the first row in any date range is `NaN` — always skip it in return calculations

## Technical analysis conventions

- RSI(14): use 14-period standard. Overbought > 70, oversold < 30.
- MA crossovers: MA5 (short), MA20 (medium), MA50 (long)
- ATR(14): 14-period Average True Range. ATR% = ATR / Close * 100
- Volume spike: current volume > 2x the 20-day average volume
- Minimum data requirement: at least 20 rows for any meaningful technical indicator
