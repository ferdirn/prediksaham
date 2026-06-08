# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BEI (Bursa Efek Indonesia) stock data pipeline with three prediction approaches: LSTM (harga), Ridge Regression (magnitude return), dan Logistic Regression (arah naik/turun). Downloads daily OHLCV data via Yahoo Finance into SQLite.

## Environment Setup

Python 3.12, local `.venv`. Always activate before running anything:

```bash
source .venv/bin/activate
```

Key dependencies: `yfinance`, `pandas`, `tensorflow==2.16.2`, `scikit-learn`, `matplotlib`.

## Project Files

```
— Data
bei_stock_downloader.py        — Download OHLCV data from Yahoo Finance → SQLite
stock_viewer.py                — View last N rows for a ticker from the DB
watchlist.txt                  — One IDX ticker per line (no .JK suffix, # = comment)
bei_stocks.db                  — SQLite database (local artifact, not committed)

— LSTM Prediction (prediksi harga)
lstm_predictor.py              — Train LSTM, predict next-day Close price for one ticker
lstm_lookback_search.py        — Lookback hyperparameter search (3–60) for one ticker
lstm_batch_config_search.py    — Batch lookback search for multiple tickers
lstm_batch_predict.py          — Predict next-day price for all configured tickers
lstm_configs.json              — Saved optimal LSTM config per ticker
ticker_configs_research/       — Lookback search outputs: CSV + PNG per ticker
prediction_images/             — LSTM prediction plots: {TICKER}_lstm_prediction.png
predictions_YYYY-MM-DD.csv     — Batch LSTM prediction results per run date

— Ridge Regression (prediksi magnitude return %)
ridge_predictor.py             — Predict next-day DayReturn_Pct using Ridge regression
ridge_config_search.py         — Search optimal lookback + alpha per ticker
ridge_configs.json             — Saved optimal Ridge config per ticker

— Logistic Regression (prediksi arah naik/turun)
logistic_classifier.py         — Predict next-day direction (up/down) + confidence
logistic_config_search.py      — Search optimal lookback + C per ticker
logistic_configs.json          — Saved optimal Logistic config per ticker
```

## Data Downloader

```bash
python bei_stock_downloader.py --ticker BBCA --days 30
python bei_stock_downloader.py --tickers BBCA TLKM GOTO --days 14
python bei_stock_downloader.py --file watchlist.txt --days 30
python bei_stock_downloader.py --ticker BBCA --years 5
```

`bei_stock_downloader.py` — four logical sections:
1. `init_db` — creates `daily_prices` table with `UNIQUE(Ticker, Date)`, safe to call repeatedly
2. `download_stock` — fetches via Yahoo Finance `.JK` suffix, `auto_adjust=True`, computes `GainLoss_IDR`, `GainLoss_Pct`, `DayReturn_Pct`, `IntraDay_Range`. `Frequency` always `NULL`.
3. `save` — upserts via `INSERT OR REPLACE`, idempotent
4. Query helpers: `get_ticker`, `get_date_range`, `list_tickers`

## LSTM Predictor

```bash
# Uses saved config from lstm_configs.json automatically
python lstm_predictor.py --ticker BBCA

# Override params manually
python lstm_predictor.py --ticker BBCA --lookback 20 --epochs 150 --forecast 3

# Save current run's config for this ticker
python lstm_predictor.py --ticker TLKM --lookback 22 --save-config
```

Key CONFIG constants in `lstm_predictor.py` (overridden by `lstm_configs.json` if available):

| Constant | Default | Description |
|---|---|---|
| `LOOKBACK` | 48 | Days of history the model sees per sequence |
| `FORECAST` | 1 | Days ahead to predict |
| `FEATURES` | `[Close, Volume, GainLoss_Pct, DayReturn_Pct]` | Input features |
| `LSTM_UNITS` | 64 | Neurons per LSTM layer |
| `NUM_LAYERS` | 2 | Stacked LSTM layers |
| `DROPOUT` | 0.2 | Dropout rate |
| `EPOCHS` | 300 | Max training epochs (early stopping applies) |
| `PATIENCE` | 25 | Early stopping patience |
| `SEED` | 42 | Fixed for reproducibility |

## Hyperparameter Research

```bash
# Search optimal lookback for one ticker → saves to ticker_configs_research/
python lstm_lookback_search.py --ticker BBCA
python lstm_lookback_search.py --ticker BBCA --start 3 --end 60

# Batch search for multiple tickers
python lstm_batch_config_search.py
python lstm_batch_config_search.py --tickers ANTM CLEO
python lstm_batch_config_search.py --force   # re-run even if config exists
```

Research outputs go to `ticker_configs_research/`:
- `{TICKER}_lookback_search.csv` — MAPE/MAE/RMSE per lookback value
- `{TICKER}_lookback_search.png` — plot of MAPE curve

Best config per ticker is auto-saved to `lstm_configs.json`.

## Batch Prediction (LSTM)

```bash
# Predict next-day price for all tickers with saved configs
python lstm_batch_predict.py
```

Output: ranked table + `predictions_YYYY-MM-DD.csv`.

## lstm_configs.json

Stores optimal LSTM hyperparameters per ticker. `lstm_predictor.py` reads this automatically — no need to pass `--lookback` or other flags for configured tickers.

## Ridge Regression

```bash
python ridge_config_search.py --ticker DMAS          # cari config optimal
python ridge_predictor.py --ticker DMAS               # prediksi satu ticker
python ridge_predictor.py --all                       # semua watchlist
python ridge_predictor.py --ticker DMAS --backtest 30 # backtest
```

## Logistic Regression (Classifier)

```bash
python logistic_config_search.py                         # riset semua ticker di DB
python logistic_classifier.py --all                      # prediksi arah semua watchlist
python logistic_classifier.py --all --backtest 30        # backtest semua watchlist
python logistic_classifier.py --ticker DMAS --backtest 30
```

## Database

`bei_stocks.db` — SQLite, single table `daily_prices`.

Schema: `Ticker, Date, Open, High, Low, Close, Volume, Frequency(NULL), GainLoss_IDR, GainLoss_Pct, DayReturn_Pct, IntraDay_Range, UpdatedAt`

Primary lookup: `WHERE Ticker = ? AND Date BETWEEN ? AND ?`

## .claude Folder Structure

```
.claude/
├── settings.json
├── settings.local.json        — local overrides (not committed)
├── agents/
│   ├── data-analyst.md        — TA analysis and screening from DB
│   ├── downloader.md          — data download management
│   └── optimus.md             — hyperparameter optimization per ticker
├── commands/
│   ├── download-watchlist.md  → /download-watchlist [days]
│   ├── db-status.md           → /db-status
│   ├── refresh.md             → /refresh [days]
│   ├── analyze-ticker.md      → /analyze-ticker TICKER [days]
│   └── screen.md              → /screen [preset]
└── rules/
    ├── python.md
    ├── database.md
    └── finance-data.md
```

### Slash commands

| Command | Description |
|---|---|
| `/download-watchlist [days]` | Download all watchlist tickers (default: 30 days) |
| `/db-status` | Row counts, date coverage, staleness per ticker |
| `/refresh [days]` | Quick update — last N days (default: 7) |
| `/analyze-ticker TICKER [days]` | Full TA analysis for one ticker |
| `/screen [preset]` | Screen tickers: `momentum`, `oversold`, `breakout`, `volatile` |

### Sub-agents

| Agent | Trigger | Description |
|---|---|---|
| `data-analyst` | "analisis", "screen", "bandingkan" | TA from DB: MA, RSI, ATR, ranking |
| `downloader` | "download", "update data" | Fetch/refresh data, troubleshoot |
| `optimus` | "riset config", "cari lookback optimal" | Lookback search, save config to lstm_configs.json |
