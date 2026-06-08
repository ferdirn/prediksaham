# BEI Stock Predictor

Stock price prediction pipeline for Bursa Efek Indonesia (BEI/IDX) using three complementary models:

| Model | Target | Script |
|---|---|---|
| LSTM | Next-day Close price (IDR) | `lstm_predictor.py` |
| Ridge Regression | Next-day return magnitude (%) | `ridge_predictor.py` |
| Logistic Regression | Next-day direction (up/down) | `logistic_classifier.py` |

Data is sourced from Yahoo Finance and stored in a local SQLite database.

## Features

- Download daily OHLCV data for any BEI/IDX ticker via Yahoo Finance
- Three independent prediction models that complement each other
- Per-ticker hyperparameter search and saved optimal configs
- Batch prediction across all watchlist tickers
- Backtest mode for all predictors
- SQLite storage with idempotent upserts

## Prerequisites

- Python 3.12+
- `pip` and `venv`

## Installation

```bash
git clone <repo-url>
cd prediksaham

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Key dependencies: `yfinance`, `pandas`, `numpy`, `tensorflow==2.16.2`, `scikit-learn`, `matplotlib`.

## Usage

Always activate the virtual environment first:

```bash
source .venv/bin/activate
```

### 1. Download Data

```bash
# Default: all watchlist tickers, last 5 years
python bei_stock_downloader.py

# Single ticker, last 30 days
python bei_stock_downloader.py --ticker BBCA --days 30

# Multiple tickers
python bei_stock_downloader.py --tickers BBCA TLKM GOTO --days 14

# All tickers in watchlist.txt
python bei_stock_downloader.py --file watchlist.txt --days 30

# By years
python bei_stock_downloader.py --ticker BBCA --years 5
```

Edit `watchlist.txt` to manage your default ticker list (one IDX code per line, `#` for comments).

### 2. LSTM — Price Prediction

```bash
# Uses saved config from lstm_configs.json automatically
python lstm_predictor.py --ticker BBCA

# Override params
python lstm_predictor.py --ticker BBCA --lookback 20 --epochs 150 --forecast 3

# Save this run's config
python lstm_predictor.py --ticker TLKM --lookback 22 --save-config

# Batch predict all configured tickers
python lstm_batch_predict.py
```

Output: ranked table printed to terminal + `predictions_YYYY-MM-DD.csv` + prediction plot saved to `prediction_images/`.

#### Hyperparameter Search (LSTM)

```bash
# Search optimal lookback for one ticker
python lstm_lookback_search.py --ticker BBCA
python lstm_lookback_search.py --ticker BBCA --start 3 --end 60

# Batch search across multiple tickers
python lstm_batch_config_search.py
python lstm_batch_config_search.py --tickers ANTM CLEO
python lstm_batch_config_search.py --force   # re-run even if config exists
```

Results saved to `ticker_configs_research/{TICKER}_lookback_search.csv` and `.png`. Best config is auto-saved to `lstm_configs.json`.

### 3. Ridge Regression — Return Magnitude

```bash
# Find optimal config
python ridge_config_search.py --ticker DMAS

# Predict next-day return %
python ridge_predictor.py --ticker DMAS
python ridge_predictor.py --all               # all watchlist tickers

# Backtest (last 30 trading days)
python ridge_predictor.py --ticker DMAS --backtest 30
```

### 4. Logistic Regression — Direction Classifier

```bash
# Find optimal config for all tickers in DB
python logistic_config_search.py

# Predict next-day direction (up/down + confidence)
python logistic_classifier.py --ticker DMAS
python logistic_classifier.py --all                  # all watchlist
python logistic_classifier.py --all --backtest 30    # backtest
```

### 5. View Raw Data

```bash
python stock_viewer.py --ticker BBCA --rows 20
```

## Running Tests

```bash
source .venv/bin/activate
pytest
```

## Project Structure

```
prediksaham/
├── bei_stock_downloader.py       # Download OHLCV → SQLite
├── stock_viewer.py               # View last N rows for a ticker
├── utils.py                      # Shared helpers (watchlist, config I/O)
├── watchlist.txt                 # Default ticker list
├── bei_stocks.db                 # SQLite database (local, not committed)
│
├── lstm_predictor.py             # LSTM: train + predict next-day price
├── lstm_lookback_search.py       # LSTM: lookback hyperparameter search
├── lstm_batch_config_search.py   # LSTM: batch hyperparameter search
├── lstm_batch_predict.py         # LSTM: batch prediction for all tickers
├── lstm_configs.json             # Saved optimal LSTM configs per ticker
│
├── ridge_predictor.py            # Ridge: predict next-day return %
├── ridge_config_search.py        # Ridge: hyperparameter search
├── ridge_configs.json            # Saved optimal Ridge configs
│
├── logistic_classifier.py        # Logistic: predict next-day direction
├── logistic_config_search.py     # Logistic: hyperparameter search
├── logistic_configs.json         # Saved optimal Logistic configs
│
├── ticker_configs_research/      # Lookback search outputs (CSV + PNG)
├── prediction_images/            # LSTM prediction plots
└── tests/                        # pytest test suite
```

## Database Schema

Table `daily_prices` in `bei_stocks.db`:

| Column | Type | Description |
|---|---|---|
| `Ticker` | TEXT | IDX ticker code (e.g. `BBCA`) |
| `Date` | TEXT | Trading date (`YYYY-MM-DD`) |
| `Open/High/Low/Close` | REAL | Price in IDR, split/dividend adjusted |
| `Volume` | INTEGER | Shares traded |
| `GainLoss_IDR` | REAL | Close − Open (intraday move) |
| `GainLoss_Pct` | REAL | (Close − Open) / Open × 100 |
| `DayReturn_Pct` | REAL | Day-over-day return vs previous Close |
| `IntraDay_Range` | REAL | High − Low (intraday volatility proxy) |
| `UpdatedAt` | TEXT | Timestamp of last upsert |

Uniqueness constraint: `UNIQUE(Ticker, Date)`. All upserts use `INSERT OR REPLACE`.

## LSTM Configuration Reference

Default hyperparameters (overridden per ticker by `lstm_configs.json`):

| Parameter | Default | Description |
|---|---|---|
| `LOOKBACK` | 48 | Days of history per sequence |
| `FORECAST` | 1 | Days ahead to predict |
| `LSTM_UNITS` | 64 | Neurons per LSTM layer |
| `NUM_LAYERS` | 2 | Stacked LSTM layers |
| `DROPOUT` | 0.2 | Dropout rate |
| `EPOCHS` | 300 | Max training epochs |
| `PATIENCE` | 25 | Early stopping patience |

## Notes

- BEI tickers use bare 4-letter IDX codes (`BBCA`, not `BBCA.JK`). The `.JK` suffix is added internally for Yahoo Finance.
- Prices are in IDR (Indonesian Rupiah).
- `DayReturn_Pct` for the first row of any date range is always `NaN` — skip it in return calculations.
- BEI is closed on Indonesian national holidays; date gaps are normal.
- `bei_stocks.db` is a local artifact and is not committed to git.
