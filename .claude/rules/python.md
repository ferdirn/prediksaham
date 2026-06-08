# Python Style Rules

## Environment

- Always use `.venv` (Python 3.13). Never install packages globally.
- Run scripts with: `source .venv/bin/activate && python ...`
- Do not add `pyproject.toml` or `setup.py` — this is a single-file script project.

## Code style

- Follow PEP 8: 4-space indent, 100-char line limit.
- Use type hints on all function signatures.
- No docstrings unless the logic is genuinely non-obvious. The function name and type hints should be self-explanatory.
- Use f-strings, not `.format()` or `%`.

## pandas conventions

- Reset index after `yf.download()` — it returns a DatetimeIndex.
- Flatten MultiIndex columns immediately after download: `raw.columns = raw.columns.get_level_values(0)`.
- Use `.copy()` when slicing a DataFrame to avoid SettingWithCopyWarning.
- Round prices to 2 dp, percentages to 4 dp.

## SQLite conventions

- Use parameterized queries (`?` placeholders) — never f-string SQL values.
- Always `conn.close()` after use (no context manager pattern is already established).
- Use `INSERT OR REPLACE` for upserts — the `UNIQUE(Ticker, Date)` constraint handles deduplication.

## Error handling

- Print a clear warning and return an empty DataFrame on download failure — don't raise.
- Do not add try/except around SQLite operations unless handling a specific known failure mode.
- Validate ticker input by stripping whitespace and uppercasing; never pass raw user input directly to yfinance.
