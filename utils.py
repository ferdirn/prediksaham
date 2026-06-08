"""
Shared utilities used across BEI predictor scripts.
"""

import json
from pathlib import Path


def load_watchlist(path: str = "watchlist.txt") -> list[str]:
    """Return tickers from a watchlist file, skipping blanks and comments."""
    lines = Path(path).read_text().splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def save_configs(config_path: str, updates: dict[str, dict]) -> None:
    """Merge `updates` into a JSON config file (atomic write via tmp)."""
    path = Path(config_path)
    configs = json.loads(path.read_text()) if path.exists() else {}
    configs.update(updates)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(configs, indent=2))
    tmp.replace(path)
    print(f"\n  Config disimpan → {config_path}")


def load_config_json(config_path: str, ticker: str, default: dict) -> dict:
    """Return saved config for `ticker` from a JSON file, or `default` if absent."""
    path = Path(config_path)
    if path.exists():
        configs = json.loads(path.read_text())
        if ticker in configs:
            return configs[ticker]
    return default
