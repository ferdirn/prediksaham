"""
BEI Combined Predictor
=======================
Menjalankan ketiga model (LSTM, Ridge, Logistic) untuk semua ticker
di watchlist dan menampilkan hasil dalam satu tabel ringkas.

Usage:
    python combined_predict.py
    python combined_predict.py --tickers BBCA ANTM DMAS
    python combined_predict.py --no-lstm
    python combined_predict.py --no-ridge
    python combined_predict.py --no-logistic
"""

import argparse
import os
import warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

from datetime import date
from utils import load_watchlist
from lstm_predictor import Spinner

# ── imports per model ─────────────────────────────────────────────────────────

from ridge_predictor import predict_ticker as ridge_predict
from logistic_classifier import predict_ticker as logistic_predict


def run_lstm(ticker: str, configs: dict) -> dict | None:
    cfg = configs.get(ticker.upper())
    if not cfg:
        return None
    from lstm_batch_predictor import predict_ticker as lstm_predict
    return lstm_predict(ticker, cfg)


def run_ridge(ticker: str) -> dict | None:
    return ridge_predict(ticker)


def run_logistic(ticker: str) -> dict | None:
    return logistic_predict(ticker)


# ── formatting helpers ────────────────────────────────────────────────────────

def fmt_lstm(r: dict | None) -> tuple[str, str]:
    if not r:
        return "—", "—"
    arrow = "▲" if r["change_pct"] >= 0 else "▼"
    return f"{r['forecast']:,.0f}", f"{arrow}{abs(r['change_pct']):.2f}%"


def fmt_ridge(r: dict | None) -> str:
    if not r:
        return "—"
    arrow = "▲" if r["pred_return"] >= 0 else "▼"
    return f"{arrow}{abs(r['pred_return']):.2f}%"


def fmt_logistic(r: dict | None) -> tuple[str, str]:
    if not r:
        return "—", "—"
    conf = r["prob_up"] if r["pred_dir"] == 1 else 1 - r["prob_up"]
    label = "▲ NAIK" if r["pred_dir"] == 1 else "▼ TURUN"
    return label, f"{conf*100:.1f}%"


def make_rekomendasi(n_up: int, n_down: int, n_models: int) -> str:
    if n_models == 0:
        return "—"
    if n_up > n_down:
        return "BELI KUAT" if n_up == n_models else "BELI"
    if n_down > n_up:
        return "JUAL KUAT" if n_down == n_models else "JUAL"
    return "NETRAL"


# ── combined backtest ─────────────────────────────────────────────────────────

def run_backtest(
    n_days: int,
    tickers: list[str] | None = None,
    use_lstm: bool = True,
    use_ridge: bool = True,
    use_logistic: bool = True,
) -> None:
    from lstm_predictor import run_backtest as lstm_backtest
    from ridge_predictor import run_backtest as ridge_backtest
    from logistic_classifier import run_backtest as logistic_backtest

    if tickers:
        tickers = [t.strip().upper() for t in tickers]
    else:
        tickers = load_watchlist()

    results = []
    for ticker in tickers:
        print(f"\n{'─'*60}")
        print(f"  {ticker}")
        print(f"{'─'*60}")
        r_lstm = lstm_backtest(ticker, n_days) if use_lstm else None
        r_ridge    = ridge_backtest(ticker, n_days, print_detail_rows=False) if use_ridge    else None
        r_logistic = logistic_backtest(ticker, n_days, print_detail_rows=False) if use_logistic else None
        results.append({
            "ticker"      : ticker,
            "lstm_mape"   : r_lstm["mape"]    if r_lstm    else None,
            "lstm_dir"    : r_lstm["dir_acc"] if r_lstm    else None,
            "ridge_mae"   : r_ridge["mae"]    if r_ridge   else None,
            "ridge_dir"   : r_ridge["dir_acc"]if r_ridge   else None,
            "logistic_dir": r_logistic["acc"] if r_logistic else None,
        })

    W = 85
    print(f"\n{'═'*W}")
    print(f"  RINGKASAN BACKTEST GABUNGAN — {n_days} hari — {date.today()}")
    print(f"{'═'*W}")
    print(f"  {'Ticker':<8} {'LSTM MAPE':>10} {'LSTM Dir%':>10} {'Ridge MAE':>10} "
          f"{'Ridge Dir%':>11} {'Logic Dir%':>11}")
    print(f"  {'─'*8} {'─'*10} {'─'*10} {'─'*10} {'─'*11} {'─'*11}")
    for r in results:
        lstm_mape = f"{r['lstm_mape']:.2f}%"  if r["lstm_mape"]    is not None else "—"
        lstm_dir  = f"{r['lstm_dir']:.1f}%"   if r["lstm_dir"]     is not None else "—"
        ridge_mae = f"{r['ridge_mae']:.4f}%"  if r["ridge_mae"]    is not None else "—"
        ridge_dir = f"{r['ridge_dir']:.1f}%"  if r["ridge_dir"]    is not None else "—"
        log_dir   = f"{r['logistic_dir']:.1f}%"if r["logistic_dir"] is not None else "—"
        print(f"  {r['ticker']:<8} {lstm_mape:>10} {lstm_dir:>10} {ridge_mae:>10} "
              f"{ridge_dir:>11} {log_dir:>11}")
    print(f"{'═'*W}\n")


# ── main ──────────────────────────────────────────────────────────────────────

def run(
    tickers: list[str] | None = None,
    use_lstm: bool = True,
    use_ridge: bool = True,
    use_logistic: bool = True,
) -> None:
    from lstm_batch_predictor import load_configs
    lstm_configs = load_configs() if use_lstm else {}

    if tickers:
        tickers = [t.strip().upper() for t in tickers]
    else:
        tickers = load_watchlist()
    rows = []

    print(f"\n  Memproses {len(tickers)} ticker...\n", flush=True)
    for i, ticker in enumerate(tickers, 1):
        spinner = Spinner(f"  [{i}/{len(tickers)}] {ticker:<8}")
        spinner.start()

        r_lstm     = run_lstm(ticker, lstm_configs) if use_lstm     else None
        r_ridge    = run_ridge(ticker)              if use_ridge    else None
        r_logistic = run_logistic(ticker)           if use_logistic else None

        spinner.stop()

        lstm_ok = "✓" if r_lstm     else "—"
        rdg_ok  = "✓" if r_ridge    else "—"
        log_ok  = "✓" if r_logistic else "—"
        print(f"  [{i}/{len(tickers)}] {ticker:<8} LSTM {lstm_ok}  Ridge {rdg_ok}  Logistic {log_ok}")

        last_close = (
            r_lstm["last_close"] if r_lstm else
            r_ridge["last_close"] if r_ridge else
            r_logistic["last_close"] if r_logistic else None
        )

        rows.append({
            "ticker"        : ticker,
            "last_close"    : last_close,
            "lstm_r"        : r_lstm,
            "ridge_r"       : r_ridge,
            "logistic_r"    : r_logistic,
        })

    # ── hitung konsensus per row ──────────────────────────────────────────────
    table = []
    for row in rows:
        lstm_up     = row["lstm_r"]["change_pct"] > 0    if row["lstm_r"]     else None
        ridge_up    = row["ridge_r"]["pred_return"] > 0  if row["ridge_r"]    else None
        logistic_up = row["logistic_r"]["pred_dir"] == 1 if row["logistic_r"] else None

        signals  = [v for v in [lstm_up, ridge_up, logistic_up] if v is not None]
        n_up     = sum(signals)
        n_models = len(signals)
        n_down   = n_models - n_up
        skor_num = max(n_up, n_down)

        l_price, l_chg  = fmt_lstm(row["lstm_r"])
        r_ret            = fmt_ridge(row["ridge_r"])
        lg_dir, lg_conf  = fmt_logistic(row["logistic_r"])
        lc               = f"{row['last_close']:>11,.0f}" if row["last_close"] else f"{'—':>11}"

        if row["logistic_r"]:
            prob_up  = row["logistic_r"]["prob_up"]
            conf_val = prob_up if row["logistic_r"]["pred_dir"] == 1 else 1 - prob_up
        else:
            conf_val = 0.0

        table.append({
            "ticker"       : row["ticker"],
            "lc"           : lc,
            "l_price"      : l_price,
            "l_chg"        : l_chg,
            "r_ret"        : r_ret,
            "lg_dir"       : lg_dir,
            "lg_conf"      : lg_conf,
            "sinyal"       : f"{skor_num}/{n_models}",
            "rekomendasi"  : make_rekomendasi(n_up, n_down, n_models),
            "_skor_num"    : skor_num,
            "_conf_val"    : conf_val,
        })

    sort_key = lambda x: (-x["_skor_num"], -x["_conf_val"])
    jual_rows   = sorted([t for t in table if t["rekomendasi"].startswith("JUAL")],   key=sort_key)
    netral_rows = [t for t in table if t["rekomendasi"] == "NETRAL"]
    beli_rows   = sorted([t for t in table if t["rekomendasi"].startswith("BELI")],   key=sort_key)

    # ── tabel ─────────────────────────────────────────────────────────────────
    W = 112
    header = (f"  {'Ticker':<7} {'Last Close':>11}  {'LSTM Forecast':>13} {'LSTM Chg%':>9}  "
              f"{'Ridge Ret%':>10}  {'Logistic':>8} {'Conf':>6}  {'Sinyal':>6}  {'Rekomendasi':>11}")
    divider = (f"  {'─'*7} {'─'*11}  {'─'*13} {'─'*9}  {'─'*10}  {'─'*8} {'─'*6}  {'─'*6}  {'─'*11}")

    def print_row(t: dict) -> None:
        print(f"  {t['ticker']:<7} {t['lc']}  {t['l_price']:>13} {t['l_chg']:>9}  "
              f"{t['r_ret']:>10}  {t['lg_dir']:>8} {t['lg_conf']:>6}  {t['sinyal']:>6}  {t['rekomendasi']:>11}")

    print(f"\n{'═'*W}")
    print(f"  PREDIKSI BESOK — {date.today()}")
    print(f"{'═'*W}")

    if jual_rows:
        print(f"  ▼ JUAL")
        print(header)
        print(divider)
        for t in jual_rows:
            print_row(t)

    if netral_rows:
        if jual_rows:
            print(f"  {'·'*W}")
        print(f"  ◆ NETRAL")
        print(header)
        print(divider)
        for t in netral_rows:
            print_row(t)

    if beli_rows:
        if jual_rows or netral_rows:
            print(f"  {'·'*W}")
        print(f"  ▲ BELI")
        print(header)
        print(divider)
        for t in beli_rows:
            print_row(t)

    print(f"{'═'*W}")
    print(f"  LSTM: prediksi harga Close  |  Ridge: estimasi DayReturn%  |  Logistic: arah + confidence")
    print(f"  Rekomendasi: BELI KUAT / BELI / NETRAL / JUAL / JUAL KUAT  (berdasarkan konsensus ketiga model)")
    print(f"{'═'*W}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="combined_predict.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "BEI Combined Predictor\n"
            "======================\n"
            "Menjalankan ketiga model prediksi (LSTM, Ridge, Logistic) untuk semua\n"
            "ticker di watchlist dan menampilkan hasil dalam satu tabel ringkas.\n\n"
            "Kolom output:\n"
            "  Ticker       — kode saham IDX\n"
            "  Last Close   — harga penutupan terakhir (IDR)\n"
            "  LSTM Forecast— prediksi harga Close besok (IDR)\n"
            "  LSTM Chg%    — estimasi perubahan harga dari LSTM\n"
            "  Ridge Ret%   — estimasi DayReturn% besok dari Ridge Regression\n"
            "  Logistic     — arah prediksi: ▲ NAIK / ▼ TURUN\n"
            "  Conf         — confidence dari model Logistic\n"
            "  Sinyal       — konsensus (misal 3/3 = semua model sepakat)"
        ),
        epilog=(
            "Alur kerja yang disarankan:\n"
            "  1. Download data     : python bei_stock_downloader.py --file watchlist.txt --days 30\n"
            "  2. Riset config LSTM : python lstm_batch_config_search.py\n"
            "  3. Riset config Ridge: python ridge_config_search.py\n"
            "  4. Riset config Logis: python logistic_config_search.py\n"
            "  5. Prediksi gabungan : python combined_predict.py\n"
            "  6. Backtest gabungan : python combined_predict.py --backtest 30\n"
            "\n"
            "Contoh:\n"
            "  python combined_predict.py\n"
            "  python combined_predict.py --tickers BBCA ANTM DMAS\n"
            "  python combined_predict.py --no-lstm\n"
            "  python combined_predict.py --tickers WIFI INET --no-ridge\n"
            "  python combined_predict.py --backtest 30\n"
            "  python combined_predict.py --tickers BBCA ANTM --backtest 20\n\n"
            "Catatan:\n"
            "  - LSTM hanya tersedia untuk ticker yang ada di lstm_configs.json\n"
            "  - Tabel diurutkan: Sinyal tertinggi → Confidence tertinggi\n"
            "  - Rekomendasi: BELI KUAT / BELI / NETRAL / JUAL / JUAL KUAT\n"
            "  - --backtest N: LSTM dilatih sekali per ticker (lambat tapi akurat)\n"
        ),
    )
    parser.add_argument(
        "--tickers", "--ticker", nargs="+", metavar="TICKER",
        help="Ticker spesifik (default: semua dari watchlist.txt)"
    )
    parser.add_argument(
        "--no-lstm", action="store_true",
        help="Lewati model LSTM"
    )
    parser.add_argument(
        "--no-ridge", action="store_true",
        help="Lewati model Ridge Regression"
    )
    parser.add_argument(
        "--no-logistic", action="store_true",
        help="Lewati model Logistic Regression"
    )
    parser.add_argument(
        "--backtest", type=int, default=None, metavar="N",
        help="Uji mundur N hari: jalankan backtest ketiga model dan tampilkan ringkasan akurasi"
    )
    args = parser.parse_args()
    if args.backtest is not None:
        run_backtest(
            args.backtest,
            tickers=args.tickers,
            use_lstm=not args.no_lstm,
            use_ridge=not args.no_ridge,
            use_logistic=not args.no_logistic,
        )
    else:
        run(
            tickers=args.tickers,
            use_lstm=not args.no_lstm,
            use_ridge=not args.no_ridge,
            use_logistic=not args.no_logistic,
        )
