from __future__ import annotations

import json
import sqlite3
from pathlib import Path
import pandas as pd


class Storage:
    def __init__(self, path: str):
        self.path = Path(path)
        self.db = sqlite3.connect(self.path)
        self.db.execute("CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self.db.execute("""CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, side TEXT, opened_at TEXT,
            closed_at TEXT DEFAULT CURRENT_TIMESTAMP, entry REAL, exit REAL, probability REAL,
            reason TEXT, pnl REAL, equity_after REAL, payload TEXT)""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS candles (
            symbol TEXT NOT NULL, interval TEXT NOT NULL, open_time INTEGER NOT NULL,
            open REAL, high REAL, low REAL, close REAL, volume REAL, close_time INTEGER,
            quote_volume REAL, trades REAL, taker_base REAL, taker_quote REAL,
            PRIMARY KEY(symbol, interval, open_time))""")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_candles_lookup ON candles(symbol, interval, open_time DESC)")
        self.db.execute("""CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT NOT NULL, signal_time INTEGER NOT NULL,
            model_version TEXT NOT NULL, side TEXT NOT NULL, raw_p_up REAL, raw_p_down REAL,
            p_up REAL, p_down REAL, probability REAL, price REAL, tp_pct REAL, sl_pct REAL,
            horizon_bars INTEGER, expected_return REAL, calibration_samples INTEGER,
            validation_precision REAL, validation_signals INTEGER, outcome INTEGER,
            resolved_at INTEGER, UNIQUE(symbol, signal_time, model_version))""")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_predictions_unresolved ON predictions(outcome, signal_time)")
        self.db.commit()

    def get(self, key: str, default=None):
        row = self.db.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set(self, key: str, value) -> None:
        self.db.execute("INSERT INTO state(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, json.dumps(value)))
        self.db.commit()

    def save_trade(self, position, reason: str, exit_price: float, pnl: float, equity: float) -> None:
        self.db.execute("INSERT INTO trades(symbol,side,opened_at,entry,exit,probability,reason,pnl,equity_after,payload) VALUES(?,?,?,?,?,?,?,?,?,?)",
                        (position.symbol, position.side, position.opened_at, position.entry_price, exit_price,
                         position.probability, reason, pnl, equity, json.dumps(position.to_dict())))
        self.db.commit()

    def candle_count(self, symbol: str, interval: str) -> int:
        return int(self.db.execute("SELECT COUNT(*) FROM candles WHERE symbol=? AND interval=?", (symbol, interval)).fetchone()[0])

    def latest_candle_open_ms(self, symbol: str, interval: str) -> int | None:
        row = self.db.execute("SELECT MAX(open_time) FROM candles WHERE symbol=? AND interval=?", (symbol, interval)).fetchone()
        return int(row[0]) if row and row[0] is not None else None

    def upsert_candles(self, symbol: str, interval: str, df: pd.DataFrame, keep: int) -> None:
        if df.empty:
            return
        rows = []
        for x in df.itertuples():
            rows.append((symbol, interval, int(x.open_time.timestamp() * 1000), x.open, x.high, x.low,
                         x.close, x.volume, int(x.close_time.timestamp() * 1000), x.quote_volume,
                         x.trades, x.taker_base, x.taker_quote))
        self.db.executemany("""INSERT INTO candles VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(symbol,interval,open_time) DO UPDATE SET
            open=excluded.open,high=excluded.high,low=excluded.low,close=excluded.close,
            volume=excluded.volume,close_time=excluded.close_time,quote_volume=excluded.quote_volume,
            trades=excluded.trades,taker_base=excluded.taker_base,taker_quote=excluded.taker_quote""", rows)
        self.db.execute("""DELETE FROM candles WHERE symbol=? AND interval=? AND open_time NOT IN
            (SELECT open_time FROM candles WHERE symbol=? AND interval=? ORDER BY open_time DESC LIMIT ?)""",
            (symbol, interval, symbol, interval, keep))
        self.db.commit()

    def load_candles(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        rows = self.db.execute("""SELECT open_time,open,high,low,close,volume,close_time,quote_volume,
            trades,taker_base,taker_quote FROM candles WHERE symbol=? AND interval=?
            ORDER BY open_time DESC LIMIT ?""", (symbol, interval, limit)).fetchall()
        columns = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades", "taker_base", "taker_quote"]
        df = pd.DataFrame(reversed(rows), columns=columns)
        if not df.empty:
            df["open_time"] = pd.to_datetime(df.open_time, unit="ms", utc=True)
            df["close_time"] = pd.to_datetime(df.close_time, unit="ms", utc=True)
        return df

    def save_prediction(self, prediction, model_version: str, tp: float, sl: float, horizon: int) -> None:
        signal_ms = int(prediction.signal_time.timestamp() * 1000)
        self.db.execute("""INSERT OR IGNORE INTO predictions(
            symbol,signal_time,model_version,side,raw_p_up,raw_p_down,p_up,p_down,probability,
            price,tp_pct,sl_pct,horizon_bars,expected_return,calibration_samples,
            validation_precision,validation_signals) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (prediction.symbol, signal_ms, model_version, prediction.side, prediction.raw_p_up,
             prediction.raw_p_down, prediction.p_up, prediction.p_down, prediction.probability,
             prediction.price, tp, sl, horizon, prediction.expected_return_on_equity,
             prediction.calibration_samples, prediction.validation_precision,
             prediction.validation_signals))
        self.db.commit()

    def unresolved_predictions(self, limit: int = 5000) -> list[dict]:
        columns = ["id", "symbol", "signal_time", "side", "price", "tp_pct", "sl_pct", "horizon_bars"]
        rows = self.db.execute("""SELECT id,symbol,signal_time,side,price,tp_pct,sl_pct,horizon_bars
            FROM predictions WHERE outcome IS NULL ORDER BY signal_time LIMIT ?""", (limit,)).fetchall()
        return [dict(zip(columns, row)) for row in rows]

    def resolve_prediction(self, prediction_id: int, outcome: int, resolved_at: int) -> None:
        self.db.execute("UPDATE predictions SET outcome=?, resolved_at=? WHERE id=?",
                        (outcome, resolved_at, prediction_id))
        self.db.commit()
