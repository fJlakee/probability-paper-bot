from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import time
from pathlib import Path
import pandas as pd

from .binance import BinancePublicClient
from .config import Config
from .execution import Position, evaluate_bar, open_position
from .model import is_peg_like, load_bundle, predict_bundle, prefilter_score, save_bundle, train_bundle
from .notify import Notifier
from .storage import Storage


class PaperBot:
    def __init__(self, cfg: Config, config_path: Path):
        self.cfg = cfg
        db_path = Path(cfg.storage.database)
        if not db_path.is_absolute():
            db_path = config_path.parent / db_path
        model_dir = Path(cfg.storage.model_directory)
        self.model_dir = model_dir if model_dir.is_absolute() else config_path.parent / model_dir
        self.storage = Storage(str(db_path))
        self.client = BinancePublicClient(cfg.binance.base_url)
        self.notifier = Notifier(cfg.telegram.enabled, cfg.telegram.bot_token, cfg.telegram.chat_id)

    def _position(self) -> Position | None:
        raw = self.storage.get("position")
        return Position(**raw) if raw else None

    def _equity(self) -> float:
        return float(self.storage.get("equity", self.cfg.execution.initial_capital))

    def _symbols(self) -> list[str]:
        c = self.cfg.binance
        cached = self.storage.get("universe")
        refreshed = float(self.storage.get("universe_refreshed_at", 0))
        if cached and time.time() - refreshed < c.universe_refresh_minutes * 60:
            return [s for s in cached if s not in c.symbol_denylist][:c.universe_size]
        symbols = self.client.top_symbols(c.quote_asset, c.universe_size, c.symbol_allowlist, c.symbol_denylist)
        self.storage.set("universe", symbols)
        self.storage.set("universe_refreshed_at", time.time())
        return symbols

    def _download(self, symbol: str, count: int, latest: int | None):
        c = self.cfg.binance
        if count < c.history_bars:
            return symbol, self.client.historical_klines(symbol, c.interval, c.history_bars)
        # Start at the cached candle so the still-open candle is safely replaced.
        return symbol, self.client.klines(symbol, c.interval, c.incremental_bars, latest)

    def _update_market_data(self, symbols: list[str]) -> dict[str, object]:
        c = self.cfg.binance
        frames = {}
        with ThreadPoolExecutor(max_workers=c.download_workers) as pool:
            cache_state = {symbol: (self.storage.candle_count(symbol, c.interval),
                                    self.storage.latest_candle_open_ms(symbol, c.interval)) for symbol in symbols}
            futures = {pool.submit(self._download, symbol, *cache_state[symbol]): symbol for symbol in symbols}
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    _, fresh = future.result()
                    self.storage.upsert_candles(symbol, c.interval, fresh, c.history_bars)
                    frames[symbol] = self.storage.load_candles(symbol, c.interval, c.history_bars)
                except Exception as exc:
                    print(f"data skip {symbol}: {exc}", flush=True)
        return frames

    def _bundle_path(self, symbol: str) -> Path:
        m = self.cfg.model
        signature = f"v2_h{m.horizon_bars}_tp{m.take_profit_pct:g}_sl{m.stop_loss_pct:g}"
        return self.model_dir / f"{symbol}_{signature}.joblib"

    def _bundle_fresh(self, bundle) -> bool:
        try:
            trained = datetime.fromisoformat(bundle.trained_at)
            return (datetime.now(timezone.utc) - trained).total_seconds() < self.cfg.model.retrain_minutes * 60
        except (TypeError, ValueError):
            return False

    def _prediction(self, symbol: str, df):
        m, e = self.cfg.model, self.cfg.execution
        closed = df.iloc[:-1]  # Binance includes the currently open candle
        path = self._bundle_path(symbol)
        bundle = load_bundle(path)
        if bundle is None or not self._bundle_fresh(bundle):
            bundle = train_bundle(symbol, closed, m.horizon_bars, m.take_profit_pct,
                                  m.stop_loss_pct, m.calibration_fraction, m.test_fraction,
                                  m.min_training_rows, m.validation_signal_threshold,
                                  m.random_state)
            if bundle is None:
                return None
            save_bundle(bundle, path)
        roundtrip_cost = 2 * (e.taker_fee_rate + e.slippage_rate)
        return predict_bundle(bundle, closed, m.take_profit_pct, m.stop_loss_pct,
                              roundtrip_cost, e.notional_fraction)

    def _monitor_position(self, position: Position) -> bool:
        c, e = self.cfg.binance, self.cfg.execution
        last_bar = self.storage.get("last_position_bar")
        start_ms = int(pd.Timestamp(last_bar).timestamp() * 1000) + 1 if last_bar else None
        df = (self.client.klines_since(position.symbol, c.interval, start_ms)
              if start_ms else self.client.klines(position.symbol, c.interval, 3))
        now = pd.Timestamp.now(tz="UTC")
        closed = df[df.close_time < now]
        if closed.empty:
            return True
        for bar in closed.itertuples():
            position.bars_open += 1
            result = evaluate_bar(position, float(bar.high), float(bar.low), float(bar.close),
                                  e.taker_fee_rate, e.slippage_rate, e.liquidation_penalty_rate,
                                  position.bars_open >= e.max_position_bars)
            self.storage.set("last_position_bar", bar.close_time.isoformat())
            if result:
                reason, exit_price, pnl = result
                equity = max(0.0, self._equity() + pnl)
                self.storage.save_trade(position, reason, exit_price, pnl, equity)
                self.storage.set("equity", equity)
                self.storage.set("position", None)
                self.notifier.send(f"CLOSE {position.side} {position.symbol}\nReason: {reason}\nEntry: {position.entry_price:.8g}\nExit: {exit_price:.8g}\nQty: {position.quantity:.6g}\nP&L: ${pnl:.2f}\nEquity: ${equity:.2f}")
                return True
        if self._position():
            self.storage.set("position", position.to_dict())
        return True

    def _resolve_predictions(self, frames: dict) -> None:
        for item in self.storage.unresolved_predictions():
            df = frames.get(item["symbol"])
            if df is None or df.empty:
                continue
            future = df[df.close_time > pd.to_datetime(item["signal_time"], unit="ms", utc=True)].iloc[:item["horizon_bars"]]
            if len(future) < item["horizon_bars"]:
                continue
            price, tp, sl = item["price"], item["tp_pct"], item["sl_pct"]
            success = 0
            for bar in future.itertuples():
                if item["side"] == "LONG":
                    hit_tp, hit_sl = bar.high >= price * (1 + tp), bar.low <= price * (1 - sl)
                else:
                    hit_tp, hit_sl = bar.low <= price * (1 - tp), bar.high >= price * (1 + sl)
                if hit_tp or hit_sl:
                    success = int(hit_tp and not hit_sl)
                    break
            self.storage.resolve_prediction(item["id"], success, int(future.close_time.iloc[-1].timestamp() * 1000))

    def cycle(self) -> None:
        c, m, e = self.cfg.binance, self.cfg.model, self.cfg.execution
        position = self._position()
        if position:
            self._monitor_position(position)
            return

        symbols = self._symbols()
        frames = self._update_market_data(symbols)
        self._resolve_predictions(frames)
        scored = []
        for symbol, df in frames.items():
            try:
                if c.exclude_peg_like and is_peg_like(df.iloc[:-1], c.peg_price_min,
                                                      c.peg_price_max, c.peg_max_atr_pct):
                    print(f"stable/peg skip {symbol}", flush=True)
                    continue
                scored.append((prefilter_score(df.iloc[:-1]), symbol))
            except Exception as exc:
                print(f"filter skip {symbol}: {exc}", flush=True)
        selected = [symbol for _, symbol in sorted(scored, reverse=True)[:m.prefilter_size]]

        candidates = []
        with ThreadPoolExecutor(max_workers=m.model_workers) as pool:
            futures = {pool.submit(self._prediction, symbol, frames[symbol]): symbol for symbol in selected}
            for future in as_completed(futures):
                try:
                    pred = future.result()
                    if pred:
                        candidates.append(pred)
                        self.storage.save_prediction(pred, "v2-platt", m.take_profit_pct,
                                                     m.stop_loss_pct, m.horizon_bars)
                except Exception as exc:
                    print(f"model skip {futures[future]}: {exc}", flush=True)

        eligible = [x for x in candidates if x.probability >= m.min_entry_probability
                    and x.expected_return_on_equity >= e.min_expected_return_on_equity
                    and x.calibration_samples >= m.min_calibration_samples
                    and x.validation_signals >= m.min_validation_trades
                    and x.validation_precision >= m.min_validation_precision]
        if not eligible:
            best = max(candidates, key=lambda x: x.expected_return_on_equity, default=None)
            print(f"Scanned={len(frames)} ML={len(selected)}. No eligible signal. Best={best}", flush=True)
            return
        best = max(eligible, key=lambda x: (x.expected_return_on_equity, x.probability))
        equity = self._equity()
        position = open_position(best.symbol, best.side, best.price, equity, e.leverage,
                                 e.notional_fraction, m.take_profit_pct, m.stop_loss_pct,
                                 e.taker_fee_rate, e.slippage_rate, e.maintenance_margin_rate,
                                 best.probability, best.expected_return_on_equity)
        unsafe = position.sl_price <= position.liquidation_price if position.side == "LONG" else position.sl_price >= position.liquidation_price
        if unsafe:
            self.notifier.send(f"SIGNAL REJECTED {best.symbol}: configured stop lies beyond estimated liquidation at {e.leverage}x")
            return
        self.storage.set("position", position.to_dict())
        self.storage.set("last_position_bar", best.signal_time.isoformat())
        self.notifier.send(f"OPEN {position.side} {position.symbol} (PAPER ONLY)\nCalibrated probability: {position.probability:.1%}\nRaw p(up/down): {best.raw_p_up:.1%}/{best.raw_p_down:.1%}\nCalibration samples: {best.calibration_samples}\nValidation precision: {best.validation_precision:.1%} ({best.validation_signals} signals)\nEntry: {position.entry_price:.8g}\nTP: {position.tp_price:.8g}\nSL: {position.sl_price:.8g}\nEst. isolated liquidation: {position.liquidation_price:.8g}\nNotional: ${position.notional:.2f}\nIsolated margin: ${position.margin:.2f}\nExpected return on equity: {position.expected_return_on_equity:.2%}")

    def run(self) -> None:
        self.notifier.send("Probability paper bot started. No real-order endpoint exists in this program.")
        while True:
            try:
                self.cycle()
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                self.notifier.send(f"Cycle error: {type(exc).__name__}: {exc}")
            time.sleep(self.cfg.execution.scan_seconds)
