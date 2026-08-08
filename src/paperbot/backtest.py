from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import timedelta
import json
from pathlib import Path

import pandas as pd

from .config import Config
from .execution import evaluate_bar, open_position
from .model import FEATURES, build_features, predict_feature_row, prefilter_score, train_bundle


class WalkForwardBacktester:
    def __init__(self, cfg: Config, report_dir: Path):
        self.cfg = cfg
        self.report_dir = report_dir
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def _train_universe(self, frames: dict[str, pd.DataFrame], cutoff: pd.Timestamp):
        m = self.cfg.model
        start = cutoff - pd.Timedelta(days=m.backtest_training_days)
        scored = []
        windows = {}
        for symbol, df in frames.items():
            window = df[(df.close_time >= start) & (df.close_time < cutoff)]
            if len(window) < m.min_training_rows:
                continue
            try:
                scored.append((prefilter_score(window), symbol))
                windows[symbol] = window
            except Exception:
                continue
        selected = [s for _, s in sorted(scored, reverse=True)[:m.backtest_universe_size]]
        bundles = {}
        with ThreadPoolExecutor(max_workers=m.model_workers) as pool:
            futures = {
                pool.submit(train_bundle, symbol, windows[symbol], m.horizon_bars,
                            m.take_profit_pct, m.stop_loss_pct, m.calibration_fraction,
                            m.test_fraction, m.min_training_rows,
                            m.validation_signal_threshold, m.random_state): symbol
                for symbol in selected
            }
            for future in as_completed(futures):
                try:
                    bundle = future.result()
                    if bundle:
                        bundles[bundle.symbol] = bundle
                except Exception as exc:
                    print(f"backtest train skip {futures[future]}: {exc}", flush=True)
        return bundles

    def run(self, frames: dict[str, pd.DataFrame]) -> tuple[Path, Path]:
        m, e = self.cfg.model, self.cfg.execution
        closed_frames = {s: df.iloc[:-1].copy() for s, df in frames.items() if len(df) > 100}
        feature_frames = {s: build_features(df).set_index("close_time") for s, df in closed_frames.items()}
        all_times = sorted(set().union(*(set(df.close_time) for df in closed_frames.values())))
        if not all_times:
            raise RuntimeError("No cached candles available for backtest")
        first_trade_time = all_times[0] + pd.Timedelta(days=m.backtest_training_days)
        times = [t for t in all_times if t >= first_trade_time]
        if not times:
            raise RuntimeError("History is shorter than backtest_training_days")

        equity = e.initial_capital
        position = None
        trades = []
        equity_curve = []
        bundles = {}
        next_retrain = times[0]
        fee_roundtrip = 2 * (e.taker_fee_rate + e.slippage_rate)
        diagnostics = {
            "predictions_evaluated": 0, "pass_entry_probability": 0,
            "pass_expected_return": 0, "pass_calibration_samples": 0,
            "pass_validation_signals": 0, "pass_validation_precision": 0,
            "fully_eligible": 0, "max_calibrated_probability": 0.0,
            "max_raw_probability": 0.0, "max_validation_signals": 0,
            "max_validation_precision": 0.0,
        }

        for ts in times:
            if ts >= next_retrain:
                bundles = self._train_universe(closed_frames, ts)
                next_retrain = ts + pd.Timedelta(days=m.backtest_retrain_days)
                print(f"backtest {ts}: trained {len(bundles)} models", flush=True)

            if position:
                indexed = feature_frames.get(position.symbol)
                if indexed is not None and ts in indexed.index:
                    bar = indexed.loc[ts]
                    if isinstance(bar, pd.DataFrame):
                        bar = bar.iloc[-1]
                    position.bars_open += 1
                    result = evaluate_bar(position, float(bar.high), float(bar.low), float(bar.close),
                                          e.taker_fee_rate, e.slippage_rate,
                                          e.liquidation_penalty_rate,
                                          position.bars_open >= e.max_position_bars)
                    if result:
                        reason, exit_price, pnl = result
                        equity = max(0.0, equity + pnl)
                        trades.append({**asdict(position), "closed_at": ts.isoformat(),
                                       "exit_price": exit_price, "reason": reason,
                                       "pnl": pnl, "equity_after": equity})
                        position = None
                equity_curve.append({"time": ts.isoformat(), "equity": equity})
                continue

            candidates = []
            for symbol, bundle in bundles.items():
                indexed = feature_frames[symbol]
                if ts not in indexed.index:
                    continue
                row = indexed.loc[ts]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[-1]
                if row[FEATURES].isna().any():
                    continue
                pred = predict_feature_row(bundle, row, m.take_profit_pct, m.stop_loss_pct,
                                           fee_roundtrip, e.notional_fraction)
                diagnostics["predictions_evaluated"] += 1
                diagnostics["max_calibrated_probability"] = max(diagnostics["max_calibrated_probability"], pred.probability)
                diagnostics["max_raw_probability"] = max(diagnostics["max_raw_probability"], pred.raw_p_up, pred.raw_p_down)
                diagnostics["max_validation_signals"] = max(diagnostics["max_validation_signals"], pred.validation_signals)
                diagnostics["max_validation_precision"] = max(diagnostics["max_validation_precision"], pred.validation_precision)
                gates = [
                    pred.probability >= m.min_entry_probability,
                    pred.expected_return_on_equity >= e.min_expected_return_on_equity,
                    pred.calibration_samples >= m.min_calibration_samples,
                    pred.validation_signals >= m.min_validation_trades,
                    pred.validation_precision >= m.min_validation_precision,
                ]
                for key, passed in zip(("pass_entry_probability", "pass_expected_return",
                                        "pass_calibration_samples", "pass_validation_signals",
                                        "pass_validation_precision"), gates):
                    diagnostics[key] += int(passed)
                if all(gates):
                    diagnostics["fully_eligible"] += 1
                    candidates.append(pred)
            if candidates:
                best = max(candidates, key=lambda x: (x.expected_return_on_equity, x.probability))
                position = open_position(best.symbol, best.side, best.price, equity, e.leverage,
                                         e.notional_fraction, m.take_profit_pct, m.stop_loss_pct,
                                         e.taker_fee_rate, e.slippage_rate,
                                         e.maintenance_margin_rate, best.probability,
                                         best.expected_return_on_equity)
                position.opened_at = ts.isoformat()
            equity_curve.append({"time": ts.isoformat(), "equity": equity})

        stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
        trades_path = self.report_dir / f"backtest_trades_{stamp}.csv"
        summary_path = self.report_dir / f"backtest_summary_{stamp}.json"
        pd.DataFrame(trades).to_csv(trades_path, index=False)
        summary = self._summary(trades, equity, diagnostics)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        pd.DataFrame(equity_curve).to_csv(self.report_dir / f"backtest_equity_{stamp}.csv", index=False)
        print(json.dumps(summary, indent=2), flush=True)
        return trades_path, summary_path

    def _summary(self, trades: list[dict], equity: float, diagnostics: dict) -> dict:
        if not trades:
            return {"trades": 0, "initial_equity": self.cfg.execution.initial_capital,
                    "final_equity": equity, "net_pnl": equity - self.cfg.execution.initial_capital,
                    "signal_diagnostics": diagnostics}
        pnl = pd.Series([x["pnl"] for x in trades], dtype=float)
        wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
        equities = pd.Series([self.cfg.execution.initial_capital] + [x["equity_after"] for x in trades])
        drawdown = equities / equities.cummax() - 1
        loss_streak = longest = 0
        for value in pnl:
            loss_streak = loss_streak + 1 if value <= 0 else 0
            longest = max(longest, loss_streak)
        frame = pd.DataFrame(trades)
        by_side = frame.groupby("side")["pnl"].agg(["count", "sum", "mean"]).round(6).to_dict("index")
        by_symbol = frame.groupby("symbol")["pnl"].agg(["count", "sum", "mean"]).round(6).to_dict("index")
        return {
            "trades": len(trades), "initial_equity": self.cfg.execution.initial_capital,
            "final_equity": equity, "net_pnl": float(pnl.sum()),
            "win_rate": float((pnl > 0).mean()), "average_pnl": float(pnl.mean()),
            "average_win": float(wins.mean()) if len(wins) else None,
            "average_loss": float(losses.mean()) if len(losses) else None,
            "profit_factor": float(wins.sum() / abs(losses.sum())) if losses.sum() else None,
            "max_drawdown": float(drawdown.min()), "longest_losing_streak": longest,
            "by_side": by_side, "by_symbol": by_symbol, "fees_are_included": True,
            "signal_diagnostics": diagnostics,
        }
