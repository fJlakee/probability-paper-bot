from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import pandas as pd

from .backtest import WalkForwardBacktester
from .config import Config
from .execution import evaluate_bar, open_position
from .meta_model_v3 import train_global_meta_model
from .strategy_v3 import V3_FEATURES, build_global_meta_dataset, build_setup_frame


class V3WalkForwardBacktester:
    """Global meta-label walk-forward simulation with one position at a time."""

    def __init__(self, cfg: Config, report_dir: Path):
        self.cfg = cfg
        self.report_dir = report_dir
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def run(self, frames: dict[str, pd.DataFrame]) -> tuple[Path, Path]:
        cfg, m, v3, e = self.cfg, self.cfg.model, self.cfg.v3, self.cfg.execution
        closed = {s: df.iloc[:-1].copy() for s, df in frames.items() if len(df) > m.min_training_rows}
        if "BTCUSDT" not in closed:
            raise RuntimeError("BTCUSDT history is required for v3 market context")
        setup_frames = {
            symbol: build_setup_frame(df, closed["BTCUSDT"], v3).set_index("close_time")
            for symbol, df in closed.items()
        }
        dataset = build_global_meta_dataset(closed, "BTCUSDT", v3, m.horizon_bars)
        if dataset.empty:
            raise RuntimeError("No v3 setups were generated")

        all_times = sorted(set().union(*(set(df.index) for df in setup_frames.values())))
        first_trade_time = all_times[0] + pd.Timedelta(days=m.backtest_training_days)
        times = [t for t in all_times if t >= first_trade_time]
        if not times:
            raise RuntimeError("History is shorter than backtest_training_days")

        equity, position, model = e.initial_capital, None, None
        trades, equity_curve = [], []
        next_retrain = times[0]
        fee_roundtrip = 2 * (e.taker_fee_rate + e.slippage_rate)
        horizon_delta = pd.Timedelta(minutes=15 * m.horizon_bars)
        diagnostics = {
            "setups_evaluated": 0, "pass_entry_probability": 0,
            "pass_expected_return": 0, "pass_calibration_samples": 0,
            "pass_validation_signals": 0, "pass_validation_precision": 0,
            "inside_liquidation_distance": 0, "fully_eligible": 0,
            "max_calibrated_probability": 0.0,
        }

        for ts in times:
            if ts >= next_retrain:
                start = ts - pd.Timedelta(days=m.backtest_training_days)
                # A label is admitted only when its entire future horizon ended before cutoff.
                training = dataset[(dataset.close_time >= start) & (dataset.close_time < ts - horizon_delta)]
                try:
                    model = train_global_meta_model(training, m.calibration_fraction,
                                                    m.test_fraction,
                                                    v3.validation_signal_threshold,
                                                    m.random_state)
                    print(f"v3 backtest {ts}: trained global model on {len(training)} setups", flush=True)
                except ValueError as exc:
                    model = None
                    print(f"v3 backtest {ts}: no model ({exc})", flush=True)
                next_retrain = ts + pd.Timedelta(days=m.backtest_retrain_days)

            if position:
                frame = setup_frames.get(position.symbol)
                if frame is not None and ts in frame.index:
                    bar = frame.loc[ts]
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
                                       "pnl": pnl, "equity_after": equity,
                                       "strategy": "v3-meta-label"})
                        position = None
                equity_curve.append({"time": ts.isoformat(), "equity": equity})
                continue

            if model is None:
                equity_curve.append({"time": ts.isoformat(), "equity": equity})
                continue
            rows = []
            for symbol, frame in setup_frames.items():
                if ts not in frame.index:
                    continue
                row = frame.loc[ts]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[-1]
                if int(row.setup_side) == 0 or row[V3_FEATURES].isna().any():
                    continue
                item = row.copy()
                item["symbol"] = symbol
                rows.append(item)
            if not rows:
                equity_curve.append({"time": ts.isoformat(), "equity": equity})
                continue
            candidates = pd.DataFrame(rows)
            candidates["probability"] = model.predict_probability(candidates)
            candidates["expected_return"] = (
                candidates.probability * candidates.dynamic_tp_pct
                - (1 - candidates.probability) * candidates.dynamic_sl_pct
                - fee_roundtrip
            ) * e.notional_fraction
            for row in candidates.itertuples():
                diagnostics["setups_evaluated"] += 1
                diagnostics["max_calibrated_probability"] = max(
                    diagnostics["max_calibrated_probability"], float(row.probability))
                liquidation_distance = 1 / e.leverage - e.maintenance_margin_rate
                gates = [
                    row.probability >= v3.min_entry_probability,
                    row.expected_return >= e.min_expected_return_on_equity,
                    model.calibration_samples >= v3.min_calibration_samples,
                    model.test_signals >= v3.min_validation_signals,
                    model.test_precision >= v3.min_validation_precision,
                    row.dynamic_sl_pct < liquidation_distance,
                ]
                for key, passed in zip(("pass_entry_probability", "pass_expected_return",
                                        "pass_calibration_samples", "pass_validation_signals",
                                        "pass_validation_precision", "inside_liquidation_distance"), gates):
                    diagnostics[key] += int(passed)
                if all(gates):
                    diagnostics["fully_eligible"] += 1
            eligible = candidates[
                (candidates.probability >= v3.min_entry_probability)
                & (candidates.expected_return >= e.min_expected_return_on_equity)
                & (candidates.dynamic_sl_pct < 1 / e.leverage - e.maintenance_margin_rate)
            ]
            if (not eligible.empty and model.calibration_samples >= v3.min_calibration_samples
                    and model.test_signals >= v3.min_validation_signals
                    and model.test_precision >= v3.min_validation_precision):
                best = eligible.sort_values(["expected_return", "probability"], ascending=False).iloc[0]
                side = "LONG" if int(best.setup_side) == 1 else "SHORT"
                position = open_position(best.symbol, side, float(best.close), equity, e.leverage,
                                         e.notional_fraction, float(best.dynamic_tp_pct),
                                         float(best.dynamic_sl_pct), e.taker_fee_rate,
                                         e.slippage_rate, e.maintenance_margin_rate,
                                         float(best.probability), float(best.expected_return))
                position.opened_at = ts.isoformat()
            equity_curve.append({"time": ts.isoformat(), "equity": equity})

        stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
        trades_path = self.report_dir / f"v3_backtest_trades_{stamp}.csv"
        summary_path = self.report_dir / f"v3_backtest_summary_{stamp}.json"
        pd.DataFrame(trades).to_csv(trades_path, index=False)
        pd.DataFrame(equity_curve).to_csv(self.report_dir / f"v3_backtest_equity_{stamp}.csv", index=False)
        summary = WalkForwardBacktester(cfg, self.report_dir)._summary(trades, equity, diagnostics)
        summary["strategy"] = "v3-meta-label"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
        return trades_path, summary_path
