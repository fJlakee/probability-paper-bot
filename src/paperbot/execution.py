from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone


@dataclass
class Position:
    symbol: str
    side: str
    entry_price: float
    quantity: float
    notional: float
    margin: float
    leverage: int
    tp_price: float
    sl_price: float
    liquidation_price: float
    probability: float
    expected_return_on_equity: float
    opened_at: str
    bars_open: int = 0
    entry_fee: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def open_position(symbol: str, side: str, price: float, equity: float, leverage: int,
                  notional_fraction: float, tp: float, sl: float, fee: float,
                  slippage: float, maintenance_margin_rate: float,
                  probability: float, expected_return: float) -> Position:
    entry = price * (1 + slippage if side == "LONG" else 1 - slippage)
    notional = equity * notional_fraction
    margin = notional / leverage
    quantity = notional / entry
    if side == "LONG":
        tp_price, sl_price = entry * (1 + tp), entry * (1 - sl)
        liquidation = entry * (1 - (1 / leverage - maintenance_margin_rate))
    else:
        tp_price, sl_price = entry * (1 - tp), entry * (1 + sl)
        liquidation = entry * (1 + (1 / leverage - maintenance_margin_rate))
    return Position(symbol, side, entry, quantity, notional, margin, leverage, tp_price,
                    sl_price, liquidation, probability, expected_return,
                    datetime.now(timezone.utc).isoformat(), entry_fee=notional * fee)


def evaluate_bar(position: Position, high: float, low: float, close: float,
                 fee: float, slippage: float, liquidation_penalty: float,
                 force_close: bool = False) -> tuple[str, float, float] | None:
    long = position.side == "LONG"
    hit_liq = low <= position.liquidation_price if long else high >= position.liquidation_price
    hit_sl = low <= position.sl_price if long else high >= position.sl_price
    hit_tp = high >= position.tp_price if long else low <= position.tp_price
    if hit_liq:
        raw_exit, reason = position.liquidation_price, "LIQUIDATION"
    elif hit_sl and hit_tp:
        raw_exit, reason = position.sl_price, "SL_AMBIGUOUS"
    elif hit_sl:
        raw_exit, reason = position.sl_price, "STOP_LOSS"
    elif hit_tp:
        raw_exit, reason = position.tp_price, "TAKE_PROFIT"
    elif force_close:
        raw_exit, reason = close, "TIME_EXIT"
    else:
        return None
    exit_price = raw_exit * (1 - slippage if long else 1 + slippage)
    gross = (exit_price - position.entry_price) * position.quantity * (1 if long else -1)
    exit_fee = abs(exit_price * position.quantity) * fee
    penalty = position.notional * liquidation_penalty if reason == "LIQUIDATION" else 0.0
    pnl = gross - position.entry_fee - exit_fee - penalty
    return reason, exit_price, pnl
