from __future__ import annotations

import numpy as np
import pandas as pd

from .config import V3Config
from .model import FEATURES, build_features


V3_FEATURES = FEATURES + [
    "ema50_200_gap", "ema50_slope", "breakout_strength", "btc_ret_12",
    "relative_ret_12", "setup_side", "dynamic_tp_pct", "dynamic_sl_pct",
]


def build_setup_frame(raw: pd.DataFrame, btc_raw: pd.DataFrame | None, cfg: V3Config) -> pd.DataFrame:
    """Build causal trend/breakout setups; rolling levels are shifted by one bar."""
    out = build_features(raw)
    fast = out.close.ewm(span=cfg.ema_fast, adjust=False).mean()
    slow = out.close.ewm(span=cfg.ema_slow, adjust=False).mean()
    prior_high = out.high.rolling(cfg.breakout_bars).max().shift(1)
    prior_low = out.low.rolling(cfg.breakout_bars).min().shift(1)
    out["ema50_200_gap"] = fast / slow - 1
    out["ema50_slope"] = fast.pct_change(3)
    up_strength = out.close / prior_high - 1
    down_strength = prior_low / out.close - 1

    if btc_raw is not None and not btc_raw.empty:
        btc = build_features(btc_raw)[["close_time", "ret_12"]].rename(columns={"ret_12": "btc_ret_12"})
        out = out.merge(btc, on="close_time", how="left")
    else:
        out["btc_ret_12"] = 0.0
    out["btc_ret_12"] = out.btc_ret_12.fillna(0.0)
    out["relative_ret_12"] = out.ret_12 - out.btc_ret_12

    volatility_ok = out.atr_pct.between(cfg.min_atr_pct, cfg.max_atr_pct)
    volume_ok = out.volume_z >= cfg.min_volume_z
    long_setup = ((out.close > prior_high) & (fast > slow) & (out.ema50_slope > 0)
                  & volume_ok & volatility_ok
                  & (out.btc_ret_12 >= -cfg.btc_adverse_return_limit))
    short_setup = ((out.close < prior_low) & (fast < slow) & (out.ema50_slope < 0)
                   & volume_ok & volatility_ok
                   & (out.btc_ret_12 <= cfg.btc_adverse_return_limit))
    out["setup_side"] = np.select([long_setup, short_setup], [1, -1], default=0)
    out["breakout_strength"] = np.where(long_setup, up_strength, np.where(short_setup, down_strength, 0.0))
    out["dynamic_tp_pct"] = (out.atr_pct * cfg.tp_atr_multiple).clip(cfg.min_tp_pct, cfg.max_tp_pct)
    out["dynamic_sl_pct"] = (out.atr_pct * cfg.sl_atr_multiple).clip(cfg.min_sl_pct, cfg.max_sl_pct)
    return out.replace([np.inf, -np.inf], np.nan)


def label_setups(frame: pd.DataFrame, horizon: int) -> pd.Series:
    """Meta-label: 1 when a setup's dynamic TP is touched before its dynamic SL."""
    return label_setup_outcomes(frame, horizon)["target"]


def label_setup_outcomes(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    labels = pd.Series(np.nan, index=frame.index, dtype=float)
    returns = pd.Series(np.nan, index=frame.index, dtype=float)
    highs, lows = frame.high.to_numpy(), frame.low.to_numpy()
    for i in np.flatnonzero(frame.setup_side.to_numpy() != 0):
        if i + horizon >= len(frame):
            continue
        side = int(frame.setup_side.iloc[i])
        entry = float(frame.close.iloc[i])
        tp, sl = float(frame.dynamic_tp_pct.iloc[i]), float(frame.dynamic_sl_pct.iloc[i])
        labels.iloc[i] = 0.0
        returns.iloc[i] = side * (float(frame.close.iloc[i + horizon]) / entry - 1)
        for j in range(i + 1, i + horizon + 1):
            if side == 1:
                hit_tp, hit_sl = highs[j] >= entry * (1 + tp), lows[j] <= entry * (1 - sl)
            else:
                hit_tp, hit_sl = lows[j] <= entry * (1 - tp), highs[j] >= entry * (1 + sl)
            if hit_tp or hit_sl:
                labels.iloc[i] = float(hit_tp and not hit_sl)
                returns.iloc[i] = tp if hit_tp and not hit_sl else -sl
                break
    return pd.DataFrame({"target": labels, "gross_return": returns})


def build_global_meta_dataset(frames: dict[str, pd.DataFrame], btc_symbol: str,
                              cfg: V3Config, horizon: int) -> pd.DataFrame:
    btc = frames.get(btc_symbol)
    parts = []
    for symbol, raw in frames.items():
        setup = build_setup_frame(raw, btc, cfg)
        outcomes = label_setup_outcomes(setup, horizon)
        setup["target"] = outcomes.target
        setup["gross_return"] = outcomes.gross_return
        setup["symbol"] = symbol
        selected = setup[setup.setup_side != 0].dropna(subset=V3_FEATURES + ["target"])
        if not selected.empty:
            parts.append(selected)
    return pd.concat(parts, ignore_index=True).sort_values("close_time") if parts else pd.DataFrame()
