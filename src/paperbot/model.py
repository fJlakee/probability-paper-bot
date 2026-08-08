from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


FEATURES = [
    "ret_1", "ret_3", "ret_6", "ret_12", "ema_gap", "rsi", "atr_pct",
    "volume_z", "range_pct", "body_pct", "taker_ratio", "hour_sin", "hour_cos",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]
    for n in (1, 3, 6, 12):
        out[f"ret_{n}"] = close.pct_change(n)
    ema_fast = close.ewm(span=20, adjust=False).mean()
    ema_slow = close.ewm(span=100, adjust=False).mean()
    out["ema_gap"] = ema_fast / ema_slow - 1
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    out["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    prev_close = close.shift(1)
    tr = pd.concat([(out.high - out.low), (out.high - prev_close).abs(), (out.low - prev_close).abs()], axis=1).max(axis=1)
    out["atr_pct"] = tr.ewm(alpha=1 / 14, adjust=False).mean() / close
    log_volume = np.log1p(out["quote_volume"])
    out["volume_z"] = (log_volume - log_volume.rolling(50).mean()) / log_volume.rolling(50).std()
    out["range_pct"] = (out.high - out.low) / close
    out["body_pct"] = (out.close - out.open) / out.open
    out["taker_ratio"] = out.taker_quote / out.quote_volume.replace(0, np.nan)
    hour = out.open_time.dt.hour + out.open_time.dt.minute / 60
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return out.replace([np.inf, -np.inf], np.nan)


def triple_barrier_labels(df: pd.DataFrame, horizon: int, tp: float, sl: float) -> pd.Series:
    labels = np.zeros(len(df), dtype=int)
    close, highs, lows = df.close.to_numpy(), df.high.to_numpy(), df.low.to_numpy()
    for i in range(len(df) - horizon):
        upper, lower = close[i] * (1 + tp), close[i] * (1 - sl)
        for j in range(i + 1, i + horizon + 1):
            hit_up, hit_down = highs[j] >= upper, lows[j] <= lower
            if hit_up and hit_down:
                labels[i] = -1  # conservative ambiguity rule
                break
            if hit_up:
                labels[i] = 1
                break
            if hit_down:
                labels[i] = -1
                break
    result = pd.Series(labels, index=df.index, dtype=int)
    result.iloc[-horizon:] = np.nan
    return result


def directional_barrier_labels(df: pd.DataFrame, horizon: int, tp: float, sl: float) -> tuple[pd.Series, pd.Series]:
    """Binary success labels for asymmetric long and short trades.

    Long succeeds when +TP is touched before -SL. Short succeeds when -TP is
    touched before +SL. A same-candle collision is conservatively a failure.
    """
    up = np.zeros(len(df), dtype=float)
    down = np.zeros(len(df), dtype=float)
    close, highs, lows = df.close.to_numpy(), df.high.to_numpy(), df.low.to_numpy()
    for i in range(len(df) - horizon):
        long_tp, long_sl = close[i] * (1 + tp), close[i] * (1 - sl)
        short_tp, short_sl = close[i] * (1 - tp), close[i] * (1 + sl)
        long_done = short_done = False
        for j in range(i + 1, i + horizon + 1):
            if not long_done:
                hit_tp, hit_sl = highs[j] >= long_tp, lows[j] <= long_sl
                if hit_tp or hit_sl:
                    up[i] = 1.0 if hit_tp and not hit_sl else 0.0
                    long_done = True
            if not short_done:
                hit_tp, hit_sl = lows[j] <= short_tp, highs[j] >= short_sl
                if hit_tp or hit_sl:
                    down[i] = 1.0 if hit_tp and not hit_sl else 0.0
                    short_done = True
            if long_done and short_done:
                break
    up_s, down_s = pd.Series(up, index=df.index), pd.Series(down, index=df.index)
    up_s.iloc[-horizon:] = np.nan
    down_s.iloc[-horizon:] = np.nan
    return up_s, down_s


@dataclass
class Prediction:
    symbol: str
    side: str
    probability: float
    p_up: float
    p_down: float
    raw_p_up: float
    raw_p_down: float
    expected_return_on_equity: float
    validation_precision: float
    validation_signals: int
    price: float
    signal_time: pd.Timestamp
    calibration_samples: int


@dataclass
class ModelBundle:
    symbol: str
    model_up: HistGradientBoostingClassifier
    model_down: HistGradientBoostingClassifier
    calibrator_up: LogisticRegression
    calibrator_down: LogisticRegression
    validation_precision: float
    validation_signals: int
    trained_at: str
    calibration_samples: int


def prefilter_score(raw: pd.DataFrame) -> float:
    """Cheap regime/activity score used on every symbol before ML inference."""
    data = build_features(raw).dropna(subset=FEATURES)
    if data.empty:
        return float("-inf")
    x = data.iloc[-1]
    return float(abs(x.ret_12) + 2 * x.atr_pct + 0.002 * max(x.volume_z, 0))


def is_peg_like(raw: pd.DataFrame, price_min: float, price_max: float, max_atr_pct: float) -> bool:
    """Detect quiet instruments trading near a one-dollar peg."""
    data = build_features(raw).dropna(subset=["atr_pct"])
    if data.empty:
        return False
    recent = data.tail(96)
    median_price = float(recent.close.median())
    median_atr = float(recent.atr_pct.median())
    return price_min <= median_price <= price_max and median_atr <= max_atr_pct


def train_bundle(symbol: str, raw: pd.DataFrame, horizon: int, tp: float, sl: float,
                 calibration_fraction: float, test_fraction: float, min_rows: int,
                 validation_signal_threshold: float,
                 random_state: int) -> ModelBundle | None:
    data = build_features(raw)
    data["target_up"], data["target_down"] = directional_barrier_labels(raw, horizon, tp, sl)
    trainable = data.dropna(subset=FEATURES + ["target_up", "target_down"])
    if len(trainable) < min_rows or trainable.target_up.nunique() < 2 or trainable.target_down.nunique() < 2:
        return None
    n = len(trainable)
    train_end = int(n * (1 - calibration_fraction - test_fraction))
    calibration_end = int(n * (1 - test_fraction))
    train = trainable.iloc[:train_end]
    calibration = trainable.iloc[train_end:calibration_end]
    valid = trainable.iloc[calibration_end:]
    if min(len(train), len(calibration), len(valid)) < 20:
        return None
    for part in (train, calibration):
        if part.target_up.nunique() < 2 or part.target_down.nunique() < 2:
            return None
    params = dict(max_depth=4, learning_rate=0.05, max_iter=150, l2_regularization=1.0, random_state=random_state)
    model_up = HistGradientBoostingClassifier(**params)
    model_down = HistGradientBoostingClassifier(**params)
    model_up.fit(train[FEATURES], train.target_up.astype(int))
    model_down.fit(train[FEATURES], train.target_down.astype(int))
    raw_up_c = model_up.predict_proba(calibration[FEATURES])[:, list(model_up.classes_).index(1)]
    raw_down_c = model_down.predict_proba(calibration[FEATURES])[:, list(model_down.classes_).index(1)]
    calibrator_up = LogisticRegression(C=1.0, solver="lbfgs", random_state=random_state)
    calibrator_down = LogisticRegression(C=1.0, solver="lbfgs", random_state=random_state)
    calibrator_up.fit(raw_up_c.reshape(-1, 1), calibration.target_up.astype(int))
    calibrator_down.fit(raw_down_c.reshape(-1, 1), calibration.target_down.astype(int))
    raw_up_v = model_up.predict_proba(valid[FEATURES])[:, list(model_up.classes_).index(1)]
    raw_down_v = model_down.predict_proba(valid[FEATURES])[:, list(model_down.classes_).index(1)]
    p_up_v = calibrator_up.predict_proba(raw_up_v.reshape(-1, 1))[:, 1]
    p_down_v = calibrator_down.predict_proba(raw_down_v.reshape(-1, 1))[:, 1]
    selected = np.maximum(p_up_v, p_down_v) >= validation_signal_threshold
    correct = np.where(p_up_v >= p_down_v, valid.target_up.to_numpy(), valid.target_down.to_numpy())
    precision = float(correct[selected].mean()) if selected.any() else 0.0
    return ModelBundle(symbol, model_up, model_down, calibrator_up, calibrator_down,
                       precision, int(selected.sum()), datetime.now(timezone.utc).isoformat(), len(calibration))


def predict_bundle(bundle: ModelBundle, raw: pd.DataFrame, tp: float, sl: float,
                   fee_roundtrip: float, notional_fraction: float) -> Prediction | None:
    data = build_features(raw).dropna(subset=FEATURES)
    if data.empty:
        return None
    latest = data.iloc[-1]
    latest_x = latest[FEATURES].to_frame().T
    return predict_feature_row(bundle, latest, tp, sl, fee_roundtrip, notional_fraction)


def predict_feature_row(bundle: ModelBundle, latest: pd.Series, tp: float, sl: float,
                        fee_roundtrip: float, notional_fraction: float) -> Prediction:
    latest_x = latest[FEATURES].to_frame().T
    raw_p_up = float(bundle.model_up.predict_proba(latest_x)[0, list(bundle.model_up.classes_).index(1)])
    raw_p_down = float(bundle.model_down.predict_proba(latest_x)[0, list(bundle.model_down.classes_).index(1)])
    p_up = float(bundle.calibrator_up.predict_proba([[raw_p_up]])[0, 1])
    p_down = float(bundle.calibrator_down.predict_proba([[raw_p_down]])[0, 1])
    side = "LONG" if p_up >= p_down else "SHORT"
    probability = max(p_up, p_down)
    expected_underlying = probability * tp - (1 - probability) * sl - fee_roundtrip
    signal_time = latest.get("close_time", latest.name)
    return Prediction(bundle.symbol, side, probability, p_up, p_down, raw_p_up, raw_p_down,
                      expected_underlying * notional_fraction, bundle.validation_precision,
                      bundle.validation_signals, float(latest.close), pd.Timestamp(signal_time),
                      bundle.calibration_samples)


def save_bundle(bundle: ModelBundle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path: Path) -> ModelBundle | None:
    try:
        return joblib.load(path) if path.exists() else None
    except Exception:
        return None
