import pandas as pd
import pytest
from types import SimpleNamespace
from paperbot.execution import evaluate_bar, open_position
from paperbot.model import directional_barrier_labels, is_peg_like, triple_barrier_labels
from paperbot.storage import Storage
from paperbot.config import V3Config
from paperbot.strategy_v3 import build_setup_frame, label_setups


def test_triple_barrier_first_touch():
    df = pd.DataFrame({"close": [100, 100, 100, 100], "high": [100, 102, 100, 100], "low": [100, 99.8, 98, 100]})
    labels = triple_barrier_labels(df, horizon=2, tp=0.01, sl=0.01)
    assert labels.iloc[0] == 1
    assert labels.iloc[1] == -1


def test_asymmetric_directional_targets():
    df = pd.DataFrame({"close": [100, 100, 100, 100], "high": [100, 100.3, 100.3, 100], "low": [100, 99.3, 98.9, 100]})
    up, down = directional_barrier_labels(df, horizon=2, tp=0.01, sl=0.004)
    assert up.iloc[0] == 0
    assert down.iloc[0] == 1


def test_long_take_profit_includes_fees():
    p = open_position("BTCUSDT", "LONG", 100, 450, 10, 1, 0.01, 0.005, 0.0005, 0, 0.004, 0.7, 0.1)
    result = evaluate_bar(p, 101.1, 100, 101, 0.0005, 0, 0)
    assert result is not None
    reason, _, pnl = result
    assert reason == "TAKE_PROFIT"
    assert p.notional == pytest.approx(450)
    assert p.margin == pytest.approx(45)
    assert pnl == pytest.approx(4.04775)


def test_liquidation_wins_over_stop():
    p = open_position("BTCUSDT", "LONG", 100, 450, 100, 1, 0.01, 0.008, 0.0005, 0, 0.004, 0.7, 0.1)
    result = evaluate_bar(p, 100, 99.3, 99.5, 0.0005, 0, 0.002)
    assert result[0] == "LIQUIDATION"


def test_quiet_one_dollar_asset_is_peg_like():
    n = 120
    times = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    close = [1 + ((i % 3) - 1) * 0.0001 for i in range(n)]
    df = pd.DataFrame({"open_time": times, "open": close, "high": [x + 0.0002 for x in close],
                       "low": [x - 0.0002 for x in close], "close": close, "volume": 1,
                       "quote_volume": 1, "taker_quote": 0.5})
    assert is_peg_like(df, 0.95, 1.05, 0.003)


def test_prediction_journal_deduplicates_same_candle(tmp_path):
    storage = Storage(str(tmp_path / "test.sqlite3"))
    prediction = SimpleNamespace(
        symbol="BTCUSDT", signal_time=pd.Timestamp("2026-01-01", tz="UTC"), side="LONG",
        raw_p_up=.8, raw_p_down=.1, p_up=.7, p_down=.2, probability=.7, price=100,
        expected_return_on_equity=.004, calibration_samples=300,
        validation_precision=.6, validation_signals=120,
    )
    storage.save_prediction(prediction, "v2", .01, .004, 12)
    storage.save_prediction(prediction, "v2", .01, .004, 12)
    assert len(storage.unresolved_predictions()) == 1
    storage.resolve_prediction(storage.unresolved_predictions()[0]["id"], 1, 123)
    assert storage.unresolved_predictions() == []


def test_v3_breakout_uses_prior_bars_only():
    n = 260
    times = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    close = pd.Series([100 + i * .02 for i in range(n)], dtype=float)
    close.iloc[-1] += 3
    volume = pd.Series([100 + (i % 7) for i in range(n)], dtype=float)
    volume.iloc[-1] = 1000
    df = pd.DataFrame({"open_time": times, "close_time": times + pd.Timedelta(minutes=15),
                       "open": close - .01, "high": close + .05, "low": close - .05,
                       "close": close, "volume": volume, "quote_volume": volume * close,
                       "taker_quote": volume * close * .55})
    setup = build_setup_frame(df, None, V3Config(min_atr_pct=.0001))
    assert setup.setup_side.iloc[-1] == 1


def test_v3_dynamic_barrier_label():
    frame = pd.DataFrame({"setup_side": [1, 0, 0], "close": [100, 100, 100],
                          "high": [100, 101.1, 100], "low": [100, 99.9, 100],
                          "dynamic_tp_pct": [.01, .01, .01], "dynamic_sl_pct": [.004, .004, .004]})
    assert label_setups(frame, 2).iloc[0] == 1
