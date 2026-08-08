from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from .strategy_v3 import V3_FEATURES


@dataclass
class GlobalMetaModel:
    base_model: HistGradientBoostingClassifier
    calibrator: LogisticRegression
    test_precision: float
    test_signals: int
    calibration_samples: int
    test_mean_ev: float
    test_ev_lower_bound: float
    test_brier_score: float

    def predict_probability(self, rows: pd.DataFrame) -> np.ndarray:
        raw = self.base_model.predict_proba(rows[V3_FEATURES])[:, 1]
        return self.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]


def train_global_meta_model(dataset: pd.DataFrame, calibration_fraction: float = 0.2,
                            test_fraction: float = 0.2, threshold: float = 0.5,
                            random_state: int = 42, fee_roundtrip: float = 0.0) -> GlobalMetaModel:
    if len(dataset) < 500 or dataset.target.nunique() < 2:
        raise ValueError("At least 500 setup rows with both outcomes are required")
    n = len(dataset)
    train_end = int(n * (1 - calibration_fraction - test_fraction))
    calibration_end = int(n * (1 - test_fraction))
    train, calibration, test = dataset.iloc[:train_end], dataset.iloc[train_end:calibration_end], dataset.iloc[calibration_end:]
    if calibration.target.nunique() < 2 or test.target.nunique() < 2:
        raise ValueError("Calibration and test periods must contain both outcomes")
    base = HistGradientBoostingClassifier(max_depth=5, learning_rate=.04, max_iter=200,
                                          l2_regularization=2.0, random_state=random_state)
    base.fit(train[V3_FEATURES], train.target.astype(int))
    raw_cal = base.predict_proba(calibration[V3_FEATURES])[:, 1]
    calibrator = LogisticRegression(C=1.0, solver="lbfgs", random_state=random_state)
    calibrator.fit(raw_cal.reshape(-1, 1), calibration.target.astype(int))
    raw_test = base.predict_proba(test[V3_FEATURES])[:, 1]
    calibrated = calibrator.predict_proba(raw_test.reshape(-1, 1))[:, 1]
    selected = calibrated >= threshold
    precision = float(test.target.to_numpy()[selected].mean()) if selected.any() else 0.0
    realized = test.gross_return.to_numpy() - fee_roundtrip
    selected_returns = realized[selected]
    mean_ev = float(selected_returns.mean()) if len(selected_returns) else float("-inf")
    if len(selected_returns) > 1:
        lower_bound = float(mean_ev - 1.645 * selected_returns.std(ddof=1) / np.sqrt(len(selected_returns)))
    else:
        lower_bound = float("-inf")
    brier = float(np.mean((calibrated - test.target.to_numpy()) ** 2))
    return GlobalMetaModel(base, calibrator, precision, int(selected.sum()), len(calibration),
                           mean_ev, lower_bound, brier)
