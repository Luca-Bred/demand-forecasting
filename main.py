from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Literal, Dict, Any, Tuple
import warnings
import math

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.tsa.holtwinters import (
    SimpleExpSmoothing,
    Holt,
    ExponentialSmoothing,
)
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

app = FastAPI(title="Demand Forecast API", version="6.0.0")


# =========================================================
# Request schema
# =========================================================

AllowedModel = Literal[
    "auto",
    "naive",
    "seasonal_naive",
    "moving_average",
    "ets",
    "holt",
    "holt_winters",
    "arima",
    "sarima",
    "croston",
    "sba",
    "tsb",
    "gradient_boosting",
]


class ForecastRequest(BaseModel):
    sku: str
    demand: List[float]
    stockout: List[int]
    periods: int = 15
    model: Optional[AllowedModel] = "auto"


# =========================================================
# Validation & preprocessing
# =========================================================

def validate_inputs(demand: List[float], stockout: List[int], periods: int) -> None:
    if len(demand) != len(stockout):
        raise ValueError("demand und stockout müssen gleich lang sein.")
    if len(demand) == 0:
        raise ValueError("demand darf nicht leer sein.")
    if any(s not in (0, 1) for s in stockout):
        raise ValueError("stockout darf nur 0 oder 1 enthalten.")
    if periods < 1:
        raise ValueError("periods muss >= 1 sein.")


def clean_demand(demand: List[float], stockout: List[int]) -> List[float]:
    """
    stockout = 1 wird als fehlende Beobachtung behandelt und entfernt.
    """
    return [float(d) for d, s in zip(demand, stockout) if s == 0]


def count_nonzero(values: List[float]) -> int:
    return sum(1 for x in values if x > 0)


def compute_adi_cv2(series: List[float]) -> Tuple[float, float]:
    """
    ADI = Average Demand Interval
    CV² = (std / mean)^2 auf positiven Nachfragen
    """
    if not series:
        return float("inf"), float("inf")

    non_zero = [x for x in series if x > 0]
    if len(non_zero) == 0:
        return float("inf"), float("inf")

    adi = len(series) / len(non_zero)
    mean_nz = np.mean(non_zero)
    std_nz = np.std(non_zero, ddof=0)
    cv2 = (std_nz / mean_nz) ** 2 if mean_nz != 0 else float("inf")
    return float(adi), float(cv2)


def detect_demand_pattern(series: List[float]) -> str:
    if not series:
        return "no_demand"
    if all(x == 0 for x in series):
        return "all_zero"

    adi, cv2 = compute_adi_cv2(series)

    if adi > 1.32 and cv2 <= 0.49:
        return "intermittent"
    if adi > 1.32 and cv2 > 0.49:
        return "lumpy"
    if adi <= 1.32 and cv2 > 0.49:
        return "erratic"
    return "smooth"


def has_seasonality_candidate(series: List[float], seasonal_periods: int = 12) -> bool:
    """
    Einfache Heuristik:
    Saisonale Modelle erst ab 2 vollen Saisons testen.
    """
    return len(series) >= seasonal_periods * 2


# =========================================================
# Metrics
# =========================================================

def mae(actual: List[float], predicted: List[float]) -> float:
    if not actual:
        return 0.0
    return float(np.mean([abs(a - p) for a, p in zip(actual, predicted)]))


def wape(actual: List[float], predicted: List[float]) -> float:
    denom = float(np.sum(np.abs(actual)))
    if denom == 0:
        return 0.0
    return float(np.sum([abs(a - p) for a, p in zip(actual, predicted)]) / denom)


def bias(actual: List[float], predicted: List[float]) -> float:
    if not actual:
        return 0.0
    return float(np.mean([p - a for a, p in zip(actual, predicted)]))


def mase(actual: List[float], predicted: List[float], train: List[float], m: int = 1) -> float:
    if len(train) <= m or not actual:
        return 0.0

    naive_errors = [abs(train[i] - train[i - m]) for i in range(m, len(train))]
    if len(naive_errors) == 0:
        return 0.0

    scale = float(np.mean(naive_errors))
    if scale == 0:
        return 0.0

    return float(mae(actual, predicted) / scale)


# =========================================================
# Forecast models
# =========================================================

def naive_forecast(train: List[float], periods: int) -> List[float]:
    last_value = float(train[-1])
    return [last_value] * periods


def seasonal_naive_forecast(train: List[float], periods: int, season_length: int = 12) -> List[float]:
    if len(train) < season_length:
        raise ValueError("Zu wenig Historie für seasonal_naive.")
    history = list(map(float, train))
    forecasts = []
    for i in range(periods):
        forecasts.append(history[-season_length + (i % season_length)])
    return forecasts


def moving_average_forecast(train: List[float], periods: int, window: int = 3) -> List[float]:
    """
    Iterativer Moving Average Forecast.
    """
    history = list(map(float, train))
    forecasts = []

    for _ in range(periods):
        effective_window = min(window, len(history))
        value = float(np.mean(history[-effective_window:]))
        forecasts.append(value)
        history.append(value)

    return forecasts


def ets_forecast(train: List[float], periods: int) -> List[float]:
    """
    Simple Exponential Smoothing
    """
    model = SimpleExpSmoothing(np.asarray(train, dtype=float), initialization_method="estimated")
    fit = model.fit(optimized=True)
    fc = fit.forecast(periods)
    return [float(x) for x in fc]


def holt_forecast(train: List[float], periods: int) -> List[float]:
    model = Holt(np.asarray(train, dtype=float), initialization_method="estimated")
    fit = model.fit(optimized=True)
    fc = fit.forecast(periods)
    return [float(x) for x in fc]


def holt_winters_forecast(train: List[float], periods: int, season_length: int = 12) -> List[float]:
    if len(train) < season_length * 2:
        raise ValueError("Zu wenig Historie für Holt-Winters.")
    model = ExponentialSmoothing(
        np.asarray(train, dtype=float),
        trend="add",
        seasonal="add",
        seasonal_periods=season_length,
        initialization_method="estimated",
    )
    fit = model.fit(optimized=True)
    fc = fit.forecast(periods)
    return [float(x) for x in fc]


def arima_forecast(train: List[float], periods: int, order=(1, 1, 1)) -> List[float]:
    model = SARIMAX(
        np.asarray(train, dtype=float),
        order=order,
        seasonal_order=(0, 0, 0, 0),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False)
    fc = fit.forecast(periods)
    return [float(x) for x in fc]


def sarima_forecast(train: List[float], periods: int, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)) -> List[float]:
    if len(train) < 24:
        raise ValueError("Zu wenig Historie für SARIMA.")
    model = SARIMAX(
        np.asarray(train, dtype=float),
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False)
    fc = fit.forecast(periods)
    return [float(x) for x in fc]


def croston_forecast(train: List[float], periods: int, alpha: float = 0.1) -> List[float]:
    """
    Croston für intermittierende Nachfrage.
    """
    ts = np.asarray(train, dtype=float)
    demand = []
    intervals = []

    interval = 1
    for x in ts:
        if x > 0:
            demand.append(x)
            intervals.append(interval)
            interval = 1
        else:
            interval += 1

    if len(demand) == 0:
        return [0.0] * periods

    z = demand[0]
    p = float(intervals[0])

    for i in range(1, len(demand)):
        z = alpha * demand[i] + (1 - alpha) * z
        p = alpha * intervals[i] + (1 - alpha) * p

    f = float(z / p) if p != 0 else 0.0
    return [f] * periods


def sba_forecast(train: List[float], periods: int, alpha: float = 0.1) -> List[float]:
    """
    Syntetos-Boylan Approximation:
    SBA = Croston * (1 - alpha/2)
    """
    croston_fc = croston_forecast(train, periods, alpha=alpha)
    correction = 1 - alpha / 2
    return [float(x * correction) for x in croston_fc]


def tsb_forecast(train: List[float], periods: int, alpha: float = 0.2, beta: float = 0.2) -> List[float]:
    """
    TSB (Teunter-Syntetos-Babai):
    glättet Nachfragehöhe und Auftretenswahrscheinlichkeit.
    """
    ts = np.asarray(train, dtype=float)

    if len(ts) == 0:
        return [0.0] * periods

    demand_occurrence = np.where(ts > 0, 1.0, 0.0)

    first_nonzero_idx = np.argmax(ts > 0) if np.any(ts > 0) else None
    if first_nonzero_idx is None or np.all(ts == 0):
        return [0.0] * periods

    z = float(ts[first_nonzero_idx])  # demand size estimate
    p = float(demand_occurrence[first_nonzero_idx])  # occurrence probability

    for x in ts[first_nonzero_idx + 1:]:
        occ = 1.0 if x > 0 else 0.0
        p = beta * occ + (1 - beta) * p
        if x > 0:
            z = alpha * x + (1 - alpha) * z

    f = z * p
    return [float(f)] * periods


def create_ml_features(series: List[float], season_length: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lag-basierte Features für Gradient Boosting.
    Features:
    - lag1, lag2, lag3, lag6, lag12
    - rolling mean 3
    - rolling mean 6
    - month index (0..11)
    """
    values = list(map(float, series))
    rows = []
    targets = []

    min_required = 12
    if len(values) <= min_required:
        raise ValueError("Zu wenig Historie für Gradient Boosting.")

    for t in range(12, len(values)):
        lag1 = values[t - 1]
        lag2 = values[t - 2]
        lag3 = values[t - 3]
        lag6 = values[t - 6]
        lag12 = values[t - 12]
        rmean3 = float(np.mean(values[t - 3:t]))
        rmean6 = float(np.mean(values[t - 6:t]))
        month_idx = t % season_length

        rows.append([lag1, lag2, lag3, lag6, lag12, rmean3, rmean6, month_idx])
        targets.append(values[t])

    return np.asarray(rows), np.asarray(targets)


def gradient_boosting_forecast(train: List[float], periods: int, season_length: int = 12) -> List[float]:
    if len(train) < 24:
        raise ValueError("Zu wenig Historie für Gradient Boosting.")

    X, y = create_ml_features(train, season_length=season_length)
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X, y)

    history = list(map(float, train))
    forecasts = []

    for step in range(periods):
        t = len(history)
        lag1 = history[t - 1]
        lag2 = history[t - 2]
        lag3 = history[t - 3]
        lag6 = history[t - 6]
        lag12 = history[t - 12]
        rmean3 = float(np.mean(history[t - 3:t]))
        rmean6 = float(np.mean(history[t - 6:t]))
        month_idx = t % season_length

        x_new = np.asarray([[lag1, lag2, lag3, lag6, lag12, rmean3, rmean6, month_idx]])
        pred = float(model.predict(x_new)[0])

        forecasts.append(pred)
        history.append(pred)

    return forecasts


def run_model(model_name: str, train: List[float], periods: int) -> List[float]:
    if model_name == "naive":
        return naive_forecast(train, periods)
    if model_name == "seasonal_naive":
        return seasonal_naive_forecast(train, periods, season_length=12)
    if model_name == "moving_average":
        return moving_average_forecast(train, periods, window=3)
    if model_name == "ets":
        return ets_forecast(train, periods)
    if model_name == "holt":
        return holt_forecast(train, periods)
    if model_name == "holt_winters":
        return holt_winters_forecast(train, periods, season_length=12)
    if model_name == "arima":
        return arima_forecast(train, periods, order=(1, 1, 1))
    if model_name == "sarima":
        return sarima_forecast(train, periods, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    if model_name == "croston":
        return croston_forecast(train, periods, alpha=0.1)
    if model_name == "sba":
        return sba_forecast(train, periods, alpha=0.1)
    if model_name == "tsb":
        return tsb_forecast(train, periods, alpha=0.2, beta=0.2)
    if model_name == "gradient_boosting":
        return gradient_boosting_forecast(train, periods, season_length=12)
    raise ValueError(f"Unbekanntes Modell: {model_name}")


# =========================================================
# Model eligibility
# =========================================================

def model_is_allowed(model_name: str, train: List[float], pattern: str) -> bool:
    n = len(train)
    non_zero = count_nonzero(train)

    if model_name == "naive":
        return n >= 1
    if model_name == "seasonal_naive":
        return n >= 24
    if model_name == "moving_average":
        return n >= 3
    if model_name == "ets":
        return n >= 4
    if model_name == "holt":
        return n >= 6
    if model_name == "holt_winters":
        return n >= 24
    if model_name == "arima":
        return n >= 12
    if model_name == "sarima":
        return n >= 24
    if model_name == "croston":
        return pattern in ("intermittent", "lumpy") and non_zero >= 2
    if model_name == "sba":
        return pattern in ("intermittent", "lumpy") and non_zero >= 2
    if model_name == "tsb":
        return pattern in ("intermittent", "lumpy") and non_zero >= 2
    if model_name == "gradient_boosting":
        return n >= 24
    return False


# =========================================================
# Rolling backtesting
# =========================================================

def generate_rolling_splits(series: List[float], n_splits: int = 3, horizon: int = 1) -> List[Tuple[List[float], List[float]]]:
    splits = []
    n = len(series)
    last_test_start = n - horizon

    if last_test_start <= 2:
        return splits

    possible_origins = list(range(3, last_test_start + 1))
    selected_origins = possible_origins[-n_splits:]

    for origin in selected_origins:
        train = series[:origin]
        test = series[origin:origin + horizon]
        if len(test) == horizon:
            splits.append((train, test))

    return splits


def evaluate_model_rolling(series: List[float], model_name: str, pattern: str, n_splits: int = 3, horizon: int = 1) -> Dict[str, Any]:
    splits = generate_rolling_splits(series, n_splits=n_splits, horizon=horizon)

    if not splits:
        raise ValueError("Zu wenig Historie für Rolling Backtesting.")

    split_results = []

    for idx, (train, test) in enumerate(splits, start=1):
        if not model_is_allowed(model_name, train, pattern):
            raise ValueError(f"Modell '{model_name}' ist in Split {idx} nicht zulässig.")

        preds = run_model(model_name, train, len(test))
        split_results.append(
            {
                "split": idx,
                "mae": mae(test, preds),
                "wape": wape(test, preds),
                "mase": mase(test, preds, train, m=1),
                "bias": bias(test, preds),
            }
        )

    avg_mae = float(np.mean([x["mae"] for x in split_results]))
    avg_wape = float(np.mean([x["wape"] for x in split_results]))
    avg_mase = float(np.mean([x["mase"] for x in split_results]))
    avg_bias = float(np.mean([x["bias"] for x in split_results]))

    return {
        "model": model_name,
        "metrics": {
            "mae": round(avg_mae, 6),
            "wape": round(avg_wape, 6),
            "mase": round(avg_mase, 6),
            "bias": round(avg_bias, 6),
        },
        "backtest_config": {
            "splits": len(split_results),
            "horizon": horizon,
        },
        "split_metrics": [
            {
                "split": x["split"],
                "mae": round(x["mae"], 6),
                "wape": round(x["wape"], 6),
                "mase": round(x["mase"], 6),
                "bias": round(x["bias"], 6),
            }
            for x in split_results
        ],
    }


def choose_best_model(series: List[float], pattern: str) -> Dict[str, Any]:
    candidate_models = [
        "naive",
        "seasonal_naive",
        "moving_average",
        "ets",
        "holt",
        "holt_winters",
        "arima",
        "sarima",
        "croston",
        "sba",
        "tsb",
        "gradient_boosting",
    ]

    valid_models = [m for m in candidate_models if model_is_allowed(m, series[:-1] if len(series) > 1 else series, pattern)]

    if not valid_models:
        raise ValueError("Keine zulässigen Modelle verfügbar.")

    results = []
    excluded_models = []

    for model_name in valid_models:
        try:
            result = evaluate_model_rolling(series, model_name, pattern, n_splits=3, horizon=1)
            results.append(result)
        except Exception as e:
            excluded_models.append({"model": model_name, "reason": str(e)})

    if not results:
        raise ValueError("Kein Modell konnte erfolgreich bewertet werden.")

    results.sort(
        key=lambda x: (
            0 if x["metrics"]["mase"] < 1 else 1,
            x["metrics"]["mae"],
            x["metrics"]["wape"],
            abs(x["metrics"]["bias"]),
        )
    )

    return {
        "best": results[0],
        "ranking": results,
        "excluded_models": excluded_models,
    }


# =========================================================
# API
# =========================================================

@app.get("/")
def root():
    return {"status": "ok", "message": "Demand Forecast API läuft"}


@app.post("/forecast")
def forecast(req: ForecastRequest):
    try:
        validate_inputs(req.demand, req.stockout, req.periods)
        cleaned = clean_demand(req.demand, req.stockout)

        if len(cleaned) == 0:
            return {
                "sku": req.sku,
                "model": "no_forecast",
                "forecast": [],
                "metrics": {"mae": 0.0, "wape": 0.0, "mase": 0.0, "bias": 0.0},
                "analysis": {
                    "demand_pattern": "no_demand",
                    "adi": None,
                    "cv2": None,
                },
                "backtest_config": {"splits": 0, "horizon": 0},
                "model_ranking": [],
                "excluded_models": [],
            }

        if count_nonzero(cleaned) == 0:
            return {
                "sku": req.sku,
                "model": "no_forecast",
                "forecast": [0.0] * req.periods,
                "metrics": {"mae": 0.0, "wape": 0.0, "mase": 0.0, "bias": 0.0},
                "analysis": {
                    "demand_pattern": "all_zero",
                    "adi": None,
                    "cv2": None,
                },
                "backtest_config": {"splits": 0, "horizon": 0},
                "model_ranking": [],
                "excluded_models": [],
            }

        pattern = detect_demand_pattern(cleaned)
        adi, cv2 = compute_adi_cv2(cleaned)

        if len(cleaned) < 6:
            selected_model = "naive"
            final_forecast = naive_forecast(cleaned, req.periods)
            return {
                "sku": req.sku,
                "model": selected_model,
                "forecast": [round(x, 6) for x in final_forecast],
                "metrics": {"mae": 0.0, "wape": 0.0, "mase": 0.0, "bias": 0.0},
                "analysis": {
                    "demand_pattern": pattern,
                    "adi": round(adi, 6) if math.isfinite(adi) else None,
                    "cv2": round(cv2, 6) if math.isfinite(cv2) else None,
                },
                "backtest_config": {"splits": 0, "horizon": 0},
                "model_ranking": [],
                "excluded_models": [],
            }

        if req.model == "auto":
            selection = choose_best_model(cleaned, pattern)
            selected_model = selection["best"]["model"]
            backtest_metrics = selection["best"]["metrics"]
            backtest_config = selection["best"]["backtest_config"]
            ranking = selection["ranking"]
            excluded_models = selection["excluded_models"]
        else:
            if not model_is_allowed(req.model, cleaned[:-1] if len(cleaned) > 1 else cleaned, pattern):
                raise ValueError(f"Modell '{req.model}' ist für diese Zeitreihe nicht zulässig.")

            evaluation = evaluate_model_rolling(cleaned, req.model, pattern, n_splits=3, horizon=1)
            selected_model = req.model
            backtest_metrics = evaluation["metrics"]
            backtest_config = evaluation["backtest_config"]
            ranking = [evaluation]
            excluded_models = []

        final_forecast = run_model(selected_model, cleaned, req.periods)

        return {
            "sku": req.sku,
            "model": selected_model,
            "forecast": [round(x, 6) for x in final_forecast],
            "metrics": backtest_metrics,
            "analysis": {
                "demand_pattern": pattern,
                "adi": round(adi, 6) if math.isfinite(adi) else None,
                "cv2": round(cv2, 6) if math.isfinite(cv2) else None,
            },
            "backtest_config": backtest_config,
            "model_ranking": [
                {
                    "model": item["model"],
                    "mae": item["metrics"]["mae"],
                    "wape": item["metrics"]["wape"],
                    "mase": item["metrics"]["mase"],
                    "bias": item["metrics"]["bias"],
                }
                for item in ranking
            ],
            "excluded_models": excluded_models,
        }

    except Exception as e:
        return {
            "sku": req.sku,
            "status": "failed",
            "reason": "forecast_error",
            "details": str(e),
        }
