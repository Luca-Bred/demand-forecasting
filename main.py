from typing import Any, Dict, List, Literal, Optional, Tuple
import math
import warnings

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt, SimpleExpSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

app = FastAPI(title="Demand Forecast API", version="7.1.0")


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
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "mlp",
]


class ForecastRequest(BaseModel):
    sku: str
    demand: List[float]
    periods: int = Field(default=15, ge=1, le=24)
    model: Optional[AllowedModel] = "auto"
    locked_model: Optional[AllowedModel] = None
    evaluation_horizon: int = Field(default=6, ge=1, le=15)
    n_splits: int = Field(default=3, ge=1, le=6)
    season_length: int = Field(default=12, ge=2, le=24)


# =========================================================
# Validation & preprocessing
# =========================================================

def validate_inputs(demand: List[float], periods: int, evaluation_horizon: int) -> None:
    if len(demand) == 0:
        raise ValueError("demand darf nicht leer sein.")
    if periods < 1:
        raise ValueError("periods muss >= 1 sein.")
    if evaluation_horizon < 1:
        raise ValueError("evaluation_horizon muss >= 1 sein.")


def clip_non_negative(values: List[float]) -> List[float]:
    return [float(max(0.0, x)) for x in values]


def round_forecast(values: List[float]) -> List[int]:
    return [int(round(max(0.0, x))) for x in values]


def count_nonzero(values: List[float]) -> int:
    return sum(1 for x in values if x > 0)


def robust_mad(x: np.ndarray) -> float:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(mad)


def detect_outliers_robust(series: List[float]) -> List[int]:
    """
    Robuste Outlier-Erkennung:
    - negative Werte -> Flag
    - robuste Z-Scores via Median/MAD
    - lokale harte Sprünge
    - verdächtige Nullen als möglicher Stockout / verdeckter Bedarf
    """
    arr = np.asarray(series, dtype=float)
    flags = np.zeros(len(arr), dtype=int)

    # negative Werte markieren
    flags[arr < 0] = 1

    # Statistik ohne Nullen, damit 0-Werte die Demand-Basis nicht verzerren
    non_zero = arr[arr > 0]
    if len(non_zero) < 3:
        return flags.tolist()

    med = float(np.median(non_zero))
    mad = robust_mad(non_zero)

    # globale robuste Ausreißererkennung
    if mad > 0:
        robust_z = 0.6745 * (arr - med) / mad
        robust_mask = np.abs(robust_z) > 3.5
        flags[robust_mask] = 1

    # lokale Peaks / Einbrüche + verdächtige Nullen
    for i in range(1, len(arr) - 1):
        prev_v = arr[i - 1]
        curr_v = arr[i]
        next_v = arr[i + 1]

        local_med = float(np.median([prev_v, next_v]))

        if local_med > 0:
            ratio = curr_v / local_med
            if ratio > 4.0 or ratio < 0.2:
                flags[i] = 1

        # isolierte 0 in sonst positivem Umfeld -> verdächtig
        if curr_v == 0:
            local_avg = (prev_v + next_v) / 2.0
            if local_avg > med * 0.5:
                flags[i] = 1

    return flags.tolist()


def impute_series(series: List[float], outlier_flags: List[int]) -> List[float]:
    """
    Ausreißer/verdächtige Werte werden als fehlend behandelt und interpoliert.
    Nicht pauschal auf 0 setzen.
    """
    arr = np.asarray(series, dtype=float)

    for i, flag in enumerate(outlier_flags):
        if flag == 1:
            arr[i] = np.nan

    if np.all(np.isnan(arr)):
        return [0.0] * len(arr)

    idx = np.arange(len(arr))
    valid_mask = ~np.isnan(arr)

    if valid_mask.sum() == 1:
        fill_value = float(arr[valid_mask][0])
        arr = np.where(np.isnan(arr), fill_value, arr)
    else:
        arr[~valid_mask] = np.interp(idx[~valid_mask], idx[valid_mask], arr[valid_mask])

    arr = np.maximum(arr, 0.0)
    return [float(x) for x in arr]


def preprocess_demand(series: List[float]) -> Dict[str, Any]:
    outlier_flags = detect_outliers_robust(series)
    cleaned = impute_series(series, outlier_flags)

    return {
        "original": [float(v) for v in series],
        "cleaned": cleaned,
        "outlier_flags": outlier_flags,
        "outlier_count": int(sum(outlier_flags)),
    }


def compute_adi_cv2(series: List[float]) -> Tuple[float, float]:
    if not series:
        return float("inf"), float("inf")

    non_zero = [x for x in series if x > 0]
    if len(non_zero) == 0:
        return float("inf"), float("inf")

    adi = len(series) / len(non_zero)
    mean_nz = float(np.mean(non_zero))
    std_nz = float(np.std(non_zero, ddof=0))
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
    forecasts: List[float] = []

    for i in range(periods):
        idx = len(history) - season_length + (i % season_length)
        forecasts.append(history[idx])

    return forecasts


def moving_average_forecast(train: List[float], periods: int, window: int = 3) -> List[float]:
    history = list(map(float, train))
    forecasts: List[float] = []

    for _ in range(periods):
        effective_window = min(window, len(history))
        value = float(np.mean(history[-effective_window:]))
        forecasts.append(value)
        history.append(value)

    return forecasts


def ets_forecast(train: List[float], periods: int) -> List[float]:
    model = SimpleExpSmoothing(
        np.asarray(train, dtype=float),
        initialization_method="estimated",
    )
    fit = model.fit(optimized=True)
    fc = fit.forecast(periods)
    return [float(x) for x in fc]


def holt_forecast(train: List[float], periods: int) -> List[float]:
    model = Holt(
        np.asarray(train, dtype=float),
        initialization_method="estimated",
    )
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


def arima_forecast(train: List[float], periods: int, order: Tuple[int, int, int] = (1, 1, 1)) -> List[float]:
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


def sarima_forecast(
    train: List[float],
    periods: int,
    order: Tuple[int, int, int] = (1, 1, 1),
    seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 12),
) -> List[float]:
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
    ts = np.asarray(train, dtype=float)
    demand_sizes: List[float] = []
    intervals: List[int] = []

    interval = 1
    for x in ts:
        if x > 0:
            demand_sizes.append(float(x))
            intervals.append(interval)
            interval = 1
        else:
            interval += 1

    if len(demand_sizes) == 0:
        return [0.0] * periods

    z = demand_sizes[0]
    p = float(intervals[0])

    for i in range(1, len(demand_sizes)):
        z = alpha * demand_sizes[i] + (1 - alpha) * z
        p = alpha * intervals[i] + (1 - alpha) * p

    f = float(z / p) if p != 0 else 0.0
    return [f] * periods


def sba_forecast(train: List[float], periods: int, alpha: float = 0.1) -> List[float]:
    croston_fc = croston_forecast(train, periods, alpha=alpha)
    correction = 1 - alpha / 2
    return [float(x * correction) for x in croston_fc]


def tsb_forecast(train: List[float], periods: int, alpha: float = 0.2, beta: float = 0.2) -> List[float]:
    ts = np.asarray(train, dtype=float)

    if len(ts) == 0 or np.all(ts == 0):
        return [0.0] * periods

    occurrence = np.where(ts > 0, 1.0, 0.0)
    first_nonzero_idx = int(np.argmax(ts > 0))

    z = float(ts[first_nonzero_idx])
    p = float(occurrence[first_nonzero_idx])

    for x in ts[first_nonzero_idx + 1:]:
        occ = 1.0 if x > 0 else 0.0
        p = beta * occ + (1 - beta) * p
        if x > 0:
            z = alpha * x + (1 - alpha) * z

    f = z * p
    return [float(f)] * periods


# =========================================================
# ML features
# =========================================================

def create_ml_features(series: List[float], season_length: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    values = list(map(float, series))
    rows: List[List[float]] = []
    targets: List[float] = []

    if len(values) <= 12:
        raise ValueError("Zu wenig Historie für ML-Features.")

    for t in range(12, len(values)):
        lag1 = values[t - 1]
        lag2 = values[t - 2]
        lag3 = values[t - 3]
        lag6 = values[t - 6]
        lag12 = values[t - 12]
        rmean3 = float(np.mean(values[t - 3:t]))
        rmean6 = float(np.mean(values[t - 6:t]))
        rstd3 = float(np.std(values[t - 3:t], ddof=0))
        rstd6 = float(np.std(values[t - 6:t], ddof=0))
        month_idx = float(t % season_length)

        rows.append([lag1, lag2, lag3, lag6, lag12, rmean3, rmean6, rstd3, rstd6, month_idx])
        targets.append(values[t])

    return np.asarray(rows), np.asarray(targets)


def recursive_ml_forecast(model, train: List[float], periods: int, season_length: int = 12) -> List[float]:
    history = list(map(float, train))
    forecasts: List[float] = []

    for _ in range(periods):
        t = len(history)
        lag1 = history[t - 1]
        lag2 = history[t - 2]
        lag3 = history[t - 3]
        lag6 = history[t - 6]
        lag12 = history[t - 12]
        rmean3 = float(np.mean(history[t - 3:t]))
        rmean6 = float(np.mean(history[t - 6:t]))
        rstd3 = float(np.std(history[t - 3:t], ddof=0))
        rstd6 = float(np.std(history[t - 6:t], ddof=0))
        month_idx = float(t % season_length)

        x_new = np.asarray([[lag1, lag2, lag3, lag6, lag12, rmean3, rmean6, rstd3, rstd6, month_idx]])
        pred = float(model.predict(x_new)[0])
        pred = max(0.0, pred)

        forecasts.append(pred)
        history.append(pred)

    return forecasts


def gradient_boosting_forecast(train: List[float], periods: int, season_length: int = 12) -> List[float]:
    if len(train) < 24:
        raise ValueError("Zu wenig Historie für Gradient Boosting.")

    X, y = create_ml_features(train, season_length=season_length)
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X, y)
    return recursive_ml_forecast(model, train, periods, season_length=season_length)


def random_forest_forecast(train: List[float], periods: int, season_length: int = 12) -> List[float]:
    if len(train) < 24:
        raise ValueError("Zu wenig Historie für Random Forest.")

    X, y = create_ml_features(train, season_length=season_length)
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    return recursive_ml_forecast(model, train, periods, season_length=season_length)


def extra_trees_forecast(train: List[float], periods: int, season_length: int = 12) -> List[float]:
    if len(train) < 24:
        raise ValueError("Zu wenig Historie für Extra Trees.")

    X, y = create_ml_features(train, season_length=season_length)
    model = ExtraTreesRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    return recursive_ml_forecast(model, train, periods, season_length=season_length)


def hist_gradient_boosting_forecast(train: List[float], periods: int, season_length: int = 12) -> List[float]:
    if len(train) < 24:
        raise ValueError("Zu wenig Historie für HistGradientBoosting.")

    X, y = create_ml_features(train, season_length=season_length)
    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=6,
        max_iter=300,
        random_state=42,
    )
    model.fit(X, y)
    return recursive_ml_forecast(model, train, periods, season_length=season_length)


def mlp_forecast(train: List[float], periods: int, season_length: int = 12) -> List[float]:
    if len(train) < 24:
        raise ValueError("Zu wenig Historie für MLP.")

    X, y = create_ml_features(train, season_length=season_length)
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=0.001,
                learning_rate_init=0.001,
                max_iter=1000,
                random_state=42,
            )),
        ]
    )
    model.fit(X, y)
    return recursive_ml_forecast(model, train, periods, season_length=season_length)


# =========================================================
# Model dispatcher
# =========================================================

def run_model(model_name: str, train: List[float], periods: int, season_length: int = 12) -> List[float]:
    if model_name == "naive":
        return naive_forecast(train, periods)
    if model_name == "seasonal_naive":
        return seasonal_naive_forecast(train, periods, season_length=season_length)
    if model_name == "moving_average":
        return moving_average_forecast(train, periods, window=3)
    if model_name == "ets":
        return ets_forecast(train, periods)
    if model_name == "holt":
        return holt_forecast(train, periods)
    if model_name == "holt_winters":
        return holt_winters_forecast(train, periods, season_length=season_length)
    if model_name == "arima":
        return arima_forecast(train, periods, order=(1, 1, 1))
    if model_name == "sarima":
        return sarima_forecast(train, periods, order=(1, 1, 1), seasonal_order=(1, 1, 1, season_length))
    if model_name == "croston":
        return croston_forecast(train, periods, alpha=0.1)
    if model_name == "sba":
        return sba_forecast(train, periods, alpha=0.1)
    if model_name == "tsb":
        return tsb_forecast(train, periods, alpha=0.2, beta=0.2)
    if model_name == "gradient_boosting":
        return gradient_boosting_forecast(train, periods, season_length=season_length)
    if model_name == "random_forest":
        return random_forest_forecast(train, periods, season_length=season_length)
    if model_name == "extra_trees":
        return extra_trees_forecast(train, periods, season_length=season_length)
    if model_name == "hist_gradient_boosting":
        return hist_gradient_boosting_forecast(train, periods, season_length=season_length)
    if model_name == "mlp":
        return mlp_forecast(train, periods, season_length=season_length)

    raise ValueError(f"Unbekanntes Modell: {model_name}")


# =========================================================
# Model eligibility
# =========================================================

def model_is_allowed(model_name: str, train: List[float], pattern: str, season_length: int = 12) -> bool:
    n = len(train)
    non_zero = count_nonzero(train)

    if model_name == "naive":
        return n >= 1
    if model_name == "seasonal_naive":
        return n >= season_length * 2
    if model_name == "moving_average":
        return n >= 3
    if model_name == "ets":
        return n >= 4
    if model_name == "holt":
        return n >= 6
    if model_name == "holt_winters":
        return n >= season_length * 2
    if model_name == "arima":
        return n >= 12
    if model_name == "sarima":
        return n >= season_length * 2
    if model_name == "croston":
        return pattern in ("intermittent", "lumpy") and non_zero >= 2
    if model_name == "sba":
        return pattern in ("intermittent", "lumpy") and non_zero >= 2
    if model_name == "tsb":
        return pattern in ("intermittent", "lumpy") and non_zero >= 2
    if model_name == "gradient_boosting":
        return n >= 24
    if model_name == "random_forest":
        return n >= 24
    if model_name == "extra_trees":
        return n >= 24
    if model_name == "hist_gradient_boosting":
        return n >= 24
    if model_name == "mlp":
        return n >= 24

    return False


# =========================================================
# Rolling backtesting
# =========================================================

def generate_rolling_splits(
    series: List[float],
    n_splits: int = 3,
    horizon: int = 6,
    min_train_size: int = 6,
) -> List[Tuple[List[float], List[float]]]:
    splits: List[Tuple[List[float], List[float]]] = []
    n = len(series)

    if n <= horizon + min_train_size:
        return splits

    possible_origins = list(range(min_train_size, n - horizon + 1))
    if not possible_origins:
        return splits

    selected_origins = possible_origins[-n_splits:]

    for origin in selected_origins:
        train = series[:origin]
        test = series[origin:origin + horizon]
        if len(test) == horizon:
            splits.append((train, test))

    return splits


def evaluate_model_rolling(
    series: List[float],
    model_name: str,
    pattern: str,
    n_splits: int = 3,
    horizon: int = 6,
    season_length: int = 12,
) -> Dict[str, Any]:
    splits = generate_rolling_splits(
        series=series,
        n_splits=n_splits,
        horizon=horizon,
        min_train_size=max(6, min(season_length, len(series) - horizon - 1)) if len(series) > horizon + 1 else 6,
    )

    if not splits:
        raise ValueError("Zu wenig Historie für Rolling Backtesting.")

    split_results: List[Dict[str, Any]] = []

    for idx, (train, test) in enumerate(splits, start=1):
        if not model_is_allowed(model_name, train, pattern, season_length=season_length):
            raise ValueError(f"Modell '{model_name}' ist in Split {idx} nicht zulässig.")

        preds = run_model(model_name, train, len(test), season_length=season_length)
        preds = clip_non_negative(preds)

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
            "mae": round(avg_mae, 2),
            "wape": round(avg_wape, 4),
            "mase": round(avg_mase, 4),
            "bias": round(avg_bias, 2),
        },
        "backtest_config": {
            "splits": len(split_results),
            "horizon": horizon,
        },
        "split_metrics": [
            {
                "split": x["split"],
                "mae": round(x["mae"], 2),
                "wape": round(x["wape"], 4),
                "mase": round(x["mase"], 4),
                "bias": round(x["bias"], 2),
            }
            for x in split_results
        ],
    }


def choose_best_model(
    series: List[float],
    pattern: str,
    n_splits: int = 3,
    horizon: int = 6,
    season_length: int = 12,
) -> Dict[str, Any]:
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
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
        "mlp",
    ]

    ref_series = series[:-horizon] if len(series) > horizon else series[:-1] if len(series) > 1 else series
    valid_models = [m for m in candidate_models if model_is_allowed(m, ref_series, pattern, season_length=season_length)]

    if not valid_models:
        raise ValueError("Keine zulässigen Modelle verfügbar.")

    results: List[Dict[str, Any]] = []
    excluded_models: List[Dict[str, str]] = []

    for model_name in valid_models:
        try:
            result = evaluate_model_rolling(
                series=series,
                model_name=model_name,
                pattern=pattern,
                n_splits=n_splits,
                horizon=horizon,
                season_length=season_length,
            )
            results.append(result)
        except Exception as exc:
            excluded_models.append({"model": model_name, "reason": str(exc)})

    if not results:
        raise ValueError("Kein Modell konnte erfolgreich bewertet werden.")

    results.sort(
        key=lambda x: (
            0 if x["metrics"]["mase"] < 1 else 1,
            x["metrics"]["wape"],
            x["metrics"]["mae"],
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
def root() -> Dict[str, str]:
    return {"status": "ok", "message": "Demand Forecast API läuft"}


@app.post("/forecast")
def forecast(req: ForecastRequest) -> Dict[str, Any]:
    try:
        validate_inputs(req.demand, req.periods, req.evaluation_horizon)

        prep = preprocess_demand(req.demand)
        cleaned = prep["cleaned"]

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
                    "outlier_count": prep["outlier_count"],
                },
                "backtest_config": {"splits": 0, "horizon": 0},
                "model_ranking": [],
                "excluded_models": [],
            }

        if count_nonzero(cleaned) == 0:
            return {
                "sku": req.sku,
                "model": "no_forecast",
                "forecast": [0] * req.periods,
                "metrics": {"mae": 0.0, "wape": 0.0, "mase": 0.0, "bias": 0.0},
                "analysis": {
                    "demand_pattern": "all_zero",
                    "adi": None,
                    "cv2": None,
                    "outlier_count": prep["outlier_count"],
                },
                "backtest_config": {"splits": 0, "horizon": 0},
                "model_ranking": [],
                "excluded_models": [],
                "preprocessing": {
                    "original": prep["original"],
                    "cleaned": [round(x, 2) for x in cleaned],
                },
            }

        pattern = detect_demand_pattern(cleaned)
        adi, cv2 = compute_adi_cv2(cleaned)

        # Sehr kurze Historie
        if len(cleaned) < 4:
            selected_model = "naive"
            final_forecast = naive_forecast(cleaned, req.periods)

            return {
                "sku": req.sku,
                "model": selected_model,
                "model_changed": False,
                "forecast": round_forecast(final_forecast),
                "metrics": {"mae": 0.0, "wape": 0.0, "mase": 0.0, "bias": 0.0},
                "analysis": {
                    "demand_pattern": pattern,
                    "adi": round(adi, 4) if math.isfinite(adi) else None,
                    "cv2": round(cv2, 4) if math.isfinite(cv2) else None,
                    "outlier_count": prep["outlier_count"],
                    "outlier_flags": prep["outlier_flags"],
                },
                "backtest_config": {"splits": 0, "horizon": 0},
                "model_ranking": [],
                "excluded_models": [],
                "preprocessing": {
                    "original": prep["original"],
                    "cleaned": [round(x, 2) for x in cleaned],
                },
            }

        # Kurze Historie -> einfache Verfahren
        if len(cleaned) < 8 and req.model == "auto" and req.locked_model is None:
            selected_model = "ets" if model_is_allowed("ets", cleaned, pattern, req.season_length) else "naive"
            final_forecast = run_model(selected_model, cleaned, req.periods, season_length=req.season_length)

            return {
                "sku": req.sku,
                "model": selected_model,
                "model_changed": False,
                "forecast": round_forecast(final_forecast),
                "metrics": {"mae": 0.0, "wape": 0.0, "mase": 0.0, "bias": 0.0},
                "analysis": {
                    "demand_pattern": pattern,
                    "adi": round(adi, 4) if math.isfinite(adi) else None,
                    "cv2": round(cv2, 4) if math.isfinite(cv2) else None,
                    "outlier_count": prep["outlier_count"],
                    "outlier_flags": prep["outlier_flags"],
                },
                "backtest_config": {"splits": 0, "horizon": 0},
                "model_ranking": [],
                "excluded_models": [],
                "preprocessing": {
                    "original": prep["original"],
                    "cleaned": [round(x, 2) for x in cleaned],
                },
            }

        model_changed = False

        if req.locked_model and req.locked_model != "auto":
            if not model_is_allowed(req.locked_model, cleaned, pattern, req.season_length):
                raise ValueError(f"locked_model '{req.locked_model}' ist für diese Zeitreihe nicht zulässig.")

            evaluation = evaluate_model_rolling(
                series=cleaned,
                model_name=req.locked_model,
                pattern=pattern,
                n_splits=req.n_splits,
                horizon=req.evaluation_horizon,
                season_length=req.season_length,
            )
            selected_model = req.locked_model
            backtest_metrics = evaluation["metrics"]
            backtest_config = evaluation["backtest_config"]
            ranking = [evaluation]
            excluded_models = []

        elif req.model == "auto":
            selection = choose_best_model(
                series=cleaned,
                pattern=pattern,
                n_splits=req.n_splits,
                horizon=req.evaluation_horizon,
                season_length=req.season_length,
            )
            selected_model = selection["best"]["model"]
            backtest_metrics = selection["best"]["metrics"]
            backtest_config = selection["best"]["backtest_config"]
            ranking = selection["ranking"]
            excluded_models = selection["excluded_models"]

        else:
            if not model_is_allowed(req.model, cleaned, pattern, req.season_length):
                raise ValueError(f"Modell '{req.model}' ist für diese Zeitreihe nicht zulässig.")

            evaluation = evaluate_model_rolling(
                series=cleaned,
                model_name=req.model,
                pattern=pattern,
                n_splits=req.n_splits,
                horizon=req.evaluation_horizon,
                season_length=req.season_length,
            )
            selected_model = req.model
            backtest_metrics = evaluation["metrics"]
            backtest_config = evaluation["backtest_config"]
            ranking = [evaluation]
            excluded_models = []

        final_forecast = run_model(selected_model, cleaned, req.periods, season_length=req.season_length)

        return {
            "sku": req.sku,
            "model": selected_model,
            "model_changed": model_changed,
            "forecast": round_forecast(final_forecast),
            "metrics": {
                "mae": round(backtest_metrics["mae"], 2),
                "wape": round(backtest_metrics["wape"], 4),
                "mase": round(backtest_metrics["mase"], 4),
                "bias": round(backtest_metrics["bias"], 2),
            },
            "analysis": {
                "demand_pattern": pattern,
                "adi": round(adi, 4) if math.isfinite(adi) else None,
                "cv2": round(cv2, 4) if math.isfinite(cv2) else None,
                "outlier_count": prep["outlier_count"],
                "outlier_flags": prep["outlier_flags"],
            },
            "backtest_config": backtest_config,
            "model_ranking": [
                {
                    "model": item["model"],
                    "mae": round(item["metrics"]["mae"], 2),
                    "wape": round(item["metrics"]["wape"], 4),
                    "mase": round(item["metrics"]["mase"], 4),
                    "bias": round(item["metrics"]["bias"], 2),
                }
                for item in ranking
            ],
            "excluded_models": excluded_models,
            "preprocessing": {
                "original": prep["original"],
                "cleaned": [round(x, 2) for x in cleaned],
            },
        }

    except Exception as exc:
        return {
            "sku": req.sku,
            "status": "failed",
            "reason": "forecast_error",
            "details": str(exc),
        }
