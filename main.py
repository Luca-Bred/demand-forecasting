from typing import Any, Dict, List, Literal, Optional, Tuple
import io
import json
import math
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt, SimpleExpSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

app = FastAPI(title="Demand Forecast API", version="11.1.0")


# =========================================================
# AUTH
# =========================================================
API_BEARER_TOKEN = os.getenv("API_BEARER_TOKEN", "").strip()


def verify_bearer_token(authorization: Optional[str]) -> None:
    if not API_BEARER_TOKEN:
        raise HTTPException(status_code=500, detail="Server auth not configured.")

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")

    token = authorization.removeprefix("Bearer ").strip()
    if token != API_BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid bearer token")


# =========================================================
# REGISTRY / MODELLBINDUNG
# =========================================================
MODEL_REGISTRY_PATH = Path(os.getenv("MODEL_REGISTRY_PATH", "sku_model_registry.json"))
MODEL_SWITCH_WAPE_IMPROVEMENT = float(os.getenv("MODEL_SWITCH_WAPE_IMPROVEMENT", "0.10"))


def load_model_registry() -> Dict[str, Any]:
    if not MODEL_REGISTRY_PATH.exists():
        return {}
    try:
        with MODEL_REGISTRY_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_model_registry(registry: Dict[str, Any]) -> None:
    MODEL_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_REGISTRY_PATH.open("w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)


def get_established_model(sku: str) -> Optional[Dict[str, Any]]:
    registry = load_model_registry()
    item = registry.get(sku)
    return item if isinstance(item, dict) else None


def set_established_model(
    sku: str,
    model: str,
    metrics: Optional[Dict[str, float]] = None,
    source: str = "manual_confirmation",
) -> None:
    registry = load_model_registry()
    registry[sku] = {
        "model": model,
        "metrics": metrics or {},
        "source": source,
    }
    save_model_registry(registry)


def should_suggest_model_switch(
    established_model: str,
    established_metrics: Optional[Dict[str, Any]],
    best_model: str,
    best_metrics: Dict[str, Any],
) -> bool:
    if established_model == best_model:
        return False

    if not established_metrics:
        return True

    established_wape = established_metrics.get("wape")
    best_wape = best_metrics.get("wape")
    established_mase = established_metrics.get("mase")
    best_mase = best_metrics.get("mase")

    if isinstance(established_wape, (int, float)) and isinstance(best_wape, (int, float)):
        if established_wape > 0:
            relative_improvement = (established_wape - best_wape) / established_wape
            if relative_improvement >= MODEL_SWITCH_WAPE_IMPROVEMENT:
                return True

    if isinstance(established_mase, (int, float)) and isinstance(best_mase, (int, float)):
        if established_mase >= 1 and best_mase < 1:
            return True

    return False


# =========================================================
# KONFIGURATION
# =========================================================
SEASONAL_FACTORS: Dict[int, float] = {
    1: 1.00,
    2: 0.86,
    3: 1.03,
    4: 1.20,
    5: 1.16,
    6: 1.26,
    7: 1.28,
    8: 1.19,
    9: 1.08,
    10: 1.05,
    11: 0.93,
    12: 0.57,
}

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


# =========================================================
# RESPONSE MODELS
# =========================================================
class MetricsModel(BaseModel):
    mae: Optional[float]
    wape: Optional[float]
    mase: Optional[float]
    bias: Optional[float]


class AnalysisModel(BaseModel):
    demand_pattern: str
    adi: Optional[float]
    cv2: Optional[float]
    outlier_count: int
    outlier_flags: List[int]
    trend_factor: float
    forecast_start_month: int


class ModelRankingItem(BaseModel):
    model: str
    mae: Optional[float]
    wape: Optional[float]
    mase: Optional[float]
    bias: Optional[float]


class Top3ForecastItem(BaseModel):
    rank: int
    model: str
    metrics: MetricsModel
    forecast: List[int]


class PreprocessingModel(BaseModel):
    original: List[float]
    cleaned: List[float]


class EstablishedModelInfo(BaseModel):
    sku_has_established_model: bool
    established_model: Optional[str]
    selected_model: str
    model_switch_suggested: bool
    suggested_model: Optional[str]
    switch_reason: Optional[str]
    manual_confirmation_required: bool
    model_change_applied: bool


class ForecastResponse(BaseModel):
    status: str = "success"
    sku: str
    model: str
    forecast: List[int]
    metrics: MetricsModel
    analysis: AnalysisModel
    backtest_config: Dict[str, Any]
    model_ranking: List[ModelRankingItem]
    top_3_forecasts: List[Top3ForecastItem]
    excluded_models: List[Dict[str, str]]
    preprocessing: PreprocessingModel
    established_model_info: EstablishedModelInfo


class ErrorResponse(BaseModel):
    status: str = "failed"
    reason: str
    details: str


class ForecastRequest(BaseModel):
    sku: str = Field(..., min_length=1, max_length=100)
    demand: List[float] = Field(..., min_length=1)
    periods: int = Field(default=15, ge=1, le=24)
    model: AllowedModel = "auto"
    locked_model: Optional[AllowedModel] = None
    evaluation_horizon: int = Field(default=6, ge=1, le=15)
    n_splits: int = Field(default=3, ge=1, le=6)
    season_length: int = Field(default=12, ge=2, le=24)
    trend_factor: float = Field(default=1.0, ge=0.01, le=10.0)
    forecast_start_month: int = Field(default=1, ge=1, le=12)
    prefer_established_model: bool = Field(default=True)
    allow_model_change: bool = Field(default=False)
    confirm_model_change_to: Optional[str] = None
    save_selected_model_as_established: bool = Field(default=False)


# =========================================================
# VALIDATION & PREPROCESSING
# =========================================================
def validate_inputs(
    demand: List[float],
    periods: int,
    evaluation_horizon: int,
    n_splits: int,
    season_length: int,
) -> None:
    if len(demand) == 0:
        raise ValueError("demand darf nicht leer sein.")
    if len(demand) < 1:
        raise ValueError("Für eine Prognose wird mindestens 1 Wert benötigt.")
    if not all(isinstance(x, (int, float)) for x in demand):
        raise ValueError("Alle demand-Werte müssen numerisch sein.")
    if any(not math.isfinite(float(x)) for x in demand):
        raise ValueError("demand enthält ungültige Werte (NaN oder Infinity).")
    if periods < 1:
        raise ValueError("periods muss >= 1 sein.")
    if evaluation_horizon < 1:
        raise ValueError("evaluation_horizon muss >= 1 sein.")
    if n_splits < 1:
        raise ValueError("n_splits muss >= 1 sein.")
    if season_length < 2:
        raise ValueError("season_length muss >= 2 sein.")


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
    arr = np.asarray(series, dtype=float)
    flags = np.zeros(len(arr), dtype=int)

    flags[arr < 0] = 1

    non_zero = arr[arr > 0]
    if len(non_zero) < 3:
        return flags.tolist()

    med = float(np.median(non_zero))
    mad = robust_mad(non_zero)

    if mad > 0:
        robust_z = 0.6745 * (arr - med) / mad
        robust_mask = np.abs(robust_z) > 3.5
        flags[robust_mask] = 1

    for i in range(1, len(arr) - 1):
        prev_v = arr[i - 1]
        curr_v = arr[i]
        next_v = arr[i + 1]

        local_med = float(np.median([prev_v, next_v]))
        if local_med > 0:
            ratio = curr_v / local_med
            if ratio > 4.0 or ratio < 0.2:
                flags[i] = 1

        if curr_v == 0:
            local_avg = (prev_v + next_v) / 2.0
            if local_avg > med * 0.5:
                flags[i] = 1

    return flags.tolist()


def impute_series(series: List[float], outlier_flags: List[int]) -> List[float]:
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
# METRICS
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
# FORECAST MODELS
# =========================================================
def naive_forecast(train: List[float], periods: int) -> List[float]:
    if len(train) == 0:
        return [0.0] * periods
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
    if len(history) == 0:
        return [0.0] * periods

    forecasts: List[float] = []

    for _ in range(periods):
        effective_window = min(window, len(history))
        value = float(np.mean(history[-effective_window:]))
        forecasts.append(value)
        history.append(value)

    return forecasts


def ets_forecast(train: List[float], periods: int) -> List[float]:
    if len(train) < 2:
        return naive_forecast(train, periods)

    model = SimpleExpSmoothing(
        np.asarray(train, dtype=float),
        initialization_method="estimated",
    )
    fit = model.fit(optimized=True)
    fc = fit.forecast(periods)
    return [float(x) for x in fc]


def holt_forecast(train: List[float], periods: int) -> List[float]:
    if len(train) < 2:
        return naive_forecast(train, periods)

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
    if len(train) < 3:
        return naive_forecast(train, periods)

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
# ML FEATURES
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


def recursive_ml_forecast(model: Any, train: List[float], periods: int, season_length: int = 12) -> List[float]:
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
# BUSINESS ADJUSTMENTS
# =========================================================
def apply_business_adjustments(
    forecast: List[float],
    start_month: int,
    seasonal_factors: Dict[int, float],
    trend_factor: float = 1.0,
) -> List[float]:
    adjusted: List[float] = []

    for i, value in enumerate(forecast):
        month = ((start_month - 1 + i) % 12) + 1
        season_factor = float(seasonal_factors.get(month, 1.0))
        monthly_trend_factor = trend_factor ** (i / 12.0)

        new_val = float(value) * season_factor * monthly_trend_factor
        adjusted.append(max(0.0, new_val))

    return adjusted


# =========================================================
# MODEL DISPATCHER
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
# MODEL ELIGIBILITY
# =========================================================
def model_is_allowed(model_name: str, train: List[float], pattern: str, season_length: int = 12) -> bool:
    n = len(train)
    non_zero = count_nonzero(train)

    # Kurze Reihen: nur einfache Modelle
    if n < 12:
        return model_name in ("naive", "moving_average", "ets", "holt")

    if model_name == "naive":
        return n >= 1
    if model_name == "seasonal_naive":
        return n >= season_length * 2
    if model_name == "moving_average":
        return n >= 3
    if model_name == "ets":
        return n >= 2
    if model_name == "holt":
        return n >= 2
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
# ROLLING BACKTESTING
# =========================================================
def determine_backtest_config(
    series: List[float],
    requested_horizon: int,
    requested_splits: int,
) -> Tuple[int, int]:
    n = len(series)

    if n <= 1:
        return 1, 1

    if n < 12:
        horizon = min(3, n - 1)
    else:
        horizon = min(requested_horizon, n - 1)

    max_possible_splits = max(1, n - horizon - 1)
    splits = min(requested_splits, max_possible_splits)

    return max(1, horizon), max(1, splits)


def generate_rolling_splits(
    series: List[float],
    n_splits: int,
    horizon: int,
) -> List[Tuple[List[float], List[float]]]:
    n = len(series)
    splits: List[Tuple[List[float], List[float]]] = []

    if n <= 1:
        return [(series, series)]

    max_origin = n - horizon
    if max_origin <= 1:
        return [(series[:-1], series[-1:])]

    origins = list(range(1, max_origin))
    selected = origins[-n_splits:] if len(origins) >= n_splits else origins

    for origin in selected:
        train = series[:origin]
        test = series[origin:origin + horizon]

        if len(test) == 0:
            continue

        splits.append((train, test))

    if not splits:
        splits.append((series[:-1], series[-1:]))

    return splits


def evaluate_model_rolling(
    series: List[float],
    model_name: str,
    pattern: str,
    n_splits: int,
    horizon: int,
    season_length: int = 12,
) -> Dict[str, Any]:
    horizon, n_splits = determine_backtest_config(series, horizon, n_splits)
    splits = generate_rolling_splits(series, n_splits, horizon)

    split_results: List[Dict[str, Any]] = []

    for idx, (train, test) in enumerate(splits, start=1):
        if len(train) == 0 or len(test) == 0:
            continue

        if not model_is_allowed(model_name, train, pattern, season_length):
            continue

        try:
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
        except Exception:
            continue

    if not split_results:
        return {
            "model": model_name,
            "metrics": {"mae": None, "wape": None, "mase": None, "bias": None},
            "backtest_config": {"splits": 0, "horizon": horizon, "backtest_available": True},
            "split_metrics": [],
        }

    return {
        "model": model_name,
        "metrics": {
            "mae": round(float(np.mean([x["mae"] for x in split_results])), 2),
            "wape": round(float(np.mean([x["wape"] for x in split_results])), 4),
            "mase": round(float(np.mean([x["mase"] for x in split_results])), 4),
            "bias": round(float(np.mean([x["bias"] for x in split_results])), 2),
        },
        "backtest_config": {
            "splits": len(split_results),
            "horizon": horizon,
            "backtest_available": True,
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

    valid_models = [m for m in candidate_models if model_is_allowed(m, series, pattern, season_length=season_length)]

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
            1 if x["metrics"]["mase"] is None else (0 if x["metrics"]["mase"] < 1 else 1),
            x["metrics"]["wape"] if x["metrics"]["wape"] is not None else float("inf"),
            x["metrics"]["mae"] if x["metrics"]["mae"] is not None else float("inf"),
        )
    )

    return {
        "best": results[0],
        "ranking": results,
        "excluded_models": excluded_models,
    }


# =========================================================
# FORECAST CORE
# =========================================================
def metrics_none() -> Dict[str, Optional[float]]:
    return {"mae": None, "wape": None, "mase": None, "bias": None}


def build_success_response(
    sku: str,
    model: str,
    forecast: List[int],
    metrics: Dict[str, Optional[float]],
    analysis: Dict[str, Any],
    backtest_config: Dict[str, Any],
    model_ranking: List[Dict[str, Any]],
    top_3_forecasts: List[Dict[str, Any]],
    excluded_models: List[Dict[str, str]],
    preprocessing: Dict[str, Any],
    established_model_info: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "status": "success",
        "sku": sku,
        "model": model,
        "forecast": forecast,
        "metrics": metrics,
        "analysis": analysis,
        "backtest_config": backtest_config,
        "model_ranking": model_ranking,
        "top_3_forecasts": top_3_forecasts,
        "excluded_models": excluded_models,
        "preprocessing": preprocessing,
        "established_model_info": established_model_info,
    }


def fallback_model_for_short_history(cleaned: List[float], pattern: str, season_length: int) -> str:
    if len(cleaned) < 4:
        return "naive"
    if model_is_allowed("ets", cleaned, pattern, season_length):
        return "ets"
    if model_is_allowed("moving_average", cleaned, pattern, season_length):
        return "moving_average"
    return "naive"


def build_top_3_forecasts_from_ranking(
    ranking: List[Dict[str, Any]],
    cleaned: List[float],
    periods: int,
    season_length: int,
    trend_factor: float,
    forecast_start_month: int,
) -> List[Dict[str, Any]]:
    top_3: List[Dict[str, Any]] = []

    for rank, item in enumerate(ranking[:3], start=1):
        model_name = item["model"]

        fc = run_model(model_name, cleaned, periods, season_length=season_length)
        fc = apply_business_adjustments(
            forecast=fc,
            start_month=forecast_start_month,
            seasonal_factors=SEASONAL_FACTORS,
            trend_factor=trend_factor,
        )

        top_3.append(
            {
                "rank": rank,
                "model": model_name,
                "metrics": {
                    "mae": item["metrics"].get("mae"),
                    "wape": item["metrics"].get("wape"),
                    "mase": item["metrics"].get("mase"),
                    "bias": item["metrics"].get("bias"),
                },
                "forecast": round_forecast(fc),
            }
        )

    return top_3


def compute_forecast_from_request(req: ForecastRequest) -> Dict[str, Any]:
    validate_inputs(
        demand=req.demand,
        periods=req.periods,
        evaluation_horizon=req.evaluation_horizon,
        n_splits=req.n_splits,
        season_length=req.season_length,
    )

    prep = preprocess_demand(req.demand)
    cleaned = prep["cleaned"]

    if len(cleaned) == 0:
        return build_success_response(
            sku=req.sku,
            model="no_forecast",
            forecast=[],
            metrics=metrics_none(),
            analysis={
                "demand_pattern": "no_demand",
                "adi": None,
                "cv2": None,
                "outlier_count": prep["outlier_count"],
                "outlier_flags": prep["outlier_flags"],
                "trend_factor": req.trend_factor,
                "forecast_start_month": req.forecast_start_month,
            },
            backtest_config={"splits": 0, "horizon": 1, "backtest_available": True},
            model_ranking=[],
            top_3_forecasts=[],
            excluded_models=[],
            preprocessing={"original": prep["original"], "cleaned": []},
            established_model_info={
                "sku_has_established_model": False,
                "established_model": None,
                "selected_model": "no_forecast",
                "model_switch_suggested": False,
                "suggested_model": None,
                "switch_reason": None,
                "manual_confirmation_required": False,
                "model_change_applied": False,
            },
        )

    if count_nonzero(cleaned) == 0:
        return build_success_response(
            sku=req.sku,
            model="no_forecast",
            forecast=[0] * req.periods,
            metrics=metrics_none(),
            analysis={
                "demand_pattern": "all_zero",
                "adi": None,
                "cv2": None,
                "outlier_count": prep["outlier_count"],
                "outlier_flags": prep["outlier_flags"],
                "trend_factor": req.trend_factor,
                "forecast_start_month": req.forecast_start_month,
            },
            backtest_config={"splits": 0, "horizon": 1, "backtest_available": True},
            model_ranking=[],
            top_3_forecasts=[],
            excluded_models=[],
            preprocessing={
                "original": prep["original"],
                "cleaned": [round(x, 2) for x in cleaned],
            },
            established_model_info={
                "sku_has_established_model": False,
                "established_model": None,
                "selected_model": "no_forecast",
                "model_switch_suggested": False,
                "suggested_model": None,
                "switch_reason": None,
                "manual_confirmation_required": False,
                "model_change_applied": False,
            },
        )

    pattern = detect_demand_pattern(cleaned)
    adi, cv2 = compute_adi_cv2(cleaned)

    analysis = {
        "demand_pattern": pattern,
        "adi": round(adi, 4) if math.isfinite(adi) else None,
        "cv2": round(cv2, 4) if math.isfinite(cv2) else None,
        "outlier_count": prep["outlier_count"],
        "outlier_flags": prep["outlier_flags"],
        "trend_factor": req.trend_factor,
        "forecast_start_month": req.forecast_start_month,
    }

    preprocessing = {
        "original": prep["original"],
        "cleaned": [round(x, 2) for x in cleaned],
    }

    established = get_established_model(req.sku)
    established_model = established.get("model") if established else None
    established_metrics = established.get("metrics") if established else None

    ranking: List[Dict[str, Any]] = []
    excluded_models: List[Dict[str, str]] = []
    backtest_config: Dict[str, Any]
    best_model: str

    if req.locked_model and req.locked_model != "auto":
        if not model_is_allowed(req.locked_model, cleaned, pattern, req.season_length):
            raise ValueError(f"locked_model '{req.locked_model}' ist für diese Zeitreihe nicht zulässig.")

        best_model = req.locked_model
        evaluation = evaluate_model_rolling(
            series=cleaned,
            model_name=req.locked_model,
            pattern=pattern,
            n_splits=req.n_splits,
            horizon=req.evaluation_horizon,
            season_length=req.season_length,
        )
        ranking = [evaluation]
        backtest_config = evaluation["backtest_config"]

    elif req.model == "auto":
        selection = choose_best_model(
            series=cleaned,
            pattern=pattern,
            n_splits=req.n_splits,
            horizon=req.evaluation_horizon,
            season_length=req.season_length,
        )
        best_model = selection["best"]["model"]
        ranking = selection["ranking"]
        excluded_models = selection["excluded_models"]
        backtest_config = selection["best"]["backtest_config"]

    else:
        if not model_is_allowed(req.model, cleaned, pattern, req.season_length):
            raise ValueError(f"Modell '{req.model}' ist für diese Zeitreihe nicht zulässig.")

        best_model = req.model
        evaluation = evaluate_model_rolling(
            series=cleaned,
            model_name=req.model,
            pattern=pattern,
            n_splits=req.n_splits,
            horizon=req.evaluation_horizon,
            season_length=req.season_length,
        )
        ranking = [evaluation]
        backtest_config = evaluation["backtest_config"]

    selected_model = best_model
    model_switch_suggested = False
    suggested_model = None
    switch_reason = None
    manual_confirmation_required = False
    model_change_applied = False

    if req.model == "auto" and req.prefer_established_model and established_model:
        if model_is_allowed(established_model, cleaned, pattern, req.season_length):
            if established_model != best_model:
                suggestion = should_suggest_model_switch(
                    established_model=established_model,
                    established_metrics=established_metrics,
                    best_model=best_model,
                    best_metrics=ranking[0]["metrics"] if ranking else metrics_none(),
                )
                if suggestion:
                    model_switch_suggested = True
                    suggested_model = best_model
                    switch_reason = (
                        f"Für SKU '{req.sku}' ist '{established_model}' etabliert. "
                        f"'{best_model}' erscheint leistungsfähiger. "
                        f"Ein Modellwechsel erfordert eine manuelle Bestätigung."
                    )
                    manual_confirmation_required = True

                if req.allow_model_change and req.confirm_model_change_to == best_model:
                    selected_model = best_model
                    model_change_applied = True
                    set_established_model(
                        sku=req.sku,
                        model=best_model,
                        metrics=ranking[0]["metrics"] if ranking else None,
                        source="manual_confirmation",
                    )
                else:
                    selected_model = established_model
            else:
                selected_model = established_model
        else:
            selected_model = best_model
    elif req.allow_model_change and req.confirm_model_change_to:
        if req.confirm_model_change_to != best_model and req.confirm_model_change_to != selected_model:
            raise ValueError("confirm_model_change_to entspricht keinem zulässigen vorgeschlagenen Modell.")
        selected_model = req.confirm_model_change_to
        model_change_applied = True
        set_established_model(
            sku=req.sku,
            model=selected_model,
            metrics=ranking[0]["metrics"] if ranking else None,
            source="manual_confirmation",
        )

    final_forecast = run_model(selected_model, cleaned, req.periods, season_length=req.season_length)
    final_forecast = apply_business_adjustments(
        forecast=final_forecast,
        start_month=req.forecast_start_month,
        seasonal_factors=SEASONAL_FACTORS,
        trend_factor=req.trend_factor,
    )

    if req.save_selected_model_as_established and not established_model:
        set_established_model(
            sku=req.sku,
            model=selected_model,
            metrics=ranking[0]["metrics"] if ranking else None,
            source="explicit_save",
        )

    top_3_forecasts = build_top_3_forecasts_from_ranking(
        ranking=ranking,
        cleaned=cleaned,
        periods=req.periods,
        season_length=req.season_length,
        trend_factor=req.trend_factor,
        forecast_start_month=req.forecast_start_month,
    )

    ranking_top_8 = ranking[:8]
    model_ranking = [
        {
            "model": item["model"],
            "mae": item["metrics"].get("mae"),
            "wape": item["metrics"].get("wape"),
            "mase": item["metrics"].get("mase"),
            "bias": item["metrics"].get("bias"),
        }
        for item in ranking_top_8
    ]

    selected_metrics = next(
        (
            item["metrics"]
            for item in ranking
            if item["model"] == selected_model
        ),
        metrics_none(),
    )

    return build_success_response(
        sku=req.sku,
        model=selected_model,
        forecast=round_forecast(final_forecast),
        metrics={
            "mae": selected_metrics.get("mae"),
            "wape": selected_metrics.get("wape"),
            "mase": selected_metrics.get("mase"),
            "bias": selected_metrics.get("bias"),
        },
        analysis=analysis,
        backtest_config=backtest_config,
        model_ranking=model_ranking,
        top_3_forecasts=top_3_forecasts,
        excluded_models=excluded_models,
        preprocessing=preprocessing,
        established_model_info={
            "sku_has_established_model": bool(established_model),
            "established_model": established_model,
            "selected_model": selected_model,
            "model_switch_suggested": model_switch_suggested,
            "suggested_model": suggested_model,
            "switch_reason": switch_reason,
            "manual_confirmation_required": manual_confirmation_required,
            "model_change_applied": model_change_applied,
        },
    )


# =========================================================
# FILE HELPERS
# =========================================================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def detect_required_columns(df: pd.DataFrame) -> Tuple[str, str, Optional[str]]:
    col_map = {c.lower(): c for c in df.columns}

    sku_candidates = ["sku", "artikel"]
    demand_candidates = ["verbrauch", "demand", "menge"]
    date_candidates = ["datum", "date", "periode", "period", "monat"]

    sku_col = None
    demand_col = None
    date_col = None

    for c in sku_candidates:
        if c in col_map:
            sku_col = col_map[c]
            break

    for c in demand_candidates:
        if c in col_map:
            demand_col = col_map[c]
            break

    for c in date_candidates:
        if c in col_map:
            date_col = col_map[c]
            break

    if sku_col is None or demand_col is None:
        raise ValueError(
            "Datei muss Spalten für SKU und Verbrauch enthalten. "
            "Erlaubte Namen: SKU/Artikel und Verbrauch/Demand/Menge."
        )

    return sku_col, demand_col, date_col


def detect_file_format(filename: str) -> str:
    lower = (filename or "").lower()
    if lower.endswith(".csv"):
        return "csv"
    if lower.endswith(".xlsx"):
        return "xlsx"
    raise ValueError("Nur CSV oder XLSX als Input erlaubt.")


def read_input_file(file: UploadFile) -> pd.DataFrame:
    input_format = detect_file_format(file.filename or "")
    try:
        file.file.seek(0)
        if input_format == "csv":
            return pd.read_csv(file.file)
        if input_format == "xlsx":
            return pd.read_excel(file.file)
    except Exception as exc:
        raise ValueError(f"Fehler beim Lesen der Datei: {exc}") from exc

    raise ValueError("Nur CSV oder XLSX als Input erlaubt.")


def build_output_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for result in results:
        sku = result["sku"]
        model = result.get("model", "")
        metrics = result.get("metrics", {})
        analysis = result.get("analysis", {})
        ranking = result.get("model_ranking", [])
        top3 = result.get("top_3_forecasts", [])
        established = result.get("established_model_info", {})

        forecast_values = result.get("forecast", [])
        for i, value in enumerate(forecast_values, start=1):
            row: Dict[str, Any] = {
                "SKU": sku,
                "Selected_Model": model,
                "Forecast_Period": i,
                "Forecast": value,
                "MAE": metrics.get("mae"),
                "WAPE": metrics.get("wape"),
                "MASE": metrics.get("mase"),
                "BIAS": metrics.get("bias"),
                "Demand_Pattern": analysis.get("demand_pattern"),
                "ADI": analysis.get("adi"),
                "CV2": analysis.get("cv2"),
                "Outlier_Count": analysis.get("outlier_count"),
                "Trend_Factor": analysis.get("trend_factor"),
                "Forecast_Start_Month": analysis.get("forecast_start_month"),
                "Established_Model": established.get("established_model"),
                "Model_Switch_Suggested": established.get("model_switch_suggested"),
                "Suggested_Model": established.get("suggested_model"),
                "Manual_Confirmation_Required": established.get("manual_confirmation_required"),
                "Top_Ranked_Model": ranking[0]["model"] if ranking else model,
            }

            for idx, item in enumerate(top3[:3], start=1):
                fc = item.get("forecast", [])
                row[f"Top{idx}_Model"] = item.get("model")
                row[f"Top{idx}_MAE"] = item.get("metrics", {}).get("mae")
                row[f"Top{idx}_WAPE"] = item.get("metrics", {}).get("wape")
                row[f"Top{idx}_MASE"] = item.get("metrics", {}).get("mase")
                row[f"Top{idx}_BIAS"] = item.get("metrics", {}).get("bias")
                row[f"Top{idx}_Forecast"] = fc[i - 1] if i - 1 < len(fc) else None

            for idx, item in enumerate(ranking[:8], start=1):
                row[f"Rank{idx}_Model"] = item.get("model")
                row[f"Rank{idx}_MAE"] = item.get("mae")
                row[f"Rank{idx}_WAPE"] = item.get("wape")
                row[f"Rank{idx}_MASE"] = item.get("mase")
                row[f"Rank{idx}_BIAS"] = item.get("bias")

            rows.append(row)

    return pd.DataFrame(rows)


def dataframe_to_file_response(df: pd.DataFrame, output_format: str) -> StreamingResponse:
    output_format = output_format.lower().strip()

    if output_format == "csv":
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        return StreamingResponse(
            io.BytesIO(buffer.getvalue().encode("utf-8")),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=forecast_output.csv"},
        )

    if output_format == "xlsx":
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Forecast")
        buffer.seek(0)
        return StreamingResponse(
            buffer,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=forecast_output.xlsx"},
        )

    raise ValueError("output_format muss 'csv', 'xlsx' oder 'same' sein.")


# =========================================================
# API
# =========================================================
@app.get("/")
def root() -> Dict[str, str]:
    return {"status": "ok", "message": "Demand Forecast API läuft"}


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "healthy"}


@app.get("/model-binding/{sku}")
def get_model_binding(
    sku: str,
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    verify_bearer_token(authorization)
    established = get_established_model(sku)
    return {
        "sku": sku,
        "binding_exists": bool(established),
        "binding": established,
    }


@app.post("/model-binding/{sku}")
def set_model_binding_endpoint(
    sku: str,
    model: str = Form(...),
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    verify_bearer_token(authorization)
    set_established_model(sku=sku, model=model, metrics=None, source="manual_endpoint")
    return {
        "status": "success",
        "sku": sku,
        "established_model": model,
    }


@app.post(
    "/forecast",
    response_model=ForecastResponse,
    responses={400: {"model": ErrorResponse}, 401: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
def forecast(
    req: ForecastRequest,
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    verify_bearer_token(authorization)

    try:
        return compute_forecast_from_request(req)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "status": "failed",
                "reason": "forecast_validation_error",
                "details": str(exc),
            },
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "status": "failed",
                "reason": "forecast_internal_error",
                "details": str(exc),
            },
        ) from exc


@app.post("/forecast-file")
async def forecast_file(
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(default=None),
    trend_factor: float = Form(1.0),
    periods: int = Form(15),
    model: str = Form("auto"),
    locked_model: str = Form(""),
    evaluation_horizon: int = Form(6),
    n_splits: int = Form(3),
    season_length: int = Form(12),
    forecast_start_month: int = Form(1),
    output_format: str = Form("same"),
    prefer_established_model: bool = Form(True),
    allow_model_change: bool = Form(False),
    confirm_model_change_to: str = Form(""),
    save_selected_model_as_established: bool = Form(False),
):
    verify_bearer_token(authorization)

    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Es wurde keine Datei übergeben.")

        input_format = detect_file_format(file.filename)
        if output_format.lower().strip() == "same":
            output_format = input_format

        df = read_input_file(file)
        df = normalize_columns(df)

        if df.empty:
            raise HTTPException(status_code=400, detail="Die Eingabedatei ist leer.")

        sku_col, demand_col, date_col = detect_required_columns(df)

        use_cols = [sku_col, demand_col] + ([date_col] if date_col else [])
        df = df[use_cols].copy()

        df[demand_col] = pd.to_numeric(df[demand_col], errors="coerce")
        df = df.dropna(subset=[sku_col, demand_col])

        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.sort_values(by=[sku_col, date_col], kind="stable")
        else:
            df = df.reset_index(drop=True)

        if df.empty:
            raise HTTPException(status_code=400, detail="Keine verwertbaren Daten nach Bereinigung vorhanden.")

        results: List[Dict[str, Any]] = []

        for sku, group in df.groupby(sku_col, sort=False):
            demand = group[demand_col].astype(float).tolist()

            req = ForecastRequest(
                sku=str(sku),
                demand=demand,
                periods=periods,
                model=model,  # type: ignore[arg-type]
                locked_model=locked_model if locked_model else None,  # type: ignore[arg-type]
                evaluation_horizon=evaluation_horizon,
                n_splits=n_splits,
                season_length=season_length,
                trend_factor=trend_factor,
                forecast_start_month=forecast_start_month,
                prefer_established_model=prefer_established_model,
                allow_model_change=allow_model_change,
                confirm_model_change_to=confirm_model_change_to or None,
                save_selected_model_as_established=save_selected_model_as_established,
            )

            result = compute_forecast_from_request(req)
            results.append(result)

        result_df = build_output_dataframe(results)
        return dataframe_to_file_response(result_df, output_format=output_format)

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
