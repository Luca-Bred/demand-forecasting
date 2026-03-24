from typing import Any, Dict, List, Literal, Optional, Tuple
import io
import math
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
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

app = FastAPI(title="Demand Forecast API", version="10.0.0")

# =========================================================
# Saisonale Faktoren
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

class ForecastRequest(BaseModel):
    sku: str
    demand: List[float]
    periods: int = 15
    model: AllowedModel = "auto"
    season_length: int = 12
    yearly_trend: float = 0.0

    # NEU: optionales Datum für automatische Monatserkennung (Einzel-Request)
    last_observation_date: Optional[str] = None  # "YYYY-MM" oder "YYYY-MM-DD"

# =========================================================
# Monatserkennung
# =========================================================

def extract_month_from_date(date_str: str) -> int:
    try:
        dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
    except:
        dt = datetime.strptime(date_str[:7], "%Y-%m")
    return dt.month

def next_month(month: int) -> int:
    return 1 if month == 12 else month + 1

# =========================================================
# Backtest Config (IMMER aktiv)
# =========================================================

def determine_backtest_config(series: List[float]) -> Tuple[int, int]:
    n = len(series)

    if n <= 1:
        return 1, 1

    if n < 12:
        horizon = min(3, n - 1)
    else:
        horizon = min(6, n - 1)

    splits = max(1, n - horizon - 1)
    return horizon, splits

def generate_rolling_splits(series: List[float], splits: int, horizon: int):
    n = len(series)
    result = []

    for i in range(splits):
        train_end = n - horizon - splits + i + 1
        if train_end <= 0:
            continue

        train = series[:train_end]
        test = series[train_end:train_end + horizon]

        if len(test) > 0:
            result.append((train, test))

    if not result:
        result.append((series[:-1], series[-1:]))

    return result

# =========================================================
# Modelle
# =========================================================

def naive_forecast(train, periods):
    return [train[-1]] * periods

def ets_forecast(train, periods):
    model = SimpleExpSmoothing(train, initialization_method="estimated").fit()
    return model.forecast(periods).tolist()

def holt_forecast(train, periods):
    model = Holt(train, initialization_method="estimated").fit()
    return model.forecast(periods).tolist()

# =========================================================
# Modell-Restriktion bei kurzen Reihen
# =========================================================

def allowed_models(n):
    if n < 12:
        return ["naive", "moving_average", "ets", "holt"]
    return [
        "naive","ets","holt","holt_winters",
        "arima","sarima","gradient_boosting",
        "random_forest","extra_trees","hist_gradient_boosting","mlp"
    ]

# =========================================================
# Business Adjustment (JETZT korrekt mit Monat)
# =========================================================

def apply_adjustments(forecast, start_month, trend):
    result = []
    for i, val in enumerate(forecast):
        month = ((start_month - 1 + i) % 12) + 1
        factor = SEASONAL_FACTORS.get(month, 1.0)
        trend_factor = (1 + trend) ** (i / 12)
        result.append(val * factor * trend_factor)
    return result

# =========================================================
# Backtesting
# =========================================================

def backtest(series, model_name):
    horizon, splits = determine_backtest_config(series)
    split_sets = generate_rolling_splits(series, splits, horizon)

    maes = []

    for train, test in split_sets:
        if len(train) == 0:
            continue

        if model_name == "naive":
            preds = naive_forecast(train, len(test))
        elif model_name == "ets":
            preds = ets_forecast(train, len(test))
        else:
            preds = naive_forecast(train, len(test))

        error = np.mean(np.abs(np.array(test) - np.array(preds)))
        maes.append(error)

    if not maes:
        return None

    return float(np.mean(maes))

# =========================================================
# Forecast Core
# =========================================================

def compute_forecast(req: ForecastRequest, start_month: int):

    series = req.demand
    n = len(series)

    models = allowed_models(n)
    results = []

    for m in models:
        try:
            score = backtest(series, m)
            results.append((m, score))
        except:
            continue

    results = [r for r in results if r[1] is not None]
    results.sort(key=lambda x: x[1])

    best_model = results[0][0] if results else "naive"

    # Forecast erzeugen
    if best_model == "naive":
        fc = naive_forecast(series, req.periods)
    elif best_model == "ets":
        fc = ets_forecast(series, req.periods)
    else:
        fc = naive_forecast(series, req.periods)

    adjusted = apply_adjustments(fc, start_month, req.yearly_trend)

    return {
        "sku": req.sku,
        "model": best_model,
        "forecast": [int(round(x)) for x in adjusted],
        "top_3_models": results[:3],
        "top_8_models": results[:8],
    }

# =========================================================
# Datei-Handling (JETZT mit automatischer Monatserkennung)
# =========================================================

def detect_columns(df):
    cols = {c.lower(): c for c in df.columns}
    sku = cols.get("sku") or cols.get("artikel")
    demand = cols.get("verbrauch") or cols.get("demand")
    date = cols.get("date") or cols.get("datum")

    if not sku or not demand:
        raise ValueError("SKU und Verbrauch erforderlich")

    return sku, demand, date

# =========================================================
# API
# =========================================================

@app.post("/forecast")
def forecast(req: ForecastRequest):

    # Monat automatisch bestimmen
    if req.last_observation_date:
        last_month = extract_month_from_date(req.last_observation_date)
        start_month = next_month(last_month)
    else:
        start_month = 1  # fallback

    return compute_forecast(req, start_month)


@app.post("/forecast-file")
async def forecast_file(file: UploadFile = File(...)):

    df = pd.read_csv(file.file) if file.filename.endswith(".csv") else pd.read_excel(file.file)

    sku_col, demand_col, date_col = detect_columns(df)

    results = []

    for sku, group in df.groupby(sku_col):

        demand = group[demand_col].astype(float).tolist()

        # 🔥 automatische Monatserkennung
        if date_col:
            last_date = str(group[date_col].iloc[-1])
            last_month = extract_month_from_date(last_date)
            start_month = next_month(last_month)
        else:
            start_month = 1

        req = ForecastRequest(
            sku=str(sku),
            demand=demand
        )

        res = compute_forecast(req, start_month)
        results.append(res)

    return results
