from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Literal, Dict, Any
import math

app = FastAPI(title="Demand Forecast API", version="3.0.0")


class ForecastRequest(BaseModel):
    sku: str
    demand: List[float]
    stockout: List[int]
    periods: int = 15
    model: Optional[Literal["auto", "naive", "moving_average", "ets", "croston"]] = "auto"


def validate_inputs(demand: List[float], stockout: List[int]) -> None:
    if len(demand) != len(stockout):
        raise ValueError("demand und stockout müssen gleich lang sein.")
    if len(demand) == 0:
        raise ValueError("demand darf nicht leer sein.")
    if any(s not in (0, 1) for s in stockout):
        raise ValueError("stockout darf nur 0 oder 1 enthalten.")


def clean_demand(demand: List[float], stockout: List[int]) -> List[float]:
    return [d for d, s in zip(demand, stockout) if s == 0]


def count_nonzero(values: List[float]) -> int:
    return sum(1 for x in values if x > 0)


def demand_pattern(cleaned: List[float]) -> str:
    if not cleaned:
        return "no_demand"

    nz = [x for x in cleaned if x > 0]
    if not nz:
        return "all_zero"

    zero_share = sum(1 for x in cleaned if x == 0) / len(cleaned)
    adi = len(cleaned) / len(nz) if len(nz) > 0 else float("inf")

    if adi > 1.32 or zero_share > 0.4:
        return "intermittent"
    return "continuous"


def naive_forecast(train: List[float], periods: int) -> List[float]:
    if not train:
        return []
    return [float(train[-1])] * periods


def moving_average_forecast(train: List[float], periods: int, window: int = 3) -> List[float]:
    if not train:
        return []
    effective_window = min(window, len(train))
    avg = sum(train[-effective_window:]) / effective_window
    return [float(avg)] * periods


def simple_exponential_smoothing_forecast(train: List[float], periods: int, alpha: float = 0.3) -> List[float]:
    """
    Einfache ETS-Variante ohne Trend/Saisonalität (SES).
    """
    if not train:
        return []
    level = float(train[0])
    for x in train[1:]:
        level = alpha * float(x) + (1 - alpha) * level
    return [level] * periods


def croston_forecast(train: List[float], periods: int, alpha: float = 0.1) -> List[float]:
    """
    Einfache Croston-Implementierung für intermittierende Nachfrage.
    """
    if not train:
        return []

    demand_sizes = []
    intervals = []

    interval = 1
    for x in train:
        if x > 0:
            demand_sizes.append(float(x))
            intervals.append(interval)
            interval = 1
        else:
            interval += 1

    if not demand_sizes:
        return [0.0] * periods

    z = demand_sizes[0]
    p = float(intervals[0])

    for i in range(1, len(demand_sizes)):
        z = alpha * demand_sizes[i] + (1 - alpha) * z
        p = alpha * intervals[i] + (1 - alpha) * p

    forecast_value = z / p if p != 0 else 0.0
    return [forecast_value] * periods


def mae(actual: List[float], predicted: List[float]) -> float:
    if not actual:
        return 0.0
    return sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual)


def wape(actual: List[float], predicted: List[float]) -> float:
    denom = sum(abs(a) for a in actual)
    if denom == 0:
        return 0.0
    return sum(abs(a - p) for a, p in zip(actual, predicted)) / denom


def bias(actual: List[float], predicted: List[float]) -> float:
    if not actual:
        return 0.0
    return sum(p - a for a, p in zip(actual, predicted)) / len(actual)


def mase(actual: List[float], predicted: List[float], train: List[float]) -> float:
    if len(train) < 2 or not actual:
        return 0.0

    naive_errors = [abs(train[i] - train[i - 1]) for i in range(1, len(train))]
    scale = sum(naive_errors) / len(naive_errors) if naive_errors else 0.0

    if scale == 0:
        return 0.0

    return mae(actual, predicted) / scale


def model_is_allowed(model_name: str, train: List[float], pattern: str) -> bool:
    if model_name == "croston":
        return pattern == "intermittent" and count_nonzero(train) >= 2
    if model_name == "ets":
        return len(train) >= 3
    if model_name == "moving_average":
        return len(train) >= 2
    if model_name == "naive":
        return len(train) >= 1
    return False


def run_model(model_name: str, train: List[float], periods: int) -> List[float]:
    if model_name == "naive":
        return naive_forecast(train, periods)
    if model_name == "moving_average":
        return moving_average_forecast(train, periods, window=3)
    if model_name == "ets":
        return simple_exponential_smoothing_forecast(train, periods, alpha=0.3)
    if model_name == "croston":
        return croston_forecast(train, periods, alpha=0.1)
    raise ValueError(f"Unbekanntes Modell: {model_name}")


def evaluate_model(train: List[float], test: List[float], model_name: str) -> Dict[str, Any]:
    preds = run_model(model_name, train, len(test))
    return {
        "model": model_name,
        "forecast": preds,
        "metrics": {
            "mae": round(mae(test, preds), 6),
            "wape": round(wape(test, preds), 6),
            "mase": round(mase(test, preds, train), 6),
            "bias": round(bias(test, preds), 6),
        },
    }


def choose_best_model(train: List[float], test: List[float], pattern: str) -> Dict[str, Any]:
    candidate_models = ["naive", "moving_average", "ets", "croston"]
    valid_candidates = [
        m for m in candidate_models if model_is_allowed(m, train, pattern)
    ]

    if not valid_candidates:
        raise ValueError("Keine zulässigen Modelle für diese Zeitreihe verfügbar.")

    results = [evaluate_model(train, test, m) for m in valid_candidates]

    # Auswahlregel:
    # 1. MASE < 1 bevorzugen
    # 2. min(MAE)
    # 3. min(WAPE)
    # 4. Bias möglichst nahe 0
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
        "ranking": results
    }


@app.get("/")
def root():
    return {"status": "ok", "message": "Demand Forecast API läuft"}


@app.post("/forecast")
def forecast(req: ForecastRequest):
    try:
        validate_inputs(req.demand, req.stockout)
        cleaned = clean_demand(req.demand, req.stockout)

        if len(cleaned) == 0:
            return {
                "sku": req.sku,
                "model": "no_forecast",
                "forecast": [],
                "metrics": {
                    "mae": 0.0,
                    "wape": 0.0,
                    "mase": 0.0,
                    "bias": 0.0
                },
                "analysis": {
                    "demand_pattern": "no_demand"
                },
                "model_ranking": []
            }

        if count_nonzero(cleaned) == 0:
            return {
                "sku": req.sku,
                "model": "no_forecast",
                "forecast": [0.0] * req.periods,
                "metrics": {
                    "mae": 0.0,
                    "wape": 0.0,
                    "mase": 0.0,
                    "bias": 0.0
                },
                "analysis": {
                    "demand_pattern": "all_zero"
                },
                "model_ranking": []
            }

        pattern = demand_pattern(cleaned)

        if len(cleaned) < 2:
            final_forecast = naive_forecast(cleaned, req.periods)
            return {
                "sku": req.sku,
                "model": "naive",
                "forecast": final_forecast,
                "metrics": {
                    "mae": 0.0,
                    "wape": 0.0,
                    "mase": 0.0,
                    "bias": 0.0
                },
                "analysis": {
                    "demand_pattern": pattern
                },
                "model_ranking": []
            }

        test_size = min(3, max(1, len(cleaned) // 4))
        train = cleaned[:-test_size]
        test = cleaned[-test_size:]

        if len(train) == 0:
            train = cleaned[:-1]
            test = cleaned[-1:]

        if req.model == "auto":
            selection = choose_best_model(train, test, pattern)
            selected_model = selection["best"]["model"]
            backtest_metrics = selection["best"]["metrics"]
            ranking = selection["ranking"]
        else:
            if not model_is_allowed(req.model, train, pattern):
                raise ValueError(f"Modell '{req.model}' ist für diese Zeitreihe nicht zulässig.")
            selected_model = req.model
            evaluation = evaluate_model(train, test, selected_model)
            backtest_metrics = evaluation["metrics"]
            ranking = [evaluation]

        final_forecast = run_model(selected_model, cleaned, req.periods)

        return {
            "sku": req.sku,
            "model": selected_model,
            "forecast": [round(x, 6) for x in final_forecast],
            "metrics": backtest_metrics,
            "analysis": {
                "demand_pattern": pattern
            },
            "model_ranking": [
                {
                    "model": item["model"],
                    "mae": item["metrics"]["mae"],
                    "wape": item["metrics"]["wape"],
                    "mase": item["metrics"]["mase"],
                    "bias": item["metrics"]["bias"]
                }
                for item in ranking
            ]
        }

    except Exception as e:
        return {
            "sku": req.sku,
            "status": "failed",
            "reason": "forecast_error",
            "details": str(e)
        }
