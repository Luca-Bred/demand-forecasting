from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Literal, Dict, Any

app = FastAPI(title="Demand Forecast API", version="4.0.0")


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


def detect_demand_pattern(cleaned: List[float]) -> str:
    if not cleaned:
        return "no_demand"

    non_zero = [x for x in cleaned if x > 0]
    if not non_zero:
        return "all_zero"

    zero_share = sum(1 for x in cleaned if x == 0) / len(cleaned)
    adi = len(cleaned) / len(non_zero) if len(non_zero) > 0 else float("inf")

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


def ets_forecast(train: List[float], periods: int, alpha: float = 0.3) -> List[float]:
    """
    Einfache Exponentialglättung (SES).
    """
    if not train:
        return []
    level = float(train[0])
    for x in train[1:]:
        level = alpha * float(x) + (1 - alpha) * level
    return [level] * periods


def croston_forecast(train: List[float], periods: int, alpha: float = 0.1) -> List[float]:
    """
    Einfache Croston-Variante für intermittierende Nachfrage.
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


def run_model(model_name: str, train: List[float], periods: int) -> List[float]:
    if model_name == "naive":
        return naive_forecast(train, periods)
    if model_name == "moving_average":
        return moving_average_forecast(train, periods, window=3)
    if model_name == "ets":
        return ets_forecast(train, periods, alpha=0.3)
    if model_name == "croston":
        return croston_forecast(train, periods, alpha=0.1)
    raise ValueError(f"Unbekanntes Modell: {model_name}")


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
    if not naive_errors:
        return 0.0

    scale = sum(naive_errors) / len(naive_errors)
    if scale == 0:
        return 0.0

    return mae(actual, predicted) / scale


def model_is_allowed(model_name: str, train: List[float], pattern: str) -> bool:
    if model_name == "naive":
        return len(train) >= 1
    if model_name == "moving_average":
        return len(train) >= 2
    if model_name == "ets":
        return len(train) >= 3
    if model_name == "croston":
        return pattern == "intermittent" and count_nonzero(train) >= 2
    return False


def generate_rolling_splits(series: List[float], n_splits: int = 3, horizon: int = 1) -> List[tuple]:
    """
    Rolling Forecast Origin:
    Beispiel bei horizon=1:
    train[:t1] -> test[t1]
    train[:t2] -> test[t2]
    train[:t3] -> test[t3]
    """
    splits = []
    n = len(series)

    # letzter möglicher Teststart
    last_test_start = n - horizon
    if last_test_start <= 1:
        return splits

    possible_origins = list(range(2, last_test_start + 1))
    selected_origins = possible_origins[-n_splits:]

    for origin in selected_origins:
        train = series[:origin]
        test = series[origin:origin + horizon]
        if len(train) >= 1 and len(test) == horizon:
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
        split_results.append({
            "split": idx,
            "mae": mae(test, preds),
            "wape": wape(test, preds),
            "mase": mase(test, preds, train),
            "bias": bias(test, preds),
        })

    avg_mae = sum(x["mae"] for x in split_results) / len(split_results)
    avg_wape = sum(x["wape"] for x in split_results) / len(split_results)
    avg_mase = sum(x["mase"] for x in split_results) / len(split_results)
    avg_bias = sum(x["bias"] for x in split_results) / len(split_results)

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
            "horizon": horizon
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
        ]
    }


def choose_best_model(series: List[float], pattern: str) -> Dict[str, Any]:
    candidate_models = ["naive", "moving_average", "ets", "croston"]
    valid_models = [m for m in candidate_models if model_is_allowed(m, series[:-1] if len(series) > 1 else series, pattern)]

    if not valid_models:
        raise ValueError("Keine zulässigen Modelle verfügbar.")

    results = []
    for model_name in valid_models:
        try:
            result = evaluate_model_rolling(series, model_name, pattern, n_splits=3, horizon=1)
            results.append(result)
        except Exception:
            continue

    if not results:
        raise ValueError("Keine Modelle konnten erfolgreich backgetestet werden.")

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
                "model_ranking": [],
                "backtest_config": {
                    "splits": 0,
                    "horizon": 0
                }
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
                "model_ranking": [],
                "backtest_config": {
                    "splits": 0,
                    "horizon": 0
                }
            }

        pattern = detect_demand_pattern(cleaned)

        if len(cleaned) < 4:
            selected_model = "naive"
            final_forecast = naive_forecast(cleaned, req.periods)
            return {
                "sku": req.sku,
                "model": selected_model,
                "forecast": [round(x, 6) for x in final_forecast],
                "metrics": {
                    "mae": 0.0,
                    "wape": 0.0,
                    "mase": 0.0,
                    "bias": 0.0
                },
                "analysis": {
                    "demand_pattern": pattern
                },
                "model_ranking": [],
                "backtest_config": {
                    "splits": 0,
                    "horizon": 0
                }
            }

        if req.model == "auto":
            selection = choose_best_model(cleaned, pattern)
            selected_model = selection["best"]["model"]
            backtest_metrics = selection["best"]["metrics"]
            backtest_config = selection["best"]["backtest_config"]
            ranking = selection["ranking"]
        else:
            if not model_is_allowed(req.model, cleaned[:-1] if len(cleaned) > 1 else cleaned, pattern):
                raise ValueError(f"Modell '{req.model}' ist für diese Zeitreihe nicht zulässig.")

            evaluation = evaluate_model_rolling(cleaned, req.model, pattern, n_splits=3, horizon=1)
            selected_model = req.model
            backtest_metrics = evaluation["metrics"]
            backtest_config = evaluation["backtest_config"]
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
            "backtest_config": backtest_config,
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
