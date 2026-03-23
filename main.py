from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Literal
import math

app = FastAPI(title="Demand Forecast API", version="2.0.0")


class ForecastRequest(BaseModel):
    sku: str
    demand: List[float]
    stockout: List[int]
    periods: int = 15
    model: Optional[Literal["auto", "naive", "moving_average"]] = "auto"


def validate_inputs(demand: List[float], stockout: List[int]) -> None:
    if len(demand) != len(stockout):
        raise ValueError("demand und stockout müssen gleich lang sein.")
    if len(demand) == 0:
        raise ValueError("demand darf nicht leer sein.")
    if any(s not in (0, 1) for s in stockout):
        raise ValueError("stockout darf nur 0 oder 1 enthalten.")


def clean_demand(demand: List[float], stockout: List[int]) -> List[float]:
    """
    Stockout=1 wird als fehlende Beobachtung behandelt und entfernt.
    """
    cleaned = [d for d, s in zip(demand, stockout) if s == 0]
    return cleaned


def naive_forecast(train: List[float], periods: int) -> List[float]:
    """
    Forecast = letzter beobachteter Wert.
    """
    if not train:
        return []
    last_value = train[-1]
    return [float(last_value)] * periods


def moving_average_forecast(train: List[float], periods: int, window: int = 3) -> List[float]:
    """
    Forecast = Durchschnitt der letzten n Werte.
    """
    if not train:
        return []
    effective_window = min(window, len(train))
    avg = sum(train[-effective_window:]) / effective_window
    return [float(avg)] * periods


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
    """
    MASE relativ zu Naive(1)-In-Sample-Fehler.
    """
    if len(train) < 2 or not actual:
        return 0.0

    naive_errors = [abs(train[i] - train[i - 1]) for i in range(1, len(train))]
    scale = sum(naive_errors) / len(naive_errors)

    if scale == 0:
        return 0.0

    model_mae = mae(actual, predicted)
    return model_mae / scale


def evaluate_model(train: List[float], test: List[float], model_name: str) -> dict:
    if model_name == "naive":
        preds = naive_forecast(train, len(test))
    elif model_name == "moving_average":
        preds = moving_average_forecast(train, len(test), window=3)
    else:
        raise ValueError(f"Unbekanntes Modell: {model_name}")

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


def choose_best_model(train: List[float], test: List[float]) -> dict:
    """
    Vergleich aktuell zwischen naive und moving_average.
    Auswahlregel:
    1. niedrigstes MAE
    2. niedrigstes WAPE
    3. Bias möglichst nahe 0
    """
    candidates = [
        evaluate_model(train, test, "naive"),
        evaluate_model(train, test, "moving_average"),
    ]

    candidates.sort(
        key=lambda x: (
            x["metrics"]["mae"],
            x["metrics"]["wape"],
            abs(x["metrics"]["bias"]),
        )
    )
    return candidates[0]


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
                    "bias": 0.0,
                },
            }

        if len(cleaned) < 2:
            final_forecast = [float(cleaned[-1])] * req.periods
            return {
                "sku": req.sku,
                "model": "naive",
                "forecast": final_forecast,
                "metrics": {
                    "mae": 0.0,
                    "wape": 0.0,
                    "mase": 0.0,
                    "bias": 0.0,
                },
            }

        # Einfacher Holdout für MVP:
        # letzte 3 Perioden oder mindestens 1 Beobachtung im Test
        test_size = min(3, max(1, len(cleaned) // 4))
        train = cleaned[:-test_size]
        test = cleaned[-test_size:]

        # Falls train leer wird, fallback
        if len(train) == 0:
            train = cleaned[:-1]
            test = cleaned[-1:]

        if req.model == "auto":
            best = choose_best_model(train, test)
            selected_model = best["model"]
            backtest_metrics = best["metrics"]
        else:
            selected_model = req.model
            best = evaluate_model(train, test, selected_model)
            backtest_metrics = best["metrics"]

        # Final Forecast auf voller bereinigter Historie
        if selected_model == "naive":
            final_forecast = naive_forecast(cleaned, req.periods)
        elif selected_model == "moving_average":
            final_forecast = moving_average_forecast(cleaned, req.periods, window=3)
        else:
            raise ValueError(f"Unbekanntes finales Modell: {selected_model}")

        return {
            "sku": req.sku,
            "model": selected_model,
            "forecast": final_forecast,
            "metrics": backtest_metrics,
        }

    except Exception as e:
        return {
            "sku": req.sku,
            "status": "failed",
            "reason": "forecast_error",
            "details": str(e),
        }
