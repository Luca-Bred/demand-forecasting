from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

class Request(BaseModel):
    sku: str
    demand: List[float]
    stockout: List[int]
    periods: int = 3

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/forecast")
def forecast(req: Request):
    avg = sum(req.demand) / len(req.demand)
    return {
        "forecast": [avg] * req.periods,
        "metrics": {
            "mae": 0,
            "wape": 0,
            "mase": 0,
            "bias": 0
        }
    }
