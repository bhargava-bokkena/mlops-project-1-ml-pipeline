from pathlib import Path
from typing import List

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel


MODEL_PATH = Path("models/model.joblib")

app = FastAPI(title="Week1 ML Pipeline API")


class PredictRequest(BaseModel):
    features: List[float]


class PredictResponse(BaseModel):
    prediction: int


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH.resolve()}")
    model = joblib.load(MODEL_PATH)
    return model


# Load model once at startup
model = load_model()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Accepts a JSON body with a 'features' list and returns the prediction.
    """
    features = np.array(request.features).reshape(1, -1)
    pred = model.predict(features)[0]
    return PredictResponse(prediction=int(pred))
