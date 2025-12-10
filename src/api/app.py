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


_model = None  # cache for the loaded model


def load_model(model_path: Path = MODEL_PATH):
    """
    Load the trained ML pipeline from disk.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path.resolve()}")
    return joblib.load(model_path)


def get_model():
    """
    Lazy-load the model the first time it's needed, then cache it.
    """
    global _model
    if _model is None:
        _model = load_model()
    return _model


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Accepts a JSON body with a 'features' list and returns the prediction.
    """
    features = np.array(request.features).reshape(1, -1)
    model = get_model()
    pred = model.predict(features)[0]
    return PredictResponse(prediction=int(pred))
