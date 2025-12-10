import joblib
from pathlib import Path
import numpy as np


def load_model(model_path: str = "models/model.joblib"):
    """
    Load the trained ML pipeline from disk.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(path)


def predict(sample):
    """
    Takes a single sample (list or array) and returns prediction.
    The model expects shape (1, n_features), so wrap the sample.
    """
    model = load_model()
    sample_array = np.array(sample).reshape(1, -1)
    return model.predict(sample_array)[0]


if __name__ == "__main__":
    # Example usage:
    example = [10, 12, 5, 3, 7, 8, 2, 1, 0, 5, 3, 2, 1, 7, 9, 4, 6, 8, 2, 1, 3, 5, 7, 9, 4, 6, 8, 2, 1, 0]

    try:
        pred = predict(example)
        print("Prediction:", pred)
    except Exception as e:
        print("Error while predicting:", e)
