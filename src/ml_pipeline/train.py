from pathlib import Path
import json
import logging
from datetime import datetime

import joblib
import yaml
from sklearn.metrics import accuracy_score

from .data import get_data, split_data
from .pipeline import build_pipeline


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """
    Load YAML config and return as a plain dict.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(logs_dir: str = "logs") -> logging.Logger:
    """
    Configure logging to write both to console and to logs/training.log.
    """
    logs_path = Path(logs_dir)
    logs_path.mkdir(parents=True, exist_ok=True)
    log_file = logs_path / "training.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
    )

    logger = logging.getLogger(__name__)
    logger.info("Logging initialized. Log file: %s", log_file.resolve())
    return logger


def save_metrics(metrics: dict, path: str = "logs/metrics.json") -> Path:
    """
    Save metrics as a JSON file so other tools (CI, monitoring, etc.)
    can read them programmatically.
    """
    metrics_path = Path(path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics_path


def main():
    # 0. Setup logging and load config
    logger = setup_logging()
    config = load_config()
    logger.info("Loaded config: %s", config)

    # 1. Load data
    X, y = get_data()
    logger.info("Loaded data with %d samples.", len(y))

    # 2. Split into train/test using config
    test_size = config["data"]["test_size"]
    random_state = config["data"]["random_state"]
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info(
        "Split data into %d train and %d test samples.",
        len(y_train),
        len(y_test),
    )

    # 3. Build pipeline (scaler + model), using config for hyperparams
    max_iter = config["model"]["max_iter"]
    pipeline = build_pipeline(max_iter=max_iter)
    logger.info("Built pipeline with LogisticRegression(max_iter=%d).", max_iter)

    # 4. Train
    pipeline.fit(X_train, y_train)
    logger.info("Finished training.")

    # 5. Evaluate
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info("Test accuracy: %.4f", acc)

    # 6. Prepare and save metrics
    metrics = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "accuracy": acc,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "test_size": test_size,
        "random_state": random_state,
        "model": {
            "type": "logistic_regression",
            "max_iter": max_iter,
        },
    }

    metrics_path = save_metrics(metrics)
    logger.info("Saved metrics to %s", metrics_path.resolve())

    # 7. Save model artifact
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "model.joblib"
    joblib.dump(pipeline, model_path)
    logger.info("Saved trained model to: %s", model_path.resolve())

    # Still print accuracy for human convenience
    print(f"Test accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
