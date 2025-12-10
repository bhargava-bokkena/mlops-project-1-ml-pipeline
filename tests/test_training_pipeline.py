from src.ml_pipeline.data import get_data, split_data
from src.ml_pipeline.pipeline import build_pipeline
from sklearn.metrics import accuracy_score


def test_training_pipeline_runs_and_has_reasonable_accuracy():
    X, y = get_data()
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=0.2, random_state=42
    )

    # Use a smaller max_iter for faster tests
    pipeline = build_pipeline(max_iter=200)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Sanity check: model should be better than random
    assert acc > 0.8
