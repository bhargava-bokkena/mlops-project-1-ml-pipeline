from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def build_pipeline(max_iter: int = 1000):
    """
    Build and return a sklearn Pipeline that:
    1. Scales features
    2. Trains a logistic regression classifier
    """
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=max_iter)),
        ]
    )
    return pipe
