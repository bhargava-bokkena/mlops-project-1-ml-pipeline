import pytest
from src.ml_pipeline.train import main as train_main


@pytest.fixture(scope="session", autouse=True)
def train_model_artifact():
    """
    Ensure the trained model artifact exists before tests that need it.
    Runs once per test session.
    """
    train_main()
    yield
