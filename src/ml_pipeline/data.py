from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def get_data():
    """
    Load the dataset and return features (X) and labels (y).
    For now, we use a built-in sklearn dataset so we don't deal with files yet.
    """
    data = load_breast_cancer()
    X = data.data
    y = data.target
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into train and test sets.
    Keeping default test_size and random_state makes runs reproducible.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
