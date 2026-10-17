"""Train/test split utilities."""
from sklearn.model_selection import train_test_split
from src.config import FEATURES, TARGET, TEST_SIZE, RANDOM_STATE


def split(df):
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )