"""Load and validate the survey dataset."""
import pandas as pd
from src.config import DATA_PATH, FEATURES, TARGET


def load_data(path=DATA_PATH) -> pd.DataFrame:
    """Load survey CSV and run basic validation."""
    df = pd.read_csv(path)

    required = set(FEATURES + [TARGET])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
    return df