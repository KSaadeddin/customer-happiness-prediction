"""Central configuration for the pipeline."""
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "raw" / "survey_data.csv"
MODEL_DIR = ROOT_DIR / "models"
FIG_DIR = ROOT_DIR / "reports" / "figures"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Features & target
TARGET = "Y"
FEATURES = ["X1", "X2", "X3", "X4", "X5", "X6"]
FEATURE_LABELS = {
    "X1": "Order delivered on time",
    "X2": "Contents as expected",
    "X3": "Ordered everything wanted",
    "X4": "Good price",
    "X5": "Satisfied with courier",
    "X6": "App easy to use",
}

# Reproducibility
RANDOM_STATE = 42
TEST_SIZE = 0.2