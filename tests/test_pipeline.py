"""
Unit and smoke tests for the Customer Happiness Prediction pipeline.
Run with: pytest tests/
"""
import os
import pytest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.config import DATA_PATH, FEATURES, TARGET
from src.data_loader import load_data
from src.preprocessing import split


# Skip all tests in this file if the data isn't present (e.g., running in a fresh CI/CD env)
pytestmark = pytest.mark.skipif(
    not os.path.exists(DATA_PATH),
    reason=f"Data file not found at {DATA_PATH}. Skipping pipeline tests."
)


# =====================================================================
# 1. Data Loading & Validation Tests
# =====================================================================

def test_data_loads_and_has_expected_columns():
    """Ensure the CSV loads and contains all required columns."""
    df = load_data()
    
    assert isinstance(df, pd.DataFrame), "load_data should return a pandas DataFrame"
    assert not df.empty, "DataFrame should not be empty"
    
    required_cols = set(FEATURES + [TARGET])
    assert required_cols.issubset(df.columns), f"Missing required columns. Expected: {required_cols}"


def test_target_is_binary():
    """Ensure the target variable contains only 0 and 1."""
    df = load_data()
    
    unique_targets = set(df[TARGET].unique())
    assert unique_targets.issubset({0, 1}), f"Target contains non-binary values: {unique_targets}"


def test_features_are_within_valid_range():
    """Ensure survey responses are strictly between 1 and 5."""
    df = load_data()
    
    for feature in FEATURES:
        min_val = df[feature].min()
        max_val = df[feature].max()
        
        assert min_val >= 1, f"{feature} has values below 1 (found {min_val})"
        assert max_val <= 5, f"{feature} has values above 5 (found {max_val})"


def test_no_missing_values_in_features():
    """Ensure the data loader dropped or handled missing values properly."""
    df = load_data()
    
    assert df[FEATURES + [TARGET]].isnull().sum().sum() == 0, "Data still contains missing values"


# =====================================================================
# 2. Preprocessing Tests
# =====================================================================

def test_train_test_split_shapes():
    """Ensure the split returns the expected train/test proportions."""
    df = load_data()
    X_train, X_test, y_train, y_test = split(df)
    
    # 80/20 split expected
    total_samples = len(df)
    assert len(X_train) + len(X_test) == total_samples, "Split does not cover the entire dataset"
    assert len(X_train) == len(y_train), "Train features and labels length mismatch"
    assert len(X_test) == len(y_test), "Test features and labels length mismatch"
    
    # Check that test set is roughly 20% (allow small rounding variance)
    expected_test_size = total_samples * 0.2
    assert abs(len(X_test) - expected_test_size) <= 2, "Test split size is significantly off from 20%"


# =====================================================================
# 3. Model Smoke Test
# =====================================================================

def test_random_forest_can_train_and_predict():
    """Ensure a model can fit on the training data and predict on the test data."""
    df = load_data()
    X_train, X_test, y_train, y_test = split(df)
    
    # Use a small, fast model for testing
    clf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    
    # Should not raise an exception
    clf.fit(X_train, y_train)
    
    # Check predictions shape
    predictions = clf.predict(X_test)
    assert len(predictions) == len(X_test), "Number of predictions does not match test set size"
    
    # Check if model outputs probabilities for both classes
    probabilities = clf.predict_proba(X_test)
    assert probabilities.shape == (len(X_test), 2), "predict_proba output shape is incorrect"