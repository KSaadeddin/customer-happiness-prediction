"""Train and tune the Random Forest model."""
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from src.config import MODEL_DIR, RANDOM_STATE
from src.data_loader import load_data
from src.preprocessing import split

PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10, 15],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
    "class_weight": [None, "balanced"],
}


def tune_random_forest(X_train, y_train):
    grid = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        PARAM_GRID,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_, grid.best_score_


def main():
    df = load_data()
    X_train, X_test, y_train, y_test = split(df)

    model, best_params, cv_score = tune_random_forest(X_train, y_train)

    print("Best params:", best_params)
    print(f"Best CV accuracy: {cv_score:.4f}")

    # Persist model + test data for evaluation step
    joblib.dump(model, MODEL_DIR / "rf_model.pkl")
    X_test.to_csv(MODEL_DIR / "X_test.csv", index=False)
    y_test.to_csv(MODEL_DIR / "y_test.csv", index=False)
    print(f"Model saved to {MODEL_DIR / 'rf_model.pkl'}")


if __name__ == "__main__":
    main()