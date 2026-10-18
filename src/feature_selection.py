"""RFECV to find the minimal informative survey."""
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV

from src.config import FEATURES, FEATURE_LABELS, MODEL_DIR, RANDOM_STATE
from src.data_loader import load_data
from src.preprocessing import split


def main():
    df = load_data()
    X_train, X_test, y_train, y_test = split(df)

    rfecv = RFECV(
        estimator=RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        step=1, cv=5, scoring="accuracy",
        min_features_to_select=1, n_jobs=-1,
    )
    rfecv.fit(X_train, y_train)

    selected = [f for f, s in zip(FEATURES, rfecv.support_) if s]
    ranked = dict(zip(FEATURES, rfecv.ranking_))

    print(f"Optimal #features: {rfecv.n_features_}")
    print(f"Selected: {[FEATURE_LABELS[f] for f in selected]}")
    print(f"Ranking (1 = best): {ranked}")

    joblib.dump(selected, MODEL_DIR / "selected_features.pkl")


if __name__ == "__main__":
    main()