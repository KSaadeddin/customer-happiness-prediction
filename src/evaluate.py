"""Evaluate the trained model and generate figures."""
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, RocCurveDisplay,
)

from src.config import FEATURES, FEATURE_LABELS, MODEL_DIR, FIG_DIR


def plot_feature_importance(model, feature_names):
    imp = pd.DataFrame({
        "feature": [FEATURE_LABELS[f] for f in feature_names],
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=True)

    plt.figure(figsize=(8, 4))
    sns.barplot(data=imp, x="importance", y="feature", palette="viridis")
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "feature_importance.png", dpi=150)
    plt.close()


def plot_confusion(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Unhappy", "Happy"],
                yticklabels=["Unhappy", "Happy"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "confusion_matrix.png", dpi=150)
    plt.close()


def main():
    model = joblib.load(MODEL_DIR / "rf_model.pkl")
    X_test = pd.read_csv(MODEL_DIR / "X_test.csv")
    y_test = pd.read_csv(MODEL_DIR / "y_test.csv").squeeze()

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC:       {roc_auc_score(y_test, y_proba):.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["Unhappy", "Happy"]))

    plot_feature_importance(model, FEATURES)
    plot_confusion(y_test, y_pred)

    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "roc_curve.png", dpi=150)
    plt.close()

    print(f"\nFigures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()