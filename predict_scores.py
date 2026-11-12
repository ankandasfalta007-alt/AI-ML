"""Train a linear regression model to predict exam score from hours studied."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # works without a display (servers, Colab, CI)
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from explore_data import load_data

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def main() -> None:
    df = load_data()
    X = df[["Hours"]]
    y = df["Score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print(f"Slope (points per extra hour): {model.coef_[0]:.2f}")
    print(f"Intercept: {model.intercept_:.2f}")
    print(f"R^2 on test set: {r2_score(y_test, predictions):.3f}")
    print(f"Mean absolute error: {mean_absolute_error(y_test, predictions):.2f}")

    # Example prediction
    hours = 7.5
    predicted = model.predict(pd.DataFrame({"Hours": [hours]}))[0]
    print(f"\nPredicted score for {hours} hours studied: {predicted:.1f}")

    # Plot data and fitted line
    OUTPUT_DIR.mkdir(exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, label="Actual data")
    plt.plot(X, model.predict(X), color="red", label="Linear regression fit")
    plt.xlabel("Hours Studied")
    plt.ylabel("Exam Score")
    plt.title("Student Performance Predictor")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out = OUTPUT_DIR / "hours_vs_score.png"
    plt.savefig(out, dpi=150)
    print(f"Plot saved to {out}")


if __name__ == "__main__":
    main()
