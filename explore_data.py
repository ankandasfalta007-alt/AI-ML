"""Task 1: Load and explore the Student Performance dataset."""
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "student_scores.csv"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the dataset from CSV."""
    return pd.read_csv(path)


def explore(df: pd.DataFrame) -> None:
    """Print the structure and basic statistics of the data."""
    print("First 5 rows:")
    print(df.head())

    print("\nShape (rows, columns):", df.shape)

    print("\nColumn info:")
    df.info()

    print("\nMissing values per column:")
    print(df.isnull().sum())

    print("\nData statistics:")
    print(df.describe())

    print("\nCorrelation between Hours and Score:")
    print(df.corr().round(3))


if __name__ == "__main__":
    explore(load_data())
