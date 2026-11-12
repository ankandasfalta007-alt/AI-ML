# Student Performance Predictor

Predict a student's exam score from the number of hours they studied, using
Python, pandas, scikit-learn and matplotlib.

![Hours vs Score](outputs/hours_vs_score.png)

## Project structure

```
student-performance-predictor/
├── data/student_scores.csv     # Hours Studied vs Exam Score (10 rows)
├── src/
│   ├── explore_data.py         # Task 1: load and explore the data
│   └── predict_scores.py       # Train a linear regression model and plot it
├── outputs/hours_vs_score.png  # Generated plot
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone <your-repo-url>
cd student-performance-predictor
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

**Task 1 - Load and explore the data**

```bash
python src/explore_data.py
```

Prints the first rows, shape, column types, missing values, summary statistics
(average hours = 5.5, average score ≈ 64.8) and the correlation between the columns.

**Train the model and make a prediction**

```bash
python src/predict_scores.py
```

Fits a linear regression on an 80/20 train/test split, prints the slope,
intercept, R² and mean absolute error, predicts the score for 7.5 hours of
study, and saves a plot to `outputs/`.

## Results

| Metric | Value |
|---|---|
| Slope | ~6.78 points per extra hour |
| Intercept | ~27.06 |
| R² (test set) | ~0.98 |
| Mean absolute error | ~2.25 points |

The dataset is tiny (10 rows), so these numbers show the workflow rather than
real-world accuracy.

## Running in Google Colab

Upload `data/student_scores.csv` and the two files in `src/`, then run
`!pip install pandas scikit-learn matplotlib` followed by the scripts.

## Next ideas

- Use a larger, real dataset
- Add more features (sleep hours, attendance)
- Compare with other models (polynomial regression, random forest)
