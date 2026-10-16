# Student Performance Analysis using EDA and Deep Learning

## Project overview
This case study explores a real-world student performance dataset, visualizes distributions and relationships, and trains a Multilayer Perceptron (MLP) regression model to estimate the final grade (`G3`).

## Dataset
- Source: UCI Machine Learning Repository — Student Performance
- Official page: https://archive.ics.uci.edu/dataset/320/student%2Bperformance
- The repository reports 649 instances and 33 features/fields for this dataset collection.
- The code downloads the data with `ucimlrepo` (internet required on first run).
- The dataset concerns secondary-school students in Portugal; do not describe it as data from your own college.

## Prediction setup
The target is final grade `G3` (0–20). The earlier-period grades `G1` and `G2` are intentionally excluded from the model predictors, because including them can make prediction easier while providing less useful early-warning insight. This is a modeling choice, not a requirement from the dataset publisher.

## Requirements
Python 3.10 or later is recommended.

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Run:
```bash
python src/main.py
```

## Outputs
- `visualizations/`: at least five EDA charts and model evaluation charts.
- `outputs/student_performance_dataset.csv`: downloaded dataset copy.
- `outputs/descriptive_statistics.csv`: descriptive statistics.
- `outputs/results_summary.json`: summary and test metrics.
- `outputs/test_predictions.csv`: actual values, predictions, and residuals.
- `outputs/student_grade_mlp.keras`: trained model.

The `outputs/` and `visualizations/` folders will be populated after running the script.

## Suggested report structure
1. Title Page
2. Abstract (100–150 words)
3. Introduction and Objectives
4. Dataset Description
5. EDA and Preprocessing
6. Data Visualization (at least five charts with interpretation)
7. Deep Learning Model (architecture, parameters, hyperparameters)
8. Result Visualization and Interpretation
9. Conclusion and Future Scope
10. References
11. Appendix: Source Code

## Important academic note
Write the final findings and conclusions only after running the program and inspecting its actual outputs. Do not invent model scores or graph interpretations. The dataset has no missing values according to the UCI dataset page, but the code still checks for missing values and uses imputers in the preprocessing pipeline.
