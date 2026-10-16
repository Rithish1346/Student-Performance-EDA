"""
Student Performance EDA and Deep Learning Case Study
Dataset: UCI Student Performance (Mathematics subject)
Target: G3 (final grade, 0-20)
This script downloads the dataset using ucimlrepo, performs EDA and preprocessing,
trains an MLP regression model, and saves charts/results.
"""
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
PLOTS = ROOT / "visualizations"
OUT.mkdir(exist_ok=True)
PLOTS.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid")


def savefig(name):
    plt.tight_layout()
    plt.savefig(PLOTS / name, dpi=160, bbox_inches="tight")
    plt.close()


def main():
    print("Downloading UCI Student Performance dataset...")
    dataset = fetch_ucirepo(id=320)
    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()
    df = pd.concat([X, y], axis=1)

    # Use the Mathematics dataset and final grade G3 as the target.
    # G1 and G2 are excluded from predictors to avoid relying on earlier-period grades.
    if "G3" not in df.columns:
        raise ValueError("Expected target column G3 was not found in the downloaded dataset.")
    target = "G3"
    excluded = [target, "G1", "G2"]
    feature_cols = [c for c in df.columns if c not in excluded]
    X = df[feature_cols].copy()
    y = pd.to_numeric(df[target], errors="coerce")

    print("\nDataset shape:", df.shape)
    print("\nFirst five rows:\n", df.head())
    print("\nData types:\n", df.dtypes)
    print("\nMissing values by column:\n", df.isnull().sum())
    print("\nDuplicate rows:", int(df.duplicated().sum()))
    print("\nDescriptive statistics:\n", df.describe(include="all").transpose())

    # Save a local copy of the combined data for reproducibility.
    df.to_csv(OUT / "student_performance_dataset.csv", index=False)
    df.describe(include="all").transpose().to_csv(OUT / "descriptive_statistics.csv")

    # Clean duplicate rows and align the target.
    clean = df.drop_duplicates().copy()
    y = pd.to_numeric(clean[target], errors="coerce")
    X = clean[[c for c in clean.columns if c not in excluded]].copy()
    valid = y.notna()
    X, y = X.loc[valid], y.loc[valid]

    # EDA visualizations (minimum 5).
    plt.figure(figsize=(8, 5))
    sns.histplot(clean[target], bins=15, kde=True)
    plt.title("Distribution of Final Grades (G3)")
    plt.xlabel("Final grade (0-20)")
    plt.ylabel("Number of students")
    savefig("01_final_grade_histogram.png")

    numeric_all = clean.select_dtypes(include=np.number)
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=clean[target])
    plt.title("Boxplot of Final Grades (G3)")
    plt.xlabel("Final grade (0-20)")
    savefig("02_final_grade_boxplot.png")

    plt.figure(figsize=(10, 8))
    corr = numeric_all.corr(numeric_only=True)
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
    plt.title("Correlation Heatmap of Numerical Features")
    savefig("03_correlation_heatmap.png")

    if "studytime" in clean.columns:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=clean, x="studytime", y=target)
        plt.title("Final Grade by Weekly Study-Time Category")
        plt.xlabel("Study-time category (1-4)")
        plt.ylabel("Final grade (0-20)")
        savefig("04_grade_by_studytime.png")

    if "absences" in clean.columns:
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=clean, x="absences", y=target, alpha=0.75)
        plt.title("School Absences vs Final Grade")
        plt.xlabel("Number of absences")
        plt.ylabel("Final grade (0-20)")
        savefig("05_absences_vs_grade.png")

    # Additional univariate/bivariate summaries.
    summary = {
        "rows_after_duplicate_removal": int(len(clean)),
        "columns_in_dataset": int(clean.shape[1]),
        "predictor_features_used": int(X.shape[1]),
        "target": target,
        "target_mean": float(y.mean()),
        "target_median": float(y.median()),
        "target_std": float(y.std()),
        "target_min": float(y.min()),
        "target_max": float(y.max()),
        "missing_values_total_before_cleaning": int(df.isnull().sum().sum()),
        "duplicate_rows_before_cleaning": int(df.duplicated().sum()),
        "note": "G1 and G2 are excluded from model predictors to avoid using earlier-period grades."
    }

    # Train / validation / test split: 70% / 15% / 15%.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED
    )

    categorical = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numerical = [c for c in X.columns if c not in categorical]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numerical),
        ("cat", categorical_pipeline, categorical)
    ])

    X_train_p = preprocessor.fit_transform(X_train).astype("float32")
    X_val_p = preprocessor.transform(X_val).astype("float32")
    X_test_p = preprocessor.transform(X_test).astype("float32")

    model = keras.Sequential([
        keras.layers.Input(shape=(X_train_p.shape[1],)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.20),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1, activation="linear")
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=20, restore_best_weights=True
    )
    history = model.fit(
        X_train_p, y_train.to_numpy(),
        validation_data=(X_val_p, y_val.to_numpy()),
        epochs=200, batch_size=32, verbose=0,
        callbacks=[early_stop]
    )

    predictions = model.predict(X_test_p, verbose=0).reshape(-1)
    mae = mean_absolute_error(y_test, predictions)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    r2 = r2_score(y_test, predictions)
    summary.update({
        "test_MAE": float(mae),
        "test_RMSE": rmse,
        "test_R2": float(r2),
        "epochs_trained": int(len(history.history["loss"])),
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "test_rows": int(len(X_test))
    })

    # Model evaluation visualizations.
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training loss")
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.title("Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Mean squared error")
    plt.legend()
    savefig("06_loss_vs_epoch.png")

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["mae"], label="Training MAE")
    plt.plot(history.history["val_mae"], label="Validation MAE")
    plt.title("MAE vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Mean absolute error")
    plt.legend()
    savefig("07_mae_vs_epoch.png")

    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, predictions, alpha=0.8)
    low = min(float(np.min(y_test)), float(np.min(predictions)))
    high = max(float(np.max(y_test)), float(np.max(predictions)))
    plt.plot([low, high], [low, high], linestyle="--")
    plt.title("Actual vs Predicted Final Grades")
    plt.xlabel("Actual G3")
    plt.ylabel("Predicted G3")
    savefig("08_actual_vs_predicted.png")

    residuals = y_test.to_numpy() - predictions
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, bins=15, kde=True)
    plt.title("Distribution of Prediction Errors")
    plt.xlabel("Actual grade - predicted grade")
    plt.ylabel("Frequency")
    savefig("09_error_distribution.png")

    model.save(OUT / "student_grade_mlp.keras")
    with open(OUT / "results_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame({
        "actual_G3": y_test.to_numpy(),
        "predicted_G3": predictions,
        "residual": residuals
    }).to_csv(OUT / "test_predictions.csv", index=False)

    print("\nMODEL RESULTS")
    print(f"MAE : {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²  : {r2:.3f}")
    print(f"Epochs trained: {len(history.history['loss'])}")
    print(f"\nCharts saved in: {PLOTS}")
    print(f"Outputs saved in: {OUT}")
    print("\nImportant: interpret the charts using the actual output files; do not copy invented findings.")


if __name__ == "__main__":
    main()
