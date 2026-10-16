# Student Performance Analysis using Exploratory Data Analysis and Deep Learning

**Student Name:** [Your Name]  
**Register Number:** [Your Register Number]  
**Department:** Artificial Intelligence and Data Science  
**Course:** Exploratory Data Analysis and Visualization (U21ADP05)  
**Academic Year:** 2026–2027  
**Submission Date:** [Enter date]

## Abstract (100–150 words)
This case study explores the UCI Student Performance dataset using exploratory data analysis, statistical summaries, data visualization, and a Deep Learning model. The dataset contains student-related demographic, family, school, and lifestyle information, along with grade information. The analysis examines the distribution of final grades, potential outliers, correlations among numerical variables, and relationships between study-time categories, absences, and final grades. After cleaning and preprocessing, categorical variables are encoded and numerical variables are standardized. A Multilayer Perceptron regression model is trained to estimate the final grade (`G3`), while earlier-period grades (`G1` and `G2`) are excluded from the predictors. Model performance is evaluated using MAE, RMSE, R², and suitable visualizations. **Replace or refine this abstract after running the code so it accurately reflects your results.**

## 1. Introduction and Objectives
Student performance can be associated with academic, demographic, family, and lifestyle factors. Exploratory Data Analysis (EDA) helps summarize a dataset, identify patterns, and form questions for further investigation.

### Objectives
- Explore a real-world dataset with more than 15 fields.
- Inspect data types, missing values, duplicates, and statistical summaries.
- Create at least five meaningful visualizations.
- Apply a Deep Learning model suited to tabular data.
- Evaluate model performance and interpret the results.

## 2. Dataset Description
The dataset is the UCI Machine Learning Repository's Student Performance dataset. It concerns student achievement in secondary education at two Portuguese schools. The dataset includes demographic, social, school-related, and grade attributes. The repository reports 649 instances and 33 features/fields for the dataset collection.

**Source:** https://archive.ics.uci.edu/dataset/320/student%2Bperformance

The target used in this project is `G3`, the final grade on a 0–20 scale. The model excludes `G1` and `G2` from predictors to avoid depending on earlier-period grades.

## 3. EDA and Preprocessing
The Python program:
- Loads the dataset using `ucimlrepo`.
- Displays the first rows, shape, data types, and descriptive statistics.
- Checks missing values and duplicate rows.
- Removes duplicate rows if present.
- Uses median imputation and standardization for numerical predictors.
- Uses most-frequent imputation and one-hot encoding for categorical predictors.
- Splits the data into training, validation, and test sets (70%, 15%, 15%).

Record the actual missing-value and duplicate counts from the program output.

## 4. Data Visualization
Include the saved graphs from the `visualizations/` folder. For each one, write what it shows, why it was created, and the insight actually supported by the graph.

### 4.1 Final Grade Histogram
**What it shows:** Distribution of final grades (`G3`).  
**Why created:** To examine the frequency and shape of grade values.  
**Insight:** [Write based on your actual chart.]

### 4.2 Final Grade Boxplot
**What it shows:** Median, spread, and potential outliers in final grades.  
**Why created:** To inspect grade variability and unusual observations.  
**Insight:** [Write based on your actual chart.]

### 4.3 Correlation Heatmap
**What it shows:** Pairwise correlations among numerical variables.  
**Why created:** To identify variables with possible linear associations.  
**Insight:** [Write based on your actual heatmap. Correlation does not prove causation.]

### 4.4 Final Grade by Study-Time Category
**What it shows:** Final grade distributions across study-time categories.  
**Why created:** To examine whether grade distributions differ across categories.  
**Insight:** [Write based on your actual chart.]

### 4.5 Absences vs Final Grade
**What it shows:** Relationship between school absences and final grades.  
**Why created:** To inspect any visible association or unusual observations.  
**Insight:** [Write based on your actual chart.]

## 5. Deep Learning Model
The model is a Multilayer Perceptron (MLP) regression network:
- Input layer: number of preprocessed features.
- Hidden layer 1: 64 neurons, ReLU activation.
- Dropout: 0.20.
- Hidden layer 2: 32 neurons, ReLU activation.
- Output layer: 1 neuron with linear activation.
- Optimizer: Adam, learning rate 0.001.
- Loss: Mean Squared Error.
- Metric: Mean Absolute Error.
- Maximum epochs: 200.
- Batch size: 32.
- Early stopping: monitors validation loss with patience 20.

## 6. Model Evaluation and Interpretation
Insert and discuss the following charts:
- Loss vs Epoch.
- MAE vs Epoch.
- Actual vs Predicted Final Grades.
- Prediction Error Distribution.

Copy the actual MAE, RMSE, and R² from `outputs/results_summary.json`.

- **MAE:** [Actual value]
- **RMSE:** [Actual value]
- **R²:** [Actual value]

Explain these metrics in context. Lower MAE and RMSE indicate smaller prediction errors. R² describes the proportion of target variance explained by the model under the evaluation setup; it can be negative if predictions perform poorly relative to a simple baseline.

## 7. Conclusion and Future Scope
Summarize the findings supported by the actual visualizations and test metrics. Discuss limitations, including the dataset's context (secondary schools in Portugal), modest size, and the fact that predictive relationships do not establish cause and effect.

Possible future work includes comparing other models, tuning hyperparameters, validating on another dataset, and evaluating whether additional features improve generalization.

## 8. References
1. Cortez, P. (2008). *Student Performance* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T
2. Cortez, P., & Silva, A. M. G. (2008). Using data mining to predict secondary school student performance. *Proceedings of the 5th Annual Future Business Technology Conference*.
3. UCI Machine Learning Repository. (n.d.). *Student Performance*. https://archive.ics.uci.edu/dataset/320/student%2Bperformance
4. TensorFlow. (n.d.). *TensorFlow documentation*. https://www.tensorflow.org/learn
5. scikit-learn developers. (n.d.). *scikit-learn documentation*. https://scikit-learn.org/stable/

## 9. Appendix: Source Code
Include `src/main.py` or attach the full source code in the report appendix.
