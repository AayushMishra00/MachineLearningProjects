# Diabetes Prediction — Logistic Regression

A binary classification project predicting diabetes from clinical data, built as part of my ML learning journey.

## Problem
Predict whether a patient has diabetes based on 8 diagnostic features from the Pima Indians Diabetes Dataset (768 patients).

## Dataset
- Source: [Pima Indians Diabetes Dataset](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv)
- 768 rows, 8 features, 1 binary target (Outcome)
- Features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigree, Age

## What I did

### Data Cleaning
- Identified disguised missing values — columns like Glucose, BMI and Insulin contained zeros that are biologically impossible
- Replaced zeros with NaN and imputed using column medians
- Insulin had ~49% missing values, making it the noisiest feature

### Modelling
Compared 4 models on the same data:

| Model | Accuracy |
|---|---|
| Logistic Regression | 0.75 |
| Neural Network (MLP) | 0.72 |
| Random Forest | 0.73 |
| XGBoost | 0.71 |

Logistic Regression won — consistent with the dataset being small (768 rows) and the relationships being largely linear.

### Evaluation & Threshold Tuning
Default threshold (0.5) missed 38% of diabetic patients — unacceptable for a medical screening tool.

Lowered threshold to 0.4 to prioritise catching diabetic cases:

| Metric | Threshold 0.5 | Threshold 0.4 |
|---|---|---|
| Diabetes recall | 0.62 | 0.71 |
| Diabetes precision | 0.67 | 0.59 |
| Accuracy | 0.75 | 0.72 |
| AUC-ROC | — | 0.82 |

Trading precision for recall is the right call in a medical context — missing a sick patient is worse than a false alarm.

## Results
- AUC-ROC: **0.82** (top end of published results on this dataset)
- Diabetes recall: **0.71** after threshold tuning
- Correctly identifies 71% of diabetic patients at a 0.4 threshold

## Key Learnings
- Accuracy is a misleading metric on imbalanced datasets
- Neural networks don't always win — simpler models often outperform on small tabular data
- Threshold tuning is a domain decision, not a math decision
- Always fit the scaler on training data only — fitting on the full dataset leaks information

## Files
- `diabetes_classification.ipynb` — full notebook with cleaning, modelling, evaluation

## Tech Stack
Python, pandas, scikit-learn, XGBoost, matplotlib, seaborn

## Next Project
CNN image classifier on MNIST → moving toward computer vision
