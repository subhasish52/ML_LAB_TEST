# Loan Default Prediction using XGBoost + Optuna

A machine learning project for predicting loan default probability using a peer-to-peer lending dataset.  
The goal is to build a high-accuracy, fair, and optimized model that helps the loan approval team at **FinSecure** evaluate borrower risk.

---

### 1. Project Overview

This project includes:

- Full preprocessing pipeline  
- Baseline XGBoost model  
- Hyperparameter-tuned XGBoost using Optuna  
- ROC curve visualization  
- Model evaluation using AUC  

The repository contains code for training, optimizing, and evaluating the final model.

---

### 2. Dataset Description

The dataset `loan_data.csv` contains borrower information and repayment outcomes.

#### Numerical Features
- annual_income  
- debt_to_income_ratio  
- credit_score  
- loan_amount  
- interest_rate  
- subgrade_num *(engineered)*  

#### Categorical Features
- gender  
- marital_status  
- education_level  
- employment_status  
- loan_purpose  
- grade *(engineered)*  

#### Target Variable
loan_paid_back
1 → Loan repaid
0 → Loan defaulted


#### Feature Engineering
`grade_subgrade` is split into:
- grade (A–G)  
- subgrade_num (1–5)

---

### 3. Tech Stack

- Python  
- XGBoost  
- Optuna  
- Scikit-learn  
- Matplotlib  

---

### 4. Model Pipeline

The pipeline includes:

1. Dropping non-useful columns  
2. One-Hot Encoding categorical variables  
3. StandardScaling numerical columns  
4. Train-test split (stratified)  
5. Unified preprocessing + modeling pipeline  

---

### 5. Baseline XGBoost Model

Baseline model:

```python
xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss"
)
```

AUC evaluation
roc_auc_score(y_test, preds)

ROC curve
RocCurveDisplay.from_estimator(pipeline_xgb, X_test, y_test)

---

### 6. Hyperparameter Optimization using Optuna

Optuna is used to find the best XGBoost parameters.

Search Space
n_estimators: 200–600
learning_rate: 0.01–0.1
max_depth: 3–10
subsample: 0.6–1.0
colsample_bytree: 0.6–1.0
gamma: 0–3
min_child_weight: 1–10

Execution
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=25)

Output

Best AUC

Best hyperparameters

---

### 7. Results
Primary Metric

AUC (Area Under ROC Curve)

Model Performance

Baseline XGBoost AUC: ~0.84–0.86

Optuna Tuned XGBoost AUC: ~0.87–0.90

Why AUC?

Works well for imbalanced data

Measures ranking quality

Ideal for financial risk scoring

---

### 8. ROC Curve

Final ROC curve:

RocCurveDisplay.from_estimator(final_pipeline, X_test, y_test)
plt.title("ROC Curve - Optuna Tuned XGBoost")

---

### 9. Folder Structure
|-- model.ipynb
|-- README.md

---

### 10. How to Run the Project
Step 1: Install Dependencies
pip install xgboost optuna scikit-learn pandas matplotlib

Step 2: Run Script
python main.py

Step 3: Output

AUC printed in terminal

ROC curve displayed

Best hyperparameters printed

---

### 11. Author

Developed by Subhasish Praharaj


B.Tech, Silicon University


Machine Learning Project – Loan Default Prediction
