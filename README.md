# 📘 Credit Default Prediction using CatBoost Classifier

This repository contains a complete, end-to-end machine learning workflow for predicting **credit card default risk** using the **CatBoostClassifier**, a state-of-the-art gradient boosting algorithm designed for tabular data and categorical features.

The project demonstrates:

- Advanced preprocessing techniques  
- Hyperparameter tuning (GridSearchCV)  
- Comprehensive evaluation (ROC AUC, confusion matrix, classification report)  
- Explainability using **SHAP**  
- Multiple creative and insightful visualizations  
- Best practices for model development in credit-risk analytics  

---

## 🔍 Project Overview

Credit default prediction is a core problem in financial risk modelling. Banks and credit providers use such models to:

- Estimate customer creditworthiness  
- Reduce financial risk  
- Set appropriate credit limits  
- Improve regulatory compliance  
- Focus on risky customer segments  

This project applies **CatBoost**, which excels on tabular datasets with categorical variables, to the **UCI Credit Card Default dataset (30,000 samples)**.

---

## 📊 Dataset

**Source:** UCI Machine Learning Repository  
**Rows:** 30,000  
**Columns:** 25  

Key features include:

- `LIMIT_BAL` – Credit limit  
- `SEX`, `EDUCATION`, `MARRIAGE` – Demographics  
- `AGE` – Customer age  
- `PAY_0` … `PAY_6` – Repayment history  
- `BILL_AMT1` … `BILL_AMT6` – Monthly bill amounts  
- `PAY_AMT1` … `PAY_AMT6` – Payment history  
- `default.payment.next.month` – Target variable  

Class distribution:

- **No Default (0):** 23,364  
- **Default (1):** 6,636  

---

## 🛠️ Project Pipeline

### 1️⃣ Data Preprocessing
- Handle categorical & numerical features  
- One-hot encoding or CatBoost native categories  
- Optional feature scaling  
- 80/20 train–test split (stratified)

### 2️⃣ Model Development
CatBoost parameters:

```python
iterations=150
depth=6
learning_rate=0.05
loss_function='Logloss'
```

### 3️⃣ Hyperparameter Tuning

Performed using GridSearchCV with 3-fold cross-validation.

### 4️⃣ Evaluation Metrics

- Accuracy ≈ 0.82

- ROC AUC ≈ 0.78

- Precision (default class=1) ≈ 0.66

- Recall (default class=1) ≈ 0.36

- Confusion matrix

- Classification report