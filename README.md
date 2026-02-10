# 🏦 Credit Risk Scoring & Default Prediction

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Libraries](https://img.shields.io/badge/Library-Scikit--Learn%20%7C%20CatBoost%20%7C%20XGBoost%20%7C%20Pandas-orange)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

## 📌 Project Overview

This project focuses on building a machine learning model to predict the probability of a loan default. The goal is to classify borrowers into two categories: **reliable** (0) and **risky** (1).


## 📂 Dataset Description

The dataset consists of **32,581** records with the following features:

| Feature | Description | Type |
| :--- | :--- | :--- |
| `person_age` | Age of the borrower | Numerical |
| `person_income` | Annual income | Numerical |
| `person_home_ownership` | Home ownership status (RENT, OWN, MORTGAGE, OTHER) | Categorical |
| `person_emp_length` | Employment length (in years) | Numerical |
| `loan_intent` | Purpose of the loan (Education, Medical, etc.) | Categorical |
| `loan_grade` | Loan grade (A-G internal risk score) | Categorical (Ordinal) |
| `loan_amnt` | Loan amount requested | Numerical |
| `loan_int_rate` | Interest rate | Numerical |
| `loan_percent_income` | Loan amount / Annual income | Numerical |
| `cb_person_default_on_file` | Historical default (Y/N) | Categorical |
| `cb_person_cred_hist_length`| Length of credit history (years) | Numerical |
| **`loan_status`** | **Target variable (0: Non-default, 1: Default)** | **Target** |


## ⚙️ Methodology

### 1. Data Cleaning & Preprocessing (`EDA.ipynb`)

* **Duplicate Removal:** Removed 165 repeated observations.
* **Outlier Removal:** Removed records with unrealistic `person_age` and `person_emp_length` (> 100).
* **Imputation:** Filled missing values in `loan_int_rate` using the median grouped by `loan_grade` and in `person_emp_length` using the median grouped by self-created `age_group`.
* **Feature Engineering:**

    * Take logarigthm of `person_income` in order to normalize the distribution (reduce skewness) and mitigate the impact of extreme outliers.
    * Created `age_group` bins to capture non-linear risk dependencies (probably, higher in youngest and oldest categories.
    * Created `emp_is_missing` in order to capture potential hidden patterns or risks associated with undisclosed employment length.
    * Merged rare category `OTHER` in `person_home_ownership` with `RENT` based on closer default mean value.
    
* **Encoding:**
  
    * *Ordinal Encoding* for `loan_grade`.
    * *One-Hot Encoding* for nominal variables (`loan_intent`, `person_home_ownership` and `age_group`).

### 2. Exploratory Data Analysis (Key Insights)

* **Interest Rate:** Higher interest rates strongly correlate with default probability.
* **Loan Grade:** The closer the grade to 'G', the higher the probability of default. Grade 'G' loans have nearly 100% default rate in some segments.
* **Interest Rate and Loan Grade** There is expected strong correlation between two features.
* **Rent vs Own:** Renters have a default rate of ~31%, while home owners have only ~7%.
* **Historical Default** Borrowers with a prior record of default show a higher probability of defaulting again

* **Loan Status** Taget variable is unbalanced. There is 25327 obsrvations with `0` (no default) and 7089 with `1` (default) in proportion ~ 78% to 22%.

### 3. Modeling Strategy (`modeling.ipynb`)

Several algorithms were compared several  handling the class imbalance problem. The primary metric for evaluation is **Recall** (to minimize missed defaults) and **F1-Score**.

$$
F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}
$$

Models tested:
* Logistic Regression (Baseline)
* Random Forest
* Gradient Boosting
* XGBoost
* CatBoost

## 📊 Premilinary results

Firstly, the models were evaluated on a hold-out test set (20%) **without hyperparameters tuning** (default sklearn (catboost, xgboost) hyperparameters were used).
The **CatBoost Classifier** showed the best performance, balancing Precision and Recall.

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **CatBoost** | **0.904** | **0.945** | **0.607** | **0.739** |
| XGBoost | 0.901 | 0.926 | 0.609 | 0.735 |
| Gradient Boosting | 0.897 | 0.923 | 0.593 | 0.722 |
| Random Forest | 0.895 | 0.901 | 0.600 | 0.721 |
| SVM | 0.886 | 0.896 | 0.556 | 0.686 |
| Logistic Regression | 0.843 | 0.722 | 0.490 | 0.584 |

But Recall (that is important in case of default detection) is low (maximum is just above 60%) in each model, so it should be taken into account and improved. 
