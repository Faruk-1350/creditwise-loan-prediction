# CreditWise — Loan Approval Prediction System

An end-to-end supervised machine learning pipeline that predicts loan approval outcomes based on applicant financial and demographic data.

## Overview

This project builds and compares three classification models to predict whether a loan application will be approved, using a dataset of 1,000 applicants with features like income, credit score, debt-to-income ratio, employment status, and collateral value.

## Dataset

- 1,000 applicant records, 20 original features
- Features include: Applicant Income, Coapplicant Income, Credit Score, DTI Ratio, Savings, Collateral Value, Loan Amount, Loan Term, Employment Status, Marital Status, Property Area, Education Level, and more

## Approach

1. **Exploratory Data Analysis (EDA)** — examined distributions, missing values, and relationships between features and loan approval
2. **Feature Engineering** — applied log transformation on skewed features (e.g., Applicant Income), created squared terms for non-linear relationships (Credit Score, DTI Ratio), and one-hot encoded categorical variables
3. **Preprocessing** — handled missing values, scaled numerical features, performed train/test split
4. **Modeling** — trained and compared three classifiers:
   - Logistic Regression
   - K-Nearest Neighbors (KNN)
   - Naive Bayes (Gaussian)
5. **Evaluation** — assessed each model using Precision, Recall, F1-score, Accuracy, and Confusion Matrix

## Results

| Model | Precision | Recall | F1-Score | Accuracy |
|---|---|---|---|---|
| **Logistic Regression** | 0.78 | 0.84 | **0.81** | **0.88** |
| KNN | 0.67 | 0.57 | 0.62 | 0.785 |
| Naive Bayes | **0.81** | 0.70 | 0.75 | 0.86 |

**Logistic Regression** performed best overall, achieving the highest accuracy (88%) and F1-score (0.81), while also offering strong interpretability — an important factor in regulated domains like lending where decisions often need to be explained.

**Naive Bayes** achieved the highest Precision (0.81), making it a strong alternative if the business goal is to minimize false approvals (bad loans) over maximizing approvals.

## Tools & Libraries

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

## How to Run

1. Clone this repository
2. Install dependencies: `pip install pandas numpy scikit-learn matplotlib seaborn`
3. Open `credit_wise.ipynb` in Jupyter Notebook or JupyterLab
4. Run all cells sequentially

## Future Improvements

- Hyperparameter tuning (GridSearchCV) for each model
- Try ensemble methods (Random Forest, Gradient Boosting) for potentially better performance
- Deploy as an interactive web app (e.g., using Streamlit) for live predictions
