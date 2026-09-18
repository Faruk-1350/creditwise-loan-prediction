# CreditWise — Loan Approval Prediction System

An end-to-end supervised machine learning pipeline that predicts loan approval outcomes based on applicant financial and demographic data, deployed as an interactive web app.

🔗 **Live Demo:** [creditwise-loan-prediction-app21-ojx6rc.streamlit.app](https://faruk-1350-creditwise-loan-prediction-app21-ojx6rc.streamlit.app)

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
6. **Deployment** — deployed the best-performing model (Logistic Regression) as an interactive Streamlit web app, allowing users to input an applicant's profile and get a real-time approval prediction with confidence score

## Results

| Model | Precision | Recall | F1-Score | Accuracy |
|---|---|---|---|---|
| **Logistic Regression** | 0.78 | 0.84 | **0.81** | **0.88** |
| KNN | 0.67 | 0.57 | 0.62 | 0.785 |
| Naive Bayes | **0.81** | 0.70 | 0.75 | 0.86 |

**Logistic Regression** performed best overall, achieving the highest accuracy (88%) and F1-score (0.81), while also offering strong interpretability — an important factor in regulated domains like lending where decisions often need to be explained.

**Naive Bayes** achieved the highest Precision (0.81), making it a strong alternative if the business goal is to minimize false approvals (bad loans) over maximizing approvals.

## Live App

The deployed app lets users enter an applicant's financial profile (income, credit score, DTI ratio, savings, collateral, loan amount/term) and instantly see:
- A prediction — **Likely Approved** or **Likely Rejected**
- A confidence score for that prediction

Try it here: **[Live Demo](https://faruk-1350-creditwise-loan-prediction-app21-ojx6rc.streamlit.app)**

## Tools & Libraries

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Streamlit (deployment)

## How to Run Locally

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. To explore the analysis and model training: open `credit_wise.ipynb` in Jupyter Notebook or JupyterLab and run all cells
4. To run the web app locally: `streamlit run app.py`

## Future Improvements

- Add XGBoost/Random Forest for potentially stronger performance over the current baseline models
- Hyperparameter tuning (GridSearchCV) for each model
- Add SHAP-based explainability to show which features drive each prediction — important for transparency in lending decisions
- Handle class imbalance (if present) using techniques like SMOTE
- Expand evaluation beyond accuracy — add ROC-AUC and a business-cost-based analysis (cost of false approvals vs. false rejections)

## Author

Built by Md Faruk Ansari
