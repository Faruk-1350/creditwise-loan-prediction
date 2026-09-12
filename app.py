import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="CreditWise - Loan Approval Predictor", page_icon="💰")

st.title("💰 CreditWise")
st.subheader("Loan Approval Prediction System")
st.write("Enter applicant details below to predict loan approval likelihood.")

# ---------- Load & train model (runs once, cached) ----------
@st.cache_resource
def load_and_train():
    df = pd.read_csv("loan_approval_data.csv")
    df = df.dropna()

    # Keep it simple: use core numeric features only
    features = [
        "Applicant_Income", "Coapplicant_Income", "Credit_Score",
        "DTI_Ratio", "Existing_Loans", "Savings",
        "Collateral_Value", "Loan_Amount", "Loan_Term"
    ]
    X = df[features]
    y = df["Loan_Approved"]

    # Encode target if it's text (Yes/No)
    if y.dtype == object:
        y = y.map({"Y": 1, "N": 0, "Yes": 1, "No": 0}).fillna(y)
        y = y.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    return model, scaler, features

model, scaler, features = load_and_train()

# ---------- Input form ----------
st.divider()
col1, col2 = st.columns(2)

with col1:
    applicant_income = st.number_input("Applicant Income (₹/month)", min_value=0, value=10000, step=500)
    coapplicant_income = st.number_input("Coapplicant Income (₹/month)", min_value=0, value=0, step=500)
    credit_score = st.slider("Credit Score", min_value=300, max_value=900, value=700)
    dti_ratio = st.slider("Debt-to-Income Ratio", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
    existing_loans = st.number_input("Existing Loans (count)", min_value=0, max_value=10, value=1)

with col2:
    savings = st.number_input("Savings (₹)", min_value=0, value=10000, step=500)
    collateral_value = st.number_input("Collateral Value (₹)", min_value=0, value=20000, step=500)
    loan_amount = st.number_input("Loan Amount Requested (₹)", min_value=0, value=15000, step=500)
    loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60, 72, 84], index=3)

# ---------- Predict ----------
st.divider()
if st.button("Predict Loan Approval", type="primary"):
    input_data = pd.DataFrame([[
        applicant_income, coapplicant_income, credit_score,
        dti_ratio, existing_loans, savings,
        collateral_value, loan_amount, loan_term
    ]], columns=features)

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.success(f"✅ Loan likely to be **Approved** (confidence: {probability:.1%})")
    else:
        st.error(f"❌ Loan likely to be **Rejected** (confidence: {(1-probability):.1%})")

    st.caption("This is a demo model built on a sample dataset for educational purposes and should not be used for real lending decisions.")

st.divider()
st.caption("Built by Md Faruk Ansari | [GitHub Repo](https://github.com/Faruk-1350/creditwise-loan-prediction)")
