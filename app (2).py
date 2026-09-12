import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="CreditWise | Loan Approval Predictor", page_icon="◆", layout="centered")

# ---------- Design system ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

:root {
    --navy: #1B3A4B;
    --paper: #FAF8F3;
    --teal: #2A9D8F;
    --coral: #E76F51;
    --slate: #22333B;
    --muted: #6B7280;
    --line: #E4DFD3;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: var(--slate); }
.stApp { background-color: var(--paper); }

/* Hero */
.cw-hero {
    padding: 2.2rem 0 1.4rem 0;
    border-bottom: 1px solid var(--line);
    margin-bottom: 1.8rem;
}
.cw-kicker {
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
    color: var(--teal);
    font-weight: 600;
    letter-spacing: 0.02em;
    margin-bottom: 0.4rem;
}
.cw-title {
    font-family: 'Lora', serif;
    font-size: 2.6rem;
    font-weight: 700;
    color: var(--navy);
    line-height: 1.1;
    margin-bottom: 0.5rem;
}
.cw-sub {
    font-size: 1.02rem;
    color: var(--muted);
    max-width: 480px;
    line-height: 1.5;
}

/* Section labels */
.cw-section-label {
    font-family: 'Lora', serif;
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--navy);
    margin: 0.2rem 0 1rem 0;
}

/* Form card */
.cw-card {
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-radius: 4px;
    padding: 1.8rem 1.8rem 0.6rem 1.8rem;
    margin-bottom: 1.6rem;
}

/* Result panel */
.cw-result {
    border-left: 4px solid var(--teal);
    background: #FFFFFF;
    border-top: 1px solid var(--line);
    border-right: 1px solid var(--line);
    border-bottom: 1px solid var(--line);
    padding: 1.6rem 1.8rem;
    margin-top: 1.2rem;
}
.cw-result.reject { border-left-color: var(--coral); }
.cw-result-label {
    font-size: 0.85rem;
    color: var(--muted);
    margin-bottom: 0.3rem;
}
.cw-result-value {
    font-family: 'Lora', serif;
    font-size: 2.1rem;
    font-weight: 700;
    color: var(--teal);
    margin-bottom: 0.6rem;
}
.cw-result.reject .cw-result-value { color: var(--coral); }
.cw-result-note {
    font-size: 0.88rem;
    color: var(--muted);
    line-height: 1.5;
}

/* Streamlit widget tweaks */
div[data-testid="stNumberInput"] label, div[data-testid="stSlider"] label, div[data-testid="stSelectbox"] label {
    color: var(--slate) !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
}
.stButton button {
    background-color: var(--navy);
    color: white;
    border-radius: 3px;
    border: none;
    font-weight: 500;
    padding: 0.6rem 1.6rem;
}
.stButton button:hover {
    background-color: var(--teal);
    color: white;
}
.cw-footer {
    border-top: 1px solid var(--line);
    margin-top: 2rem;
    padding-top: 1rem;
    font-size: 0.85rem;
    color: var(--muted);
}
.cw-footer a { color: var(--teal); }
</style>
""", unsafe_allow_html=True)

# ---------- Hero ----------
st.markdown("""
<div class="cw-hero">
    <div class="cw-kicker">Loan approval prediction</div>
    <div class="cw-title">CreditWise</div>
    <div class="cw-sub">Enter an applicant's financial profile below to estimate their loan approval likelihood, based on a logistic regression model trained on historical lending data.</div>
</div>
""", unsafe_allow_html=True)

# ---------- Load & train model (runs once, cached) ----------
@st.cache_resource
def load_and_train():
    df = pd.read_csv("loan_approval_data.csv")

    # Keep it simple: use core numeric features only
    features = [
        "Applicant_Income", "Coapplicant_Income", "Credit_Score",
        "DTI_Ratio", "Existing_Loans", "Savings",
        "Collateral_Value", "Loan_Amount", "Loan_Term"
    ]
    # Only drop rows with missing values in the columns we actually use
    df = df.dropna(subset=features + ["Loan_Approved"])

    X = df[features]
    y_raw = df["Loan_Approved"].astype(str).str.strip().str.lower()
    y = y_raw.map({"y": 1, "yes": 1, "1": 1, "n": 0, "no": 0, "0": 0})
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
st.markdown('<div class="cw-section-label">Applicant details</div>', unsafe_allow_html=True)
st.markdown('<div class="cw-card">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    applicant_income = st.number_input("Applicant income (₹/month)", min_value=0, value=10000, step=500)
    coapplicant_income = st.number_input("Coapplicant income (₹/month)", min_value=0, value=0, step=500)
    credit_score = st.slider("Credit score", min_value=300, max_value=900, value=700)
    dti_ratio = st.slider("Debt-to-income ratio", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
    existing_loans = st.number_input("Existing loans", min_value=0, max_value=10, value=1)

with col2:
    savings = st.number_input("Savings (₹)", min_value=0, value=10000, step=500)
    collateral_value = st.number_input("Collateral value (₹)", min_value=0, value=20000, step=500)
    loan_amount = st.number_input("Loan amount requested (₹)", min_value=0, value=15000, step=500)
    loan_term = st.selectbox("Loan term (months)", [12, 24, 36, 48, 60, 72, 84], index=3)

st.markdown('</div>', unsafe_allow_html=True)

predict_clicked = st.button("Predict loan approval", type="primary")

# ---------- Predict ----------
if predict_clicked:
    input_data = pd.DataFrame([[
        applicant_income, coapplicant_income, credit_score,
        dti_ratio, existing_loans, savings,
        collateral_value, loan_amount, loan_term
    ]], columns=features)

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.markdown(f"""
        <div class="cw-result">
            <div class="cw-result-label">Prediction</div>
            <div class="cw-result-value">Likely approved</div>
            <div class="cw-result-note">Estimated approval confidence: {probability:.1%}, based on the applicant profile entered above.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="cw-result reject">
            <div class="cw-result-label">Prediction</div>
            <div class="cw-result-value">Likely rejected</div>
            <div class="cw-result-note">Estimated rejection confidence: {(1-probability):.1%}, based on the applicant profile entered above.</div>
        </div>
        """, unsafe_allow_html=True)

    st.progress(float(probability))
    st.caption("This demo model was trained on a sample dataset for educational purposes and should not be used for real lending decisions.")

st.markdown("""
<div class="cw-footer">
Built by Md Faruk Ansari. <a href="https://github.com/Faruk-1350/creditwise-loan-prediction" target="_blank">View the source on GitHub</a>.
</div>
""", unsafe_allow_html=True)
