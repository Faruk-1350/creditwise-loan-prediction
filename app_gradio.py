import gradio as gr
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# ---------- Load & train model (runs once at startup) ----------
def load_and_train():
    df = pd.read_csv("loan_approval_data.csv")

    features = [
        "Applicant_Income", "Coapplicant_Income", "Credit_Score",
        "DTI_Ratio", "Existing_Loans", "Savings",
        "Collateral_Value", "Loan_Amount", "Loan_Term"
    ]
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


model, scaler, FEATURES = load_and_train()


# ---------- Prediction function ----------
def predict_loan(applicant_income, coapplicant_income, credit_score,
                  dti_ratio, existing_loans, savings,
                  collateral_value, loan_amount, loan_term):

    input_data = pd.DataFrame([[
        applicant_income, coapplicant_income, credit_score,
        dti_ratio, existing_loans, savings,
        collateral_value, loan_amount, loan_term
    ]], columns=FEATURES)

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        result_md = f"""
### ✅ Likely Approved
**Estimated approval confidence: {probability:.1%}**

Based on the applicant profile entered above.
"""
        conf_value = probability
    else:
        result_md = f"""
### ❌ Likely Rejected
**Estimated rejection confidence: {(1 - probability):.1%}**

Based on the applicant profile entered above.
"""
        conf_value = probability

    return result_md, conf_value


# ---------- Custom theme (matches CreditWise navy/teal/coral palette) ----------
theme = gr.themes.Soft(
    primary_hue="teal",
    secondary_hue="orange",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
).set(
    body_background_fill="#FAF8F3",
    block_background_fill="#FFFFFF",
    block_border_color="#E4DFD3",
    button_primary_background_fill="#1B3A4B",
    button_primary_background_fill_hover="#2A9D8F",
    button_primary_text_color="#FFFFFF",
)

custom_css = """
#title { font-family: 'Lora', serif; font-size: 2.2rem; font-weight: 700; color: #1B3A4B; }
#kicker { color: #2A9D8F; font-weight: 600; font-size: 0.85rem; letter-spacing: 0.02em; }
#subtitle { color: #6B7280; font-size: 1rem; }
#footer { color: #6B7280; font-size: 0.85rem; margin-top: 1rem; border-top: 1px solid #E4DFD3; padding-top: 0.8rem; }
#footer a { color: #2A9D8F; }
"""

# ---------- Gradio UI ----------
with gr.Blocks(theme=theme, css=custom_css, title="CreditWise | Loan Approval Predictor") as demo:

    gr.Markdown('<div id="kicker">LOAN APPROVAL PREDICTION</div>')
    gr.Markdown('<div id="title">CreditWise</div>')
    gr.Markdown(
        '<div id="subtitle">Enter an applicant\'s financial profile below to estimate their loan '
        'approval likelihood, based on a logistic regression model trained on historical lending data.</div>'
    )

    gr.Markdown("### Applicant details")

    with gr.Row():
        with gr.Column():
            applicant_income = gr.Number(label="Applicant income (₹/month)", value=10000, minimum=0, step=500)
            coapplicant_income = gr.Number(label="Coapplicant income (₹/month)", value=0, minimum=0, step=500)
            credit_score = gr.Slider(label="Credit score", minimum=300, maximum=900, value=700, step=1)
            dti_ratio = gr.Slider(label="Debt-to-income ratio", minimum=0.0, maximum=1.0, value=0.3, step=0.01)
            existing_loans = gr.Number(label="Existing loans", value=1, minimum=0, maximum=10, step=1)

        with gr.Column():
            savings = gr.Number(label="Savings (₹)", value=10000, minimum=0, step=500)
            collateral_value = gr.Number(label="Collateral value (₹)", value=20000, minimum=0, step=500)
            loan_amount = gr.Number(label="Loan amount requested (₹)", value=15000, minimum=0, step=500)
            loan_term = gr.Dropdown(label="Loan term (months)", choices=[12, 24, 36, 48, 60, 72, 84], value=48)

    predict_btn = gr.Button("Predict loan approval", variant="primary")

    result_output = gr.Markdown()
    confidence_bar = gr.Slider(label="Approval confidence", minimum=0, maximum=1, interactive=False)

    gr.Markdown(
        "*This demo model was trained on a sample dataset for educational purposes "
        "and should not be used for real lending decisions.*"
    )

    predict_btn.click(
        fn=predict_loan,
        inputs=[applicant_income, coapplicant_income, credit_score,
                dti_ratio, existing_loans, savings,
                collateral_value, loan_amount, loan_term],
        outputs=[result_output, confidence_bar]
    )

    gr.Markdown(
        '<div id="footer">Built by Md Faruk Ansari. '
        '<a href="https://github.com/Faruk-1350/creditwise-loan-prediction" target="_blank">View the source on GitHub</a>.</div>'
    )

if __name__ == "__main__":
    demo.launch()
