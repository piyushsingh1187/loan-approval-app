import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------------------------------------------------------
# Page setup -- MUST be the very first Streamlit command in the script
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Loan Approval Predictor", page_icon="💰", layout="centered")

# ---------------------------------------------------------------------------
# Load trained artifacts (created by save_artifacts.py)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("loan_model.pkl")
    scaler = joblib.load("scaler.pkl")
    le_education = joblib.load("label_encoder_education.pkl")
    le_target = joblib.load("label_encoder_target.pkl")
    ohe = joblib.load("onehot_encoder.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, scaler, le_education, le_target, ohe, feature_columns

model, scaler, le_education, le_target, ohe, feature_columns = load_artifacts()

# Columns exactly as used during training
NUMERIC_COLS = [
    "Applicant_Income", "Coapplicant_Income", "Age", "Dependents",
    "Credit_Score", "Existing_Loans", "DTI_Ratio", "Savings",
    "Collateral_Value", "Loan_Amount", "Loan_Term",
]
OHE_COLS = [
    "Employment_Status", "Marital_Status", "Loan_Purpose",
    "Property_Area", "Gender", "Employer_Category",
]

st.title("💰 Loan Approval Predictor")
st.write("Fill in the applicant's details below to predict loan approval.")

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
with st.form("loan_form"):
    st.subheader("Applicant Details")
    col1, col2 = st.columns(2)

    with col1:
        applicant_income = st.number_input("Applicant Income", min_value=0.0, value=50000.0, step=1000.0)
        coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0, value=0.0, step=1000.0)
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        dependents = st.number_input("Dependents", min_value=0, max_value=10, value=0)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
        existing_loans = st.number_input("Existing Loans", min_value=0, max_value=20, value=0)

    with col2:
        dti_ratio = st.number_input("DTI Ratio (%)", min_value=0.0, max_value=100.0, value=30.0)
        savings = st.number_input("Savings", min_value=0.0, value=10000.0, step=1000.0)
        collateral_value = st.number_input("Collateral Value", min_value=0.0, value=0.0, step=1000.0)
        loan_amount = st.number_input("Loan Amount", min_value=0.0, value=100000.0, step=1000.0)
        loan_term = st.number_input("Loan Term (months)", min_value=1.0, value=360.0, step=12.0)

    st.subheader("Other Details")
    col3, col4 = st.columns(2)

    with col3:
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married"])
        education_level = st.selectbox("Education Level", ["Graduate", "Not Graduate"])

    with col4:
        employment_status = st.selectbox("Employment Status", ["Salaried", "Self-employed", "Contract", "Unemployed"])
        employer_category = st.selectbox("Employer Category", ["Private", "Government", "MNC", "Business", "Unemployed"])
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    loan_purpose = st.selectbox("Loan Purpose", ["Personal", "Car", "Business", "Home", "Education"])

    submitted = st.form_submit_button("Predict Approval")

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
if submitted:
    # 1. Build a single-row dataframe matching the raw training schema
    raw = pd.DataFrame([{
        "Applicant_Income": applicant_income,
        "Coapplicant_Income": coapplicant_income,
        "Employment_Status": employment_status,
        "Age": age,
        "Marital_Status": marital_status,
        "Dependents": dependents,
        "Credit_Score": credit_score,
        "Existing_Loans": existing_loans,
        "DTI_Ratio": dti_ratio,
        "Savings": savings,
        "Collateral_Value": collateral_value,
        "Loan_Amount": loan_amount,
        "Loan_Term": loan_term,
        "Loan_Purpose": loan_purpose,
        "Property_Area": property_area,
        "Education_Level": education_level,
        "Gender": gender,
        "Employer_Category": employer_category,
    }])

    # 2. Label-encode Education_Level (same encoder used in training)
    raw["Education_Level"] = le_education.transform(raw["Education_Level"])

    # 3. One-hot encode the categorical columns (same encoder used in training)
    encoded = ohe.transform(raw[OHE_COLS])
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(OHE_COLS), index=raw.index)
    processed = pd.concat([raw.drop(columns=OHE_COLS), encoded_df], axis=1)

    # 4. Reorder / align columns exactly as the model expects
    #    (adds any missing dummy columns as 0, drops anything extra)
    processed = processed.reindex(columns=feature_columns, fill_value=0)

    # 5. Scale
    processed_scaled = scaler.transform(processed)

    # 6. Predict
    prediction = model.predict(processed_scaled)[0]
    probability = model.predict_proba(processed_scaled)[0]

    label = le_target.inverse_transform([prediction])[0]
    approve_prob = probability[list(le_target.classes_).index("Yes")] if "Yes" in le_target.classes_ else max(probability)

    st.divider()
    if label == "Yes":
        st.success(f"✅ Loan Approved  (confidence: {approve_prob*100:.1f}%)")
    else:
        st.error(f"❌ Loan Not Approved  (confidence: {(1-approve_prob)*100:.1f}%)")

    with st.expander("See raw model output"):
        st.write("Prediction (encoded):", prediction)
        st.write("Class probabilities:", dict(zip(le_target.classes_, probability)))