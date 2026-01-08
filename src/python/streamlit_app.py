import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(__file__))
from predict import predict_loan

# ------------------ Page config ------------------
st.set_page_config(page_title="Loan Approval Predictor", layout="centered")
st.title("🏦 Loan Approval Predictor")
st.write("Enter applicant details below:")

# ------------------ Input blocks ------------------
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

applicant_income = st.number_input("Applicant Income", min_value=0, step=500)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0, step=500)
loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0, step=10)
loan_term = st.number_input("Loan Amount Term (days)", min_value=0, step=30)
credit_history = st.selectbox("Credit History", [1.0, 0.0])

# ------------------ Predict button ------------------
if st.button("Predict Loan Approval"):

    input_data = {
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area": property_area
    }

    try:
        pred, proba = predict_loan(input_data)

        if pred == 1:
            st.success(f"✅ Loan Approved (confidence: {proba*100:.1f}%)")
        else:
            st.error(f"❌ Loan Rejected (confidence: {proba*100:.1f}%)")

    except Exception as e:
        st.error(f"Error: {e}")
