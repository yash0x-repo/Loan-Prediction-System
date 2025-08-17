# python/predict.py
import os
import pickle
import numpy as np
import pandas as pd

# ---------------- Paths ----------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # repo root
MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
MODEL_FILE = os.path.join(MODEL_DIR, "loan_model.pkl")
SCALER_FILE = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODERS_FILE = os.path.join(MODEL_DIR, "encoders.pkl")
FEATURE_ORDER_FILE = os.path.join(MODEL_DIR, "feature_order.npy")

# ---------------- Load assets ----------------
with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

with open(SCALER_FILE, "rb") as f:
    scaler = pickle.load(f)

with open(ENCODERS_FILE, "rb") as f:
    encoders: dict = pickle.load(f)

feature_order: list[str] = np.load(FEATURE_ORDER_FILE, allow_pickle=True).tolist()

# Which of these features are categorical? -> those present in encoders (except Loan_Status which is not in features)
categorical_features = set(encoders.keys()).intersection(set(feature_order))

# Friendly prompts for known Kaggle loan columns
PROMPTS = {
    "Gender": "Gender (Male/Female): ",
    "Married": "Married (Yes/No): ",
    "Dependents": "Number of dependents (0/1/2/3+): ",
    "Education": "Education (Graduate/Not Graduate): ",
    "Self_Employed": "Self Employed (Yes/No): ",
    "ApplicantIncome": "Applicant Income (integer): ",
    "CoapplicantIncome": "Coapplicant Income (integer): ",
    "LoanAmount": "Loan Amount (in thousands, e.g., 150 for 150k): ",
    "Loan_Amount_Term": "Loan Amount Term (in days, e.g., 360): ",
    "Credit_History": "Credit History (1 = good, 0 = bad): ",
    "Property_Area": "Property Area (Urban/Semiurban/Rural): ",
}

def input_numeric(prompt_text: str) -> float:
    while True:
        val = input(prompt_text).strip()
        try:
            return float(val)
        except ValueError:
            print("Please enter a valid number.")

def input_categorical(col: str) -> str:
    # show allowed categories from encoder
    le = encoders[col]
    allowed = ", ".join(map(str, le.classes_))
    prompt = PROMPTS.get(col, f"{col} ({allowed}): ")
    while True:
        val = input(prompt).strip()
        # exact match check (case sensitive to match training; we can be forgiving by normalizing)
        # Try exact, then try case-insensitive match:
        if val in le.classes_:
            return val
        # case-insensitive helper
        low_map = {c.lower(): c for c in le.classes_}
        if val.lower() in low_map:
            return low_map[val.lower()]
        print(f"Invalid value. Allowed: {allowed}")

def get_input_for_feature(col: str):
    if col in categorical_features:
        return input_categorical(col)
    # numeric
    return input_numeric(PROMPTS.get(col, f"{col} (number): "))

def main():
    print("\n====== Loan Approval Prediction ======\n")
    # Collect inputs in the exact feature order used during training
    data_dict = {}
    for col in feature_order:
        data_dict[col] = get_input_for_feature(col)

    # Build a one-row DataFrame
    df = pd.DataFrame([data_dict], columns=feature_order)

    # Apply label encoders to categorical columns
    for col in categorical_features:
        le = encoders[col]
        # Safe transform (value already validated)
        df[col] = le.transform(df[col])

    # Scale numeric columns (scaler was fit on full X in that feature order)
    X_scaled = scaler.transform(df[feature_order])

    # Predict class and probability
    pred = model.predict(X_scaled)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)[0, int(pred)]

    verdict = "✅ Loan Approved" if pred == 1 else "❌ Loan Rejected"
    if proba is not None:
        print(f"\n{verdict} (confidence: {proba*100:.1f}%)\n")
    else:
        print(f"\n{verdict}\n")

if __name__ == "__main__":
    main()
