# python/train_model.py
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ---------------- Paths ----------------
BASE_DIR = os.path.dirname(__file__)  # src/python
DATA_PATH = os.path.join(BASE_DIR, "..", "..", "data", "train.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_FILE = os.path.join(MODEL_DIR, "loan_model.pkl")
SCALER_FILE = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODERS_FILE = os.path.join(MODEL_DIR, "encoders.pkl")
FEATURE_ORDER_FILE = os.path.join(MODEL_DIR, "feature_order.npy")

# ---------------- Load ----------------
df = pd.read_csv(DATA_PATH)

# ---------------- Clean ----------------
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

# We will encode all object columns (including Loan_Status), but Loan_Status will be y
categorical_cols = [c for c in df.columns if df[c].dtype == "object"]
encoders: dict[str, LabelEncoder] = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# ---------------- Split X/y ----------------
# Keep a clear, stable feature order for inference:
feature_cols = [c for c in df.columns if c not in ("Loan_ID", "Loan_Status")]
X = df[feature_cols].copy()
y = df["Loan_Status"].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# ---------------- Scale ----------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------- Train ----------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# ---------------- Evaluate ----------------
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc*100:.2f}%")

# ---------------- Save ----------------
with open(MODEL_FILE, "wb") as f:
    pickle.dump(model, f)

with open(SCALER_FILE, "wb") as f:
    pickle.dump(scaler, f)

with open(ENCODERS_FILE, "wb") as f:
    pickle.dump(encoders, f)

# Save exact feature order for inference
np.save(FEATURE_ORDER_FILE, np.array(feature_cols, dtype=object))

print(f"✅ Saved model -> {MODEL_FILE}")
print(f"✅ Saved scaler -> {SCALER_FILE}")
print(f"✅ Saved encoders -> {ENCODERS_FILE}")
print(f"✅ Saved feature order -> {FEATURE_ORDER_FILE}")

