# 🏦 Loan Prediction System  
**Machine Learning–Based Loan Approval Prediction with Streamlit Interface**

An end-to-end Loan Prediction System that uses Machine Learning to determine whether a loan application is likely to be **Approved or Rejected**, based on applicant and loan-related attributes.

---

## ✨ Features

- 📊 Predicts loan approval using a trained ML model  
- 🧠 Handles preprocessing: encoding, scaling, and feature alignment  
- 🌐 Interactive web interface built with Streamlit  

---

## 📁 Project Structure

Loan-Prediction-System/
│   \
├── data/ \
│ └── train.csv \
│ \
├── src/ \
│ ├── cpp/ \
│ │ └── driver.cpp \
│ │ \
│ └── python/ \
│ ├── train_model.py \
│ ├── predict.py \
│ └── streamlit_app.py \
│ \
├── .gitignore \
├── requirements.txt \
└── README.md

---

## 🧠 Machine Learning Workflow

1. Load dataset (`train.csv`)
2. Preprocess data (encoding & scaling)
3. Train classification model
4. Save trained model and preprocessing objects
5. Load model for prediction
6. Predict loan approval using CLI or Streamlit UI

---

## 📊 Input Parameters

| Feature | Description |
|-------|------------|
| Gender | Male / Female |
| Married | Yes / No |
| Dependents | 0 / 1 / 2 / 3+ |
| Education | Graduate / Not Graduate |
| Self Employed | Yes / No |
| Applicant Income | Integer |
| Coapplicant Income | Integer |
| Loan Amount | In thousands |
| Loan Amount Term | In days |
| Credit History | 1 = Good, 0 = Bad |
| Property Area | Urban / Semiurban / Rural |

---

## 🧪 Tech Stack

- Python 3
- Scikit-learn
- Pandas, NumPy
- Streamlit
- Pickle
- (Optional) C++

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/Loan-Prediction-System.git
cd Loan-Prediction-System
```
### 2️⃣ Clone the Repository
```bash
pip install -r requirements.txt
```

### 🧠 Train the Model (Optional)
```bash
python src/python/train_model.py
```
### 🔮 Run Prediction Script (CLI)
```bash
python src/python/predict.py
```

### 🌐 Run the Streamlit Web App
```bash
streamlit run src/python/streamlit_app.py
```

## 🚀 Deployment

- Deployable using **Streamlit Community Cloud**
- Entry point: `src/python/streamlit_app.py`
- No additional configuration required

---

## 📈 Model Information

- **Problem Type:** Binary Classification
- **Target Variable:** Loan Status (Approved / Rejected)
- **Evaluation Metric:** Accuracy

---

## 🔮 Future Enhancements

- Show prediction confidence / probability
- Improve UI/UX
- Add model comparison
- Store prediction history
- Full-stack deployment

---

## 👨‍💻 Author

**Yash Shashikant Yeole**  
B.Tech, Electrical Engineering  
Indian Institute of Technology Gandhinagar

---


## ⭐ Acknowledgements

- Kaggle Loan Prediction Dataset
- Streamlit Documentation
- Scikit-learn Community

