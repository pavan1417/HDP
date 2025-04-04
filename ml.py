import numpy as np
import pandas as pd
import streamlit as st
import joblib
from sklearn.preprocessing import StandardScaler

# Load trained models
scaler = joblib.load("scaler.pkl")
svm_model = joblib.load("svm_model.pkl")
log_reg = joblib.load("log_reg.pkl")
knn = joblib.load("knn_model.pkl")

# Streamlit UI
st.title("🔴 Blink Heart Attack Risk Prediction App")

st.write("""
### Enter Your Health Details:
Fill in the details below to check your **risk of a heart attack**.
""")

# User Inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30)
sex = st.radio("Sex", ["Male", "Female"])
chest_pain = st.slider("Chest Pain Level (0-4)", 0, 4, 2)
blood_pressure = st.number_input("Blood Pressure", min_value=80, max_value=200, value=120)
smoking_years = st.number_input("Years of Smoking", min_value=0, max_value=50, value=5)
fasting_blood_sugar = st.number_input("Fasting Blood Sugar", min_value=70, max_value=200, value=90)
diabetes_history = st.radio("Diabetes History", ["Yes", "No"])
family_history = st.radio("Family History of Heart Disease", ["Yes", "No"])
ecg = st.number_input("ECG", min_value=-5.0, max_value=5.0, value=0.0)
pulse_rate = st.number_input("Pulse Rate", min_value=50, max_value=150, value=75)

# Prediction Button
if st.button("Predict Risk"):
    # Convert inputs
    sex = 1 if sex == "Male" else 0
    diabetes_history = 1 if diabetes_history == "Yes" else 0
    family_history = 1 if family_history == "Yes" else 0

    # Prepare input data
    user_data = np.array([[age, sex, chest_pain, blood_pressure, smoking_years,
                           fasting_blood_sugar, diabetes_history, family_history, ecg, pulse_rate]])

    # Scale numerical features
    user_data[:, [0, 3, 4, 5, 8, 9]] = scaler.transform(user_data[:, [0, 3, 4, 5, 8, 9]])

    # Predict probabilities
    svm_prob = svm_model.predict_proba(user_data)[0][1]
    log_reg_prob = log_reg.predict_proba(user_data)[0][1]
    knn_prob = knn.predict_proba(user_data)[0][1]

    # Display results
    st.write("### Model Predictions:")
    st.write(f"**SVM Model Risk Probability:** {svm_prob:.2f}")
    st.write(f"**Logistic Regression Risk Probability:** {log_reg_prob:.2f}")
    st.write(f"**KNN Model Risk Probability:** {knn_prob:.2f}")

    # Final Decision
    threshold = 0.4
    final_prediction = "High Risk" if max(svm_prob, log_reg_prob, knn_prob) > threshold else "Low Risk"
    
    st.subheader(f"Final Prediction: **{final_prediction}**")
