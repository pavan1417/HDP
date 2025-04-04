import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
from imblearn.over_sampling import SMOTE
import joblib

# Load dataset
tdata = pd.read_csv("Integrated.csv", header=None, na_values=[-9])

# Select relevant columns
columns = ['Age', 'Sex', 'Chest Pain', 'Blood Pressure', 'Smoking Years', 'Fasting Blood Sugar',
           'Diabetes History', 'Family history Cornory', 'ECG', 'Pulse Rate', 'Target']

new_data = tdata[[2, 3, 8, 9, 14, 15, 16, 17, 18, 31, 57]].copy()
new_data.columns = columns

# Fill missing values
new_data.fillna(new_data.mean(), inplace=True)
new_data['Diabetes History'].fillna(new_data['Diabetes History'].mode()[0], inplace=True)
new_data['Family history Cornory'].fillna(new_data['Family history Cornory'].mode()[0], inplace=True)

# Convert Target variable (Heart Attack Risk: 1 = High, 0 = Low)
new_data['Target'] = new_data['Target'].apply(lambda x: 1 if x >= 3 else 0)

# Split into features (X) and labels (y)
X = new_data.iloc[:, :-1].values
y = new_data.iloc[:, -1].values

# Normalize only continuous numerical features
scaler = StandardScaler()
X[:, [0, 3, 4, 5, 8, 9]] = scaler.fit_transform(X[:, [0, 3, 4, 5, 8, 9]])

# Handle Class Imbalance Using SMOTE (only on training set)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=70)
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Train Models
st.write("### Training Models...")

svm_model = svm.SVC(kernel='rbf', C=5, gamma='scale', probability=True)
svm_model.fit(X_train, y_train)

log_reg = LogisticRegression(solver='lbfgs')
log_reg.fit(X_train, y_train)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Train Neural Network
nn_model = Sequential([
    Dense(10, activation='relu', input_dim=10, kernel_regularizer=regularizers.l2(0.01)),
    Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dense(1, activation='sigmoid')
])
nn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
nn_model.fit(X_train, y_train, epochs=50, verbose=0)

# Save Models
joblib.dump(svm_model, "svm_model.pkl")
joblib.dump(log_reg, "log_reg.pkl")
joblib.dump(knn, "knn_model.pkl")
joblib.dump(scaler, "scaler.pkl")

st.success("Models Trained and Saved Successfully!")

# Streamlit UI for Predictions
st.title("Heart Attack Risk Prediction")

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

if st.button("Predict"):
    # Convert Inputs
    sex = 1 if sex == "Male" else 0
    diabetes_history = 1 if diabetes_history == "Yes" else 0
    family_history = 1 if family_history == "Yes" else 0

    user_data = np.array([[age, sex, chest_pain, blood_pressure, smoking_years,
                           fasting_blood_sugar, diabetes_history, family_history, ecg, pulse_rate]])

    # Load the saved scaler
    scaler = joblib.load("scaler.pkl")
    user_data[:, [0, 3, 4, 5, 8, 9]] = scaler.transform(user_data[:, [0, 3, 4, 5, 8, 9]])

    # Load Models and Predict
    svm_model = joblib.load("svm_model.pkl")
    log_reg = joblib.load("log_reg.pkl")
    knn = joblib.load("knn_model.pkl")

    svm_prob = svm_model.predict_proba(user_data)[0][1]
    log_reg_prob = log_reg.predict_proba(user_data)[0][1]
    knn_prob = knn.predict_proba(user_data)[0][1]
    nn_prob = nn_model.predict(user_data)[0][0].numpy()

    st.write("### Model Probabilities:")
    st.write(f"**SVM:** {svm_prob:.2f}")
    st.write(f"**Logistic Regression:** {log_reg_prob:.2f}")
    st.write(f"**KNN:** {knn_prob:.2f}")
    st.write(f"**Neural Network:** {nn_prob:.2f}")

    # Probability Threshold
    threshold = 0.4
    svm_pred = "High Risk" if svm_prob > threshold else "Low Risk"
    log_reg_pred = "High Risk" if log_reg_prob > threshold else "Low Risk"
    knn_pred = "High Risk" if knn_prob > threshold else "Low Risk"
    nn_pred = "High Risk" if nn_prob > threshold else "Low Risk"

    # Display Predictions
    st.write("### Final Predictions:")
    st.write(f"**SVM Model Prediction:** {svm_pred}")
    st.write(f"**Logistic Regression Prediction:** {log_reg_pred}")
    st.write(f"**KNN Model Prediction:** {knn_pred}")
    st.write(f"**Neural Network Prediction:** {nn_pred}")
