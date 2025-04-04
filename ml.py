import numpy as np
import pandas as pd
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
from imblearn.over_sampling import SMOTE

# Load dataset
tdata = pd.read_csv("/content/Integrated.csv", header=None, na_values=[-9])

# Select necessary columns
new_data = tdata[[2,3,8,9,14,15,16,17,18,31,57]].copy()
new_data.columns = ['Age','Sex','Chest Pain','Blood Pressure','Smoking Years','Fasting Blood Sugar','Diabetes History',
                    'Family history Cornory','ECG','Pulse Rate','Target']

# Fill missing values
new_data.fillna(new_data.mean(), inplace=True)
new_data['Diabetes History'].fillna(new_data['Diabetes History'].mode()[0], inplace=True)
new_data['Family history Cornory'].fillna(new_data['Family history Cornory'].mode()[0], inplace=True)

# Convert target to binary (Heart Attack Risk: 1 = High, 0 = Low)
new_data['Target'] = new_data['Target'].apply(lambda x: 1 if x >= 3 else 0)

# Split features and labels
X = new_data.iloc[:, :-1].values
y = new_data.iloc[:, -1].values

# *Fix: Normalize only continuous numerical features*
scaler = preprocessing.StandardScaler()
X[:, [0, 3, 4, 5, 8, 9]] = scaler.fit_transform(X[:, [0, 3, 4, 5, 8, 9]])

# *Fix: Handle Class Imbalance Using SMOTE*
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# Split into training and testing sets
training_X, testing_X, training_y, testing_y = train_test_split(X, y, test_size=0.10, random_state=70)

# Train models
print("Training Models...")

svm_model = svm.SVC(kernel='rbf', C=5, gamma='scale', probability=True)
svm_model.fit(training_X, training_y)

log_reg = LogisticRegression(solver='lbfgs')
log_reg.fit(training_X, training_y)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(training_X, training_y)

nn_model = Sequential([
    Dense(10, activation='relu', input_dim=10, kernel_regularizer=regularizers.l2(0.01)),
    Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dense(1, activation='sigmoid')  # Binary Classification: Sigmoid Activation
])
nn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
nn_model.fit(training_X, training_y, epochs=50, verbose=0)
# Function to Take User Input and Predict
def predict_heart_attack():
    print("\nEnter the following details to predict heart attack risk:")
    age = float(input("Age: "))
    sex = int(input("Sex (1=Male, 0=Female): "))
    chest_pain = int(input("Chest Pain Level (0-4): "))
    blood_pressure = float(input("Blood Pressure: "))
    smoking_years = float(input("Years of Smoking: "))
    fasting_blood_sugar = float(input("Fasting Blood Sugar: "))
    diabetes_history = int(input("Diabetes History (1=Yes, 0=No): "))
    family_history = int(input("Family History of Heart Disease (1=Yes, 0=No): "))
    ecg = float(input("ECG: "))
    pulse_rate = float(input("Pulse Rate: "))

    # Prepare user input as an array
    user_data = np.array([[age, sex, chest_pain, blood_pressure, smoking_years,
                           fasting_blood_sugar, diabetes_history, family_history, ecg, pulse_rate]])

    # *Fix: Apply same scaling as training data*
    user_data[:, [0, 3, 4, 5, 8, 9]] = scaler.transform(user_data[:, [0, 3, 4, 5, 8, 9]])

    # *Fix: Print probabilities for debugging*
    svm_prob = svm_model.predict_proba(user_data)[0][1]
    log_reg_prob = log_reg.predict_proba(user_data)[0][1]
    knn_prob = knn.predict_proba(user_data)[0][1]
    nn_prob = nn_model.predict(user_data)[0][0]

    print("\nModel Probabilities:")
    print(f"SVM Probability: {svm_prob:.2f}")
    print(f"Logistic Regression Probability: {log_reg_prob:.2f}")
    print(f"KNN Probability: {knn_prob:.2f}")
    print(f"Neural Network Probability: {nn_prob:.2f}")

    # *Fix: Adjust probability threshold*
    threshold = 0.4  # Reduce threshold slightly to allow detecting more high-risk cases

    svm_pred = svm_prob > threshold
    log_reg_pred = log_reg_prob > threshold
    knn_pred = knn_prob > threshold
    nn_pred = nn_prob > threshold

    # Show predictions
    print("\nPredictions from different models:")
    print(f"SVM Model Prediction: {'High Risk' if svm_pred else 'Low Risk'}")
    print(f"Logistic Regression Prediction: {'High Risk' if log_reg_pred else 'Low Risk'}")
    print(f"KNN Model Prediction: {'High Risk' if knn_pred else 'Low Risk'}")
    print(f"Neural Network Prediction: {'High Risk' if nn_pred else 'Low Risk'}")

# Call the function to take input and predict
predict_heart_attack()
