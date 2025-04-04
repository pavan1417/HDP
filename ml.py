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

