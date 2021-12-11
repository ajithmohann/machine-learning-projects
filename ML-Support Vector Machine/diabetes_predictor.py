import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load dataset for pandas frame
diabetes_dataset = pd.read_csv('../data/diabetes.csv')

# --- Inspect dataset ---
# print(diabetes_dataset.shape)  # number of rows and columns
# print(diabetes_dataset.head())
# describe --> statistical measures of the data
# print(diabetes_dataset.describe())
# print(diabetes_dataset['Outcome'].value_counts())
# calculate mean of all labels
# print(diabetes_dataset.groupby('Outcome').mean())

# --- Separating data and Labels ---
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']
# print(X)
# print(Y)

# Standardize the data
scaler = StandardScaler()
scaler.fit(X)

standardized_data = scaler.transform(X)
# print(standardized_data)
X = standardized_data

# --- Training and Test data ---
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=.01, stratify=Y, random_state=1)
# print(X.shape, X_train.shape, X_test.shape)
# print(X_train)
# print(Y_train)

classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Accuracy on training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Training data accuracy:', training_data_accuracy)

# Accuracy on test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Test data accuracy:', test_data_accuracy)

# --- Making a Predictive System ---
input_data = (10, 115, 0, 0, 0, 35.3, 0.134, 29)

# changing the input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)
# print(input_data_as_numpy_array)

# reshape the np array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
# print(input_data_reshaped)

prediction = classifier.predict(input_data_reshaped)
print('SVM Prediction:', prediction)

if (prediction[0] == 0):
    print('Not diabetic')
else:
    print('Yes, diabetic')
