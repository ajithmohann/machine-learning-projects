import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load dataset for pandas frame
sonar_dataset = pd.read_csv('../data/sonar_data.csv', header=None)

# --- Inspect dataset ---
# print(sonar_dataset.shape)  # number of rows and columns
# print(sonar_dataset.head())
# # describe --> statistical measures of the data
# print(sonar_dataset.describe())
# print(sonar_dataset[60].value_counts())
# print(sonar_dataset.groupby(60).mean())  # calculate mean of all labels

# --- Separating data and Labels ---
X = sonar_dataset.drop(columns=60, axis=1)
Y = sonar_dataset[60]
# print(X)
# print(Y)

# --- Training and Test data ---
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=.01, stratify=Y, random_state=1)
# print(X.shape, X_train.shape, X_test.shape)
# print(X_train)
# print(Y_train)

classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# # Accuracy on training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Training data accuracy:', training_data_accuracy)

# Accuracy on test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Test data accuracy:', test_data_accuracy)

# --- Making a Predictive System ---
input_data = (0.0307, 0.0523, 0.0653, 0.0521, 0.0611, 0.0577, 0.0665, 0.0664, 0.1460, 0.2792, 0.3877, 0.4992, 0.4981, 0.4972, 0.5607, 0.7339, 0.8230, 0.9173, 0.9975, 0.9911, 0.8240, 0.6498, 0.5980, 0.4862, 0.3150, 0.1543, 0.0989, 0.0284, 0.1008,
              0.2636, 0.2694, 0.2930, 0.2925, 0.3998, 0.3660, 0.3172, 0.4609, 0.4374, 0.1820, 0.3376, 0.6202, 0.4448, 0.1863, 0.1420, 0.0589, 0.0576, 0.0672, 0.0269, 0.0245, 0.0190, 0.0063, 0.0321, 0.0189, 0.0137, 0.0277, 0.0152, 0.0052, 0.0121, 0.0124, 0.0055)

# changing the input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)
# print(input_data_as_numpy_array)

# reshape the np array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
# print(input_data_reshaped)

prediction = classifier.predict(input_data_reshaped)
print('SVM Prediction:', prediction)

if (prediction[0] == 'R'):
    print('The object is a Rock')
else:
    print('The object is a mine')
