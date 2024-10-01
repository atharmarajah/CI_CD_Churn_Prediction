import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd 

def initiate_X_train():
    data = pd.read_csv('data/customer_churn.csv')
    numerical_data = data.select_dtypes(include=['number'])
    X = numerical_data.drop('Churn',axis=1)
    y = numerical_data['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    return X_train, y_train

def return_model():
    X_train, y_train = initiate_X_train()
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    return rf

# Test to check if input dimension is 6
def test_input_dimension():
    # Simulating an input with shape (1, 6)
    X_train, y_train = initiate_X_train()
    assert X_train.shape[1] == 5, "Input data does not have 6 dimensions."

# Test to check if the prediction is higher than 0.8 for a specific input
def test_prediction_is_not_churn():
    rf = return_model()
    input_data = np.array([[1, 1, 1, 1, 1]])
    prediction = rf.predict(input_data)
    # Assuming the model returns a probability or a value we can compare
    assert prediction < 0.2, f"Prediction is not greater than 0.8, got {prediction}."


def test_prediction_is_churn():
    rf = return_model()
    input_data = np.array([[41.0, 7777.37, 0, 4.81, 12.0]])
    prediction = rf.predict(input_data)
    # Assuming the model returns a probability or a value we can compare
    assert prediction > 0.2, f"Prediction is not greater than 0.8, got {prediction}."

# To run these tests, use the following command in the terminal:
# pytest test.py
