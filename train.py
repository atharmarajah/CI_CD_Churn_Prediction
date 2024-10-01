import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle


data = pd.read_csv('customer_churn.csv')
numerical_data = data.select_dtypes(include=['number'])


X = numerical_data.drop('Churn',axis=1)
y = numerical_data['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Entraînement du modèle
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict probabilities
y_prob = rf.predict_proba(X_test)[:, 1]

# Convert probabilities to binary predictions using a threshold (e.g., 0.5)
y_pred = (y_prob >= 0.5).astype(int)

# Calculate the metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary')  # For binary classification
recall = recall_score(y_test, y_pred, average='binary')  # For binary classification
f1 = f1_score(y_test, y_pred, average='binary')  # For binary classification

# Print the metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")


with open('model_randpm_forest.pkl', 'wb') as file:
    pickle.dump(rf, file)

