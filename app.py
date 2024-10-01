from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Charger le modèle de régression logistique sauvegardé
model = joblib.load('data/model_random_forest.pkl')
# Initialiser l'application Flask
app = Flask(__name__)

@app.route('/')
def home():
    # Rediriger vers la page index.html dans le dossier templates
    return render_template('index.html')

# Définir la route principale pour la prédiction
@app.route('/predict', methods=['POST'])
def predict():
    # Récupération des données du formulaire
    age = float(request.form['Age'])
    total_purchase = float(request.form['Total_Purchase'])
    account_manager = int(request.form['Account_Manager'])
    years = float(request.form['Years'])
    num_sites = int(request.form['Num_Sites'])

    # Préparation des données pour la prédiction
    features = np.array([[1,age, total_purchase, account_manager, years, num_sites]])

    # Faire une prédiction
    prediction_proba = model.predict_proba(input_data)[:, 1]
    prediction = (prediction_proba >= 0.5).astype(int)

    # Interprétation du résultat
    result = 'Churn' if prediction == 1 else 'No Churn'

    # Renvoyer le résultat à l'interface HTML
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
