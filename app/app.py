# app/app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# Charger le modèle et le scaler
model = joblib.load('../models/random_forest_churn.pkl')
scaler = joblib.load('../models/scaler.pkl')

# Liste des colonnes nécessaires (pour les features catégorielles)
# On utilise les colonnes du modèle entraîné
feature_names = model.feature_names_in_

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer les données du formulaire
        recency = float(request.form['recency'])
        frequency = float(request.form['frequency'])
        monetary = float(request.form['monetary'])
        age = float(request.form.get('age', 30))
        
        # Créer un DataFrame avec les bonnes colonnes
        # Pour simplifier, on utilise les 3 features principales
        # (en réalité, il faudrait toutes les colonnes)
        
        # Version simplifiée pour démo
        # Dans la vraie app, il faudrait toutes les colonnes encodées
        
        features = np.array([[recency, frequency, monetary, age]])
        
        # Normaliser
        features_scaled = scaler.transform(features)
        
        # Prédire
        prediction = model.predict(features_scaled)[0]
        proba = model.predict_proba(features_scaled)[0][1]
        
        if prediction == 1:
            result = f"⚠️ RISQUE DE DÉPART ÉLEVÉ ({proba*100:.1f}%)"
            color = "red"
        else:
            result = f"✅ CLIENT FIDÈLE (risque de départ: {proba*100:.1f}%)"
            color = "green"
        
        return render_template('index.html', 
                               prediction=result, 
                               color=color,
                               recency=recency,
                               frequency=frequency,
                               monetary=monetary,
                               age=age)
    
    except Exception as e:
        return render_template('index.html', 
                               error=f"Erreur: {str(e)}")

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API pour prédire via JSON"""
    data = request.get_json()
    features = np.array([[
        data['recency'],
        data['frequency'],
        data['monetary'],
        data.get('age', 30)
    ]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    proba = model.predict_proba(features_scaled)[0][1]
    
    return jsonify({
        'churn': int(prediction),
        'probability': float(proba),
        'risk': 'high' if prediction == 1 else 'low'
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)