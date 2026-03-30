# app/app.py
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Chemins des fichiers
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, '..', 'models', 'random_forest_churn.pkl')
scaler_path = os.path.join(base_dir, '..', 'models', 'scaler.pkl')
feature_indices_path = os.path.join(base_dir, '..', 'models', 'feature_indices.pkl')

print("="*50)
print("🚀 Démarrage de l'application Flask")
print("="*50)

# Charger le modèle et le scaler
print("📦 Chargement du modèle...")
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
print("✅ Modèle chargé avec succès!")

# Récupérer le nombre de features
try:
    n_features = model.n_features_in_
    print(f"📊 Nombre de features: {n_features}")
except AttributeError:
    n_features = scaler.mean_.shape[0]
    print(f"📊 Nombre de features estimé: {n_features}")

# Charger les indices des colonnes importantes
print("📊 Chargement des indices...")
try:
    col_indices = joblib.load(feature_indices_path)
    print(f"✅ Indices chargés:")
    for col, idx in col_indices.items():
        print(f"   - {col} -> index {idx}")
except Exception as e:
    print(f"⚠️ Erreur chargement indices: {e}")
    col_indices = {}

print("="*50)

@app.route('/')
def home():
    """Page d'accueil avec le formulaire"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Fonction qui fait la prédiction"""
    try:
        # Récupérer les données du formulaire
        recency = float(request.form['recency'])
        frequency = float(request.form['frequency'])
        monetary = float(request.form['monetary'])
        age = float(request.form.get('age', 30)) if request.form.get('age') else 30
        
        print(f"\n📝 Données reçues:")
        print(f"   - Recency: {recency}")
        print(f"   - Frequency: {frequency}")
        print(f"   - Monetary: {monetary}")
        print(f"   - Age: {age}")
        
        # Créer le vecteur de features
        n_features = scaler.mean_.shape[0]
        features = np.zeros((1, n_features))
        
        # Remplir avec les indices connus
        if col_indices:
            if 'Recency' in col_indices:
                features[0, col_indices['Recency']] = recency
                print(f"   ✅ Recency -> index {col_indices['Recency']}")
            if 'Frequency' in col_indices:
                features[0, col_indices['Frequency']] = frequency
                print(f"   ✅ Frequency -> index {col_indices['Frequency']}")
            if 'MonetaryTotal' in col_indices:
                features[0, col_indices['MonetaryTotal']] = monetary
                print(f"   ✅ MonetaryTotal -> index {col_indices['MonetaryTotal']}")
            if 'Age' in col_indices:
                features[0, col_indices['Age']] = age
                print(f"   ✅ Age -> index {col_indices['Age']}")
        else:
            # Approche alternative: utiliser les 4 premières colonnes
            print("   ⚠️ Utilisation de l'approche simplifiée")
            features[0, 0] = recency
            features[0, 1] = frequency
            features[0, 2] = monetary
            features[0, 3] = age
        
        # Normaliser les données
        features_scaled = scaler.transform(features)
        
        # Faire la prédiction
        prediction = model.predict(features_scaled)[0]
        proba = model.predict_proba(features_scaled)[0][1]
        
        print(f"\n🎯 Prédiction: {prediction} (probabilité: {proba:.2%})")
        
        # Message personnalisé selon la probabilité
        if proba > 0.7:
            result = f"⚠️ RISQUE CRITIQUE ({proba*100:.1f}%)"
            color = "red"
            advice = "🔴 ACTION URGENTE: Contactez immédiatement ce client avec une offre exclusive!"
        elif proba > 0.5:
            result = f"⚠️ RISQUE ÉLEVÉ ({proba*100:.1f}%)"
            color = "red"
            advice = "🟠 Recommandation: Envoyez une offre personnalisée pour fidéliser ce client."
        elif proba > 0.3:
            result = f"⚠️ RISQUE MODÉRÉ ({proba*100:.1f}%)"
            color = "orange"
            advice = "🟡 Recommandation: Surveillez ce client et envoyez une newsletter engageante."
        else:
            result = f"✅ RISQUE FAIBLE ({proba*100:.1f}%)"
            color = "green"
            advice = "🟢 Recommandation: Client fidèle, programme de fidélité recommandé."
        
        # Afficher le résultat
        return render_template('index.html',
                               prediction=result,
                               color=color,
                               advice=advice,
                               recency=recency,
                               frequency=frequency,
                               monetary=monetary,
                               age=age,
                               proba=f"{proba*100:.1f}")
    
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return render_template('index.html',
                               error=f"Erreur: {str(e)}")

if __name__ == '__main__':
    print("\n🌐 Lancement du serveur...")
    print("📱 Ouvrir http://127.0.0.1:5000 dans votre navigateur")
    print("⏹️  Appuyer sur Ctrl+C pour arrêter\n")
    app.run(debug=True, host='0.0.0.0', port=5000)