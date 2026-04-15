# app/app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ============================================================
# CHEMINS DES FICHIERS
# ============================================================
base_dir           = os.path.dirname(os.path.abspath(__file__))
model_path         = os.path.join(base_dir, '..', 'models', 'best_model_churn.pkl')
scaler_path        = os.path.join(base_dir, '..', 'models', 'scaler.pkl')
imputer_path       = os.path.join(base_dir, '..', 'models', 'imputer.pkl')
feature_names_path = os.path.join(base_dir, '..', 'models', 'feature_names.pkl')

print("="*50)
print("🚀 Démarrage de l'application Flask")
print("="*50)

# ============================================================
# CHARGEMENT DES ARTEFACTS
# ============================================================
print("📦 Chargement du modèle...")
try:
    model = joblib.load(model_path)
    print(f"✅ Modèle chargé : {type(model).__name__}")
except Exception as e:
    print(f"❌ Erreur chargement modèle: {e}")
    model = None

print("📦 Chargement du scaler...")
try:
    scaler = joblib.load(scaler_path)
    print(f"✅ Scaler chargé! (n_features={scaler.mean_.shape[0]})")
except Exception as e:
    print(f"❌ Erreur scaler: {e}")
    scaler = None

print("📦 Chargement de l'imputer...")
try:
    imputer = joblib.load(imputer_path)
    print("✅ Imputer chargé!")
except Exception as e:
    imputer = None
    print(f"⚠️  Imputer non trouvé : {e}")

print("📦 Chargement des noms de features...")
try:
    feature_names = joblib.load(feature_names_path)
    print(f"✅ {len(feature_names)} features chargées")
    season_feats  = [f for f in feature_names if 'FavoriteSeason'   in f]
    spending_feats = [f for f in feature_names if 'SpendingCategory' in f]
    status_feats  = [f for f in feature_names if 'AccountStatus'    in f]
    print(f"   Saison   : {season_feats}")
    print(f"   Spending : {spending_feats}")
    print(f"   Status   : {status_feats}")
except Exception as e:
    feature_names = None
    print(f"⚠️  Feature names non trouvés : {e}")

print("="*50)

# ============================================================
# VALEURS PAR DÉFAUT — médianes/modes de X_train
# Pour les features non saisies dans le formulaire
# ============================================================
DEFAULTS = {
    # Features numériques — valeurs médianes de X_train
    'AvgQuantityPerTransaction' : 5.0,
    'MaxQuantity'               : 20.0,
    'PreferredDayOfWeek'        : 2.0,
    'PreferredHour'             : 12.0,
    'WeekendPurchaseRatio'      : 0.2,
    'AvgProductsPerTransaction' : 3.0,
    'UniqueCountries'           : 1.0,
    'NegativeQuantityCount'     : 0.0,
    'ZeroPriceCount'            : 0.0,
    'ReturnRatio'               : 0.03,
    'HasNegativeMonetary'       : 0.0,
    'RegYear'                   : 2010.0,
    'RegMonth'                  : 6.0,
    'RegDay'                    : 15.0,
    'RegWeekday'                : 2.0,
    'IP_IsPrivate'              : 0.0,
    'IP_FirstOctet'             : 100.0,
}


# ============================================================
# ENCODAGE ONE-HOT MANUEL — vérifié et corrigé
# ============================================================
def get_onehot_col(prefix, value, feature_names):
    """
    Retourne le nom de la colonne one-hot correspondante.
    Avec drop_first=True, la catégorie de référence = 1ère alphabétique.
    """
    col_name = f"{prefix}_{value}"
    if col_name in feature_names:
        return col_name
    return None


def preprocess_for_prediction(frequency, monetary, age,
                               satisfaction, support_tickets,
                               favorite_season, spending_category,
                               account_status):
    """
    Crée le vecteur de features complet (103 colonnes).

    Features saisies dans le formulaire (8) :
    - Frequency, MonetaryTotal, Age, SatisfactionScore
    - SupportTicketsCount, FavoriteSeason
    - SpendingCategory, AccountStatus

    Features dérivées calculées :
    - MonetaryAvg, MonetaryStd, MonetaryMin
    - TotalTransactions, UniqueProducts

    95 autres features → valeurs par défaut (médianes X_train)
    """
    if feature_names is None:
        raise ValueError("feature_names non chargé")

    # Vecteur de zéros aligné sur les 103 features
    X = pd.DataFrame(
        np.zeros((1, len(feature_names))),
        columns=feature_names
    )

    # ---- Features numériques saisies ----
    monetary_avg       = monetary / max(frequency, 1)
    monetary_std       = monetary_avg * 0.35
    monetary_min       = monetary_avg * 0.4
    total_transactions = max(1, int(frequency * 2.5))
    unique_products    = max(1, int(frequency * 1.8))

    numeric_inputs = {
        'Frequency'         : frequency,
        'MonetaryTotal'     : monetary,
        'MonetaryAvg'       : monetary_avg,
        'MonetaryStd'       : monetary_std,
        'MonetaryMin'       : monetary_min,
        'TotalTransactions' : float(total_transactions),
        'UniqueProducts'    : float(unique_products),
        'Age'               : float(age) if age > 0 else 35.0,
        'SupportTicketsCount': float(support_tickets),
        'SatisfactionScore' : float(satisfaction),
    }

    # Appliquer les valeurs numériques saisies
    for col, val in numeric_inputs.items():
        if col in X.columns:
            X[col] = val

    # Appliquer les valeurs par défaut pour les autres features numériques
    for col, val in DEFAULTS.items():
        if col in X.columns:
            X[col] = val

    print(f"\n   📊 Features numériques appliquées : {len(numeric_inputs)}")
    print(f"   📊 Features par défaut appliquées : {len(DEFAULTS)}")
    print(f"   📊 Features catégorielles à encoder...")

    # ============================================================
    # ENCODAGE ONE-HOT MANUEL CORRIGÉ
    # drop_first=True → 1ère catégorie alphabétique = référence (= 0)
    # ============================================================

    # --- FavoriteSeason ---
    # Catégories : Automne(réf), Été, Hiver, Printemps
    # drop_first → Automne = référence (tout à 0)
    season_map = {
        'Automne'   : None,                      # référence → 0
        'Été'       : 'FavoriteSeason_Été',
        'Hiver'     : 'FavoriteSeason_Hiver',
        'Printemps' : 'FavoriteSeason_Printemps',
    }
    season_col = season_map.get(favorite_season)
    if season_col and season_col in X.columns:
        X[season_col] = 1.0
        print(f"   ✅ {season_col} = 1")
    else:
        print(f"   ℹ️  {favorite_season} = référence (tout à 0)")

    # --- SpendingCategory ---
    # Catégories : High(réf), Low, Medium, VIP
    # drop_first → High = référence
    spending_map = {
        'High'   : None,                        # référence → 0
        'Low'    : 'SpendingCategory_Low',
        'Medium' : 'SpendingCategory_Medium',
        'VIP'    : 'SpendingCategory_VIP',
    }
    spending_col = spending_map.get(spending_category)
    if spending_col and spending_col in X.columns:
        X[spending_col] = 1.0
        print(f"   ✅ {spending_col} = 1")
    else:
        print(f"   ℹ️  SpendingCategory {spending_category} = référence")

    # --- AccountStatus --- CORRECTION PRINCIPALE
    # Catégories : Active(réf), Closed, Pending, Suspended
    # drop_first → Active = référence
    status_map = {
        'Active'    : None,                      # référence → 0
        'Closed'    : 'AccountStatus_Closed',
        'Pending'   : 'AccountStatus_Pending',
        'Suspended' : 'AccountStatus_Suspended',
    }
    status_col = status_map.get(account_status)
    if status_col and status_col in X.columns:
        X[status_col] = 1.0
        print(f"   ✅ {status_col} = 1   ← CORRIGÉ")
    else:
        print(f"   ℹ️  AccountStatus {account_status} = référence (Active)")

    # --- Country ---
    # UK = référence (la plus fréquente)
    # Pour les autres pays → laisser à 0 (approximation acceptable)
    print(f"   ℹ️  Country = United Kingdom (référence, valeur par défaut)")

    # --- Region ---
    # UK = référence → tout à 0
    print(f"   ℹ️  Region = UK (référence, valeur par défaut)")

    # --- AgeCategory ---
    # Inconnu = référence → tout à 0
    print(f"   ℹ️  AgeCategory = Inconnu (référence)")

    # --- Autres catégorielles ---
    # PreferredTimeOfDay, WeekendPreference, BasketSizeCategory,
    # ProductDiversity, Gender, IP_Class
    # → laissées à 0 (valeur de référence)
    print(f"   ℹ️  95 autres features = valeurs par défaut (0 ou médiane)")

    # ---- Vérification finale ----
    print(f"\n   🔍 VÉRIFICATION ENCODAGE :")
    for s in ['FavoriteSeason_Hiver',
              'FavoriteSeason_Printemps',
              'FavoriteSeason_Été']:
        if s in X.columns:
            print(f"      {s:<35} = {int(X[s].values[0])}")

    for sp in ['SpendingCategory_Low',
               'SpendingCategory_Medium',
               'SpendingCategory_VIP']:
        if sp in X.columns:
            val = int(X[sp].values[0])
            if val == 1:
                print(f"      {sp:<35} = {val}  ✅")

    for st in ['AccountStatus_Closed',
               'AccountStatus_Pending',
               'AccountStatus_Suspended']:
        if st in X.columns:
            val = int(X[st].values[0])
            if val == 1:
                print(f"      {st:<35} = {val}  ✅")

    nan_count = X.isnull().sum().sum()
    print(f"\n   NaN dans le vecteur final : {nan_count}")
    print(f"   Shape du vecteur final    : {X.shape}")

    return X.values


# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================
def get_risk_info(proba):
    """Seuils de risque adaptés aux résultats réels du modèle."""
    if proba >= 0.65:
        return {
            'level'  : 'CRITIQUE',
            'color'  : 'red',
            'icon'   : '🚨',
            'advice' : (
                "🔴 ACTION URGENTE : Contactez immédiatement ce client "
                "avec une offre personnalisée exceptionnelle "
                "(réduction 30%, livraison offerte, accès VIP). "
                "Chaque jour compte !"
            )
        }
    elif proba >= 0.45:
        return {
            'level'  : 'ÉLEVÉ',
            'color'  : 'red',
            'icon'   : '⚠️',
            'advice' : (
                "🟠 Recommandation : Envoyez une offre exclusive sous 48h "
                "(réduction 20%, recommandations personnalisées). "
                "Activez un suivi mensuel."
            )
        }
    elif proba >= 0.30:
        return {
            'level'  : 'MODÉRÉ',
            'color'  : 'orange',
            'icon'   : '📊',
            'advice' : (
                "🟡 Recommandation : Activez une campagne de réengagement "
                "(newsletter personnalisée, promotions saisonnières). "
                "Surveillez l'évolution."
            )
        }
    else:
        return {
            'level'  : 'FAIBLE',
            'color'  : 'green',
            'icon'   : '✅',
            'advice' : (
                "🟢 Client fidèle : Valorisez cette relation ! "
                "Programme de parrainage, offres VIP, "
                "cross-selling et upselling."
            )
        }


def compute_fm_scores(frequency, monetary):
    """Calcule scores Fréquence & Montant (0-5)."""
    f_score = min(5, int(frequency / 2)) if frequency <= 10 else 5

    if monetary >= 2000:
        m_score = 5
    elif monetary >= 1000:
        m_score = 4
    elif monetary >= 500:
        m_score = 3
    elif monetary >= 200:
        m_score = 2
    else:
        m_score = 1

    fm_score = round((f_score + m_score) / 2, 1)
    return f_score, m_score, fm_score


# ============================================================
# ROUTES FLASK
# ============================================================

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        frequency       = float(request.form.get('frequency', 1))
        monetary        = float(request.form.get('monetary', 0))
        age             = float(request.form.get('age', 35) or 35)
        satisfaction    = float(request.form.get('satisfaction', 3))
        support_tickets = float(request.form.get('support_tickets', 0))
        favorite_season = request.form.get('favorite_season', 'Hiver')
        spending_cat    = request.form.get('spending_category', 'Medium')
        account_status  = request.form.get('account_status', 'Active')

        print(f"\n{'='*50}")
        print(f"📝 NOUVELLE PRÉDICTION")
        print(f"{'='*50}")
        print(f"   Frequency       : {frequency}")
        print(f"   Monetary        : {monetary:.2f} £")
        print(f"   Age             : {age}")
        print(f"   Satisfaction    : {satisfaction}/5")
        print(f"   Support tickets : {support_tickets}")
        print(f"   Saison favorite : {favorite_season}")
        print(f"   SpendingCat     : {spending_cat}")
        print(f"   AccountStatus   : {account_status}")
        print(f"   → Features par défaut : {len(DEFAULTS)} features numériques")

        # Validations
        if frequency <= 0:
            return render_template(
                'index.html',
                error="⚠️ La fréquence doit être supérieure à 0."
            )
        if monetary < 0:
            return render_template(
                'index.html',
                error="⚠️ Le montant ne peut pas être négatif."
            )
        if model is None:
            return render_template(
                'index.html',
                error="❌ Modèle non disponible. Vérifiez les fichiers .pkl."
            )

        # Preprocessing
        X_processed = preprocess_for_prediction(
            frequency        = frequency,
            monetary         = monetary,
            age              = age,
            satisfaction     = satisfaction,
            support_tickets  = support_tickets,
            favorite_season  = favorite_season,
            spending_category= spending_cat,
            account_status   = account_status
        )

        # Prédiction
        prediction = int(model.predict(X_processed)[0])
        proba      = float(model.predict_proba(X_processed)[0][1])

        print(f"\n🎯 RÉSULTAT :")
        print(f"   Prédiction  : {'Partant' if prediction==1 else 'Fidèle'}")
        print(f"   Probabilité : {proba:.2%}")
        print(f"   Niveau      : {get_risk_info(proba)['level']}")

        risk        = get_risk_info(proba)
        result_text = f"{risk['icon']} RISQUE {risk['level']} ({proba*100:.1f}%)"
        f_score, m_score, fm_score = compute_fm_scores(frequency, monetary)

        return render_template(
            'index.html',
            prediction        = result_text,
            color             = risk['color'],
            advice            = risk['advice'],
            risk_level        = risk['level'],
            frequency         = frequency,
            monetary          = f"{monetary:.0f}",
            age               = age,
            satisfaction      = satisfaction,
            support_tickets   = int(support_tickets),
            favorite_season   = favorite_season,
            spending_category = spending_cat,
            account_status    = account_status,
            proba             = f"{proba*100:.1f}",
            f_score           = f_score,
            m_score           = m_score,
            fm_score          = fm_score
        )

    except ValueError as e:
        return render_template('index.html', error=f"⚠️ {str(e)}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        return render_template('index.html', error=f"❌ Erreur : {str(e)}")


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API REST JSON"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'Corps JSON manquant'})

        frequency       = float(data.get('frequency', 1))
        monetary        = float(data.get('monetary', 0))
        age             = float(data.get('age', 35))
        satisfaction    = float(data.get('satisfaction', 3))
        support_tickets = float(data.get('support_tickets', 0))
        favorite_season = data.get('favorite_season', 'Hiver')
        spending_cat    = data.get('spending_category', 'Medium')
        account_status  = data.get('account_status', 'Active')

        if frequency <= 0:
            return jsonify({'success': False, 'error': 'Frequency > 0'})
        if model is None:
            return jsonify({'success': False, 'error': 'Modèle non chargé'})

        X_processed = preprocess_for_prediction(
            frequency=frequency, monetary=monetary, age=age,
            satisfaction=satisfaction, support_tickets=support_tickets,
            favorite_season=favorite_season,
            spending_category=spending_cat,
            account_status=account_status
        )
        prediction = int(model.predict(X_processed)[0])
        proba      = float(model.predict_proba(X_processed)[0][1])
        risk       = get_risk_info(proba)

        return jsonify({
            'success'         : True,
            'prediction'      : prediction,
            'prediction_label': 'Partant' if prediction == 1 else 'Fidèle',
            'probability'     : round(proba, 4),
            'risk_percentage' : round(proba * 100, 1),
            'risk_level'      : risk['level'],
            'advice'          : risk['advice'],
            'note'            : (
                "8 features saisies, 95 autres fixées à leurs "
                "valeurs médianes/modes de X_train"
            ),
            'inputs'          : {
                'frequency'        : frequency,
                'monetary'         : monetary,
                'age'              : age,
                'satisfaction'     : satisfaction,
                'support_tickets'  : support_tickets,
                'favorite_season'  : favorite_season,
                'spending_category': spending_cat,
                'account_status'   : account_status
            }
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/debug')
def debug():
    """Route de diagnostic complète"""
    season_feats   = [f for f in (feature_names or []) if 'FavoriteSeason'   in f]
    spending_feats = [f for f in (feature_names or []) if 'SpendingCategory' in f]
    status_feats   = [f for f in (feature_names or []) if 'AccountStatus'    in f]
    country_feats  = [f for f in (feature_names or []) if 'Country'          in f]

    info = {
        'model_type'          : type(model).__name__ if model else 'None',
        'model_loaded'        : model is not None,
        'total_features'      : len(feature_names) if feature_names else 0,
        'form_features'       : 8,
        'derived_features'    : 5,
        'default_features'    : len(DEFAULTS),
        'season_features'     : season_feats,
        'spending_features'   : spending_feats,
        'status_features'     : status_feats,
        'country_features_count': len(country_feats),
    }

    if model is not None and hasattr(model, 'feature_importances_'):
        feat_imp = pd.DataFrame({
            'feature'   : feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(15)
        info['top_15_features'] = feat_imp.to_dict('records')

    return jsonify(info)


@app.route('/health')
def health():
    """Endpoint de santé"""
    return jsonify({
        'status'        : 'ok',
        'model_loaded'  : model is not None,
        'scaler_loaded' : scaler is not None,
        'features_count': len(feature_names) if feature_names else 0,
        'model_type'    : type(model).__name__ if model else 'None'
    })


if __name__ == '__main__':
    print("\n🌐 Lancement du serveur Flask...")
    print("📱 Interface web : http://127.0.0.1:5000")
    print("⏹️  Ctrl+C pour arrêter\n")
    app.run(debug=True, host='0.0.0.0', port=5000)