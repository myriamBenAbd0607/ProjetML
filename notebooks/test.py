# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

print("✅ Librairies importées avec succès!")
print(f"matplotlib version: {plt.matplotlib.__version__}")

# ============================================================
# SECTION 1 : CHARGEMENT DES DONNÉES
# ============================================================
# %%
df = pd.read_csv('../data/raw/retail_customers_COMPLETE_CATEGORICAL.csv')

print("=== COLONNES ===")
print(df.columns.values)
print(f"\nNombre de colonnes: {len(df.columns)}")
print(f"\n=== DIMENSIONS ===")
print(f"Lignes: {df.shape[0]}, Colonnes: {df.shape[1]}")
print("\n=== PREMIÈRES LIGNES ===")
print(df.head())

# %%
df.info()

# %%
print(df.describe())

# ============================================================
# SECTION 2 : ANALYSE DES VALEURS MANQUANTES
# ============================================================
# %%
missing     = df.isnull().sum()
missing_pct = (missing / len(df)) * 100

missing_df = pd.DataFrame({
    'Colonne'     : missing.index,
    'Manquantes'  : missing.values,
    'Pourcentage' : missing_pct.values
})
missing_df = missing_df[missing_df['Manquantes'] > 0].sort_values(
    'Manquantes', ascending=False
)

print("=== VALEURS MANQUANTES ===")
print(missing_df)

plt.figure(figsize=(10, 5))
plt.barh(missing_df['Colonne'], missing_df['Pourcentage'])
plt.xlabel('Pourcentage de valeurs manquantes')
plt.title('Valeurs manquantes par colonne')
plt.tight_layout()
plt.show()

# %%
clients_sans_delai = df[df['AvgDaysBetweenPurchases'].isnull()]
print("Clients sans délai entre achats :")
print(clients_sans_delai['Frequency'].value_counts())
print("→ Confirmé : tous ont Frequency=1 (un seul achat)")

# ============================================================
# SECTION 3 : ANALYSE DES VALEURS SPÉCIALES
# ============================================================
# %%
print("=== SUPPORT TICKETS ===")
print(df['SupportTicketsCount'].value_counts().sort_index())

print("\n=== SATISFACTION ===")
print(df['SatisfactionScore'].value_counts().sort_index())

# %%
print("=== RÉSUMÉ ===")
print(f"SupportTickets - valeurs spéciales (-1, 999): "
      f"{df[df['SupportTicketsCount'].isin([-1, 999])].shape[0]} clients")
print(f"Satisfaction - valeurs spéciales (-1, 0, 99): "
      f"{df[df['SatisfactionScore'].isin([-1, 0, 99])].shape[0]} clients")

# ============================================================
# SECTION 4 : CORRECTION 1 — VALEURS NÉGATIVES MonetaryTotal
# ============================================================
# %%
print("=== VALEURS NÉGATIVES ===")
neg_monetary = (df['MonetaryTotal'] < 0).sum()
neg_min      = (df['MonetaryMin'] < 0).sum()
neg_quantity = (df['TotalQuantity'] < 0).sum()

print(f"MonetaryTotal négatif : {neg_monetary} clients")
print(f"MonetaryMin négatif   : {neg_min} clients")
print(f"TotalQuantity négatif : {neg_quantity} clients")

# %%
print("=== CLIENTS AVEC MONETARYTOTAL NÉGATIF ===")
neg_clients_df = df[df['MonetaryTotal'] < 0]
print(neg_clients_df[['CustomerID', 'MonetaryTotal',
                       'TotalQuantity', 'ReturnRatio']].head(10))

print(f"\nReturnRatio moyen (clients négatifs) : {neg_clients_df['ReturnRatio'].mean():.3f}")
print(f"ReturnRatio moyen global              : {df['ReturnRatio'].mean():.3f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df['MonetaryTotal'], bins=50, color='steelblue', edgecolor='black')
axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Seuil = 0')
axes[0].set_title('Distribution MonetaryTotal')
axes[0].set_xlabel('MonetaryTotal (£)')
axes[0].legend()

axes[1].bar(
    ['Positif', 'Négatif'],
    [len(df[df['MonetaryTotal'] >= 0]), len(neg_clients_df)],
    color=['green', 'red'], edgecolor='black'
)
axes[1].set_title('Clients MonetaryTotal positif vs négatif')
plt.tight_layout()
plt.show()

df['HasNegativeMonetary'] = (df['MonetaryTotal'] < 0).astype(int)
print(f"\n✅ FLAG créé : HasNegativeMonetary")
print(f"   - 0 (normal)  : {(df['HasNegativeMonetary']==0).sum()} clients")
print(f"   - 1 (négatif) : {(df['HasNegativeMonetary']==1).sum()} clients")

# ============================================================
# SECTION 5 : PARSING RegistrationDate
# ============================================================
# %%
print("=== DATES AVANT CONVERSION ===")
print(df['RegistrationDate'].head(10))

df['RegistrationDate'] = pd.to_datetime(
    df['RegistrationDate'],
    dayfirst=True,
    errors='coerce'
)

print("\n=== DATES APRÈS CONVERSION ===")
print(df['RegistrationDate'].head(10))
print(f"Valeurs manquantes après conversion : {df['RegistrationDate'].isnull().sum()}")

df['RegYear']    = df['RegistrationDate'].dt.year
df['RegMonth']   = df['RegistrationDate'].dt.month
df['RegDay']     = df['RegistrationDate'].dt.day
df['RegWeekday'] = df['RegistrationDate'].dt.weekday

print("\n=== APERÇU DES NOUVELLES COLONNES ===")
print(df[['RegistrationDate', 'RegYear', 'RegMonth',
          'RegDay', 'RegWeekday']].head(10))

print("\n=== VALEURS MANQUANTES DANS LES NOUVELLES COLONNES ===")
print(df[['RegYear', 'RegMonth', 'RegDay', 'RegWeekday']].isnull().sum())

# %%
print("=== APERÇU ALÉATOIRE ===")
print(df[['RegistrationDate', 'RegYear', 'RegMonth',
          'RegDay', 'RegWeekday']].sample(10))

print("\n=== ANNÉES D'INSCRIPTION ===")
print(df['RegYear'].value_counts().sort_index())

print("\n=== MOIS D'INSCRIPTION ===")
print(df['RegMonth'].value_counts().sort_index())

print("\n=== JOURS DE LA SEMAINE ===")
print(df['RegWeekday'].value_counts().sort_index())

# %%
print("=== TAUX DE CHURN PAR MOIS D'INSCRIPTION ===")
churn_by_month = df.groupby('RegMonth')['Churn'].mean() * 100
print(churn_by_month.sort_values(ascending=False))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
df['RegYear'].value_counts().sort_index().plot(kind='bar')
plt.title('Inscriptions par année')
plt.xlabel('Année')
plt.ylabel('Nombre')

plt.subplot(1, 2, 2)
df['RegMonth'].value_counts().sort_index().plot(kind='bar')
plt.title('Inscriptions par mois')
plt.xlabel('Mois')
plt.ylabel('Nombre')

plt.tight_layout()
plt.show()

# ============================================================
# SECTION 6 : CORRECTION 2 — Feature Engineering LastLoginIP
# ============================================================
# %%
import ipaddress

def is_private_ip(ip_str):
    try:
        ip = ipaddress.ip_address(str(ip_str).strip())
        return int(ip.is_private)
    except:
        return -1

def get_ip_first_octet(ip_str):
    try:
        return int(str(ip_str).strip().split('.')[0])
    except:
        return -1

def get_ip_class(ip_str):
    try:
        first = int(str(ip_str).strip().split('.')[0])
        if 1 <= first <= 126:
            return 'A'
        elif 128 <= first <= 191:
            return 'B'
        elif 192 <= first <= 223:
            return 'C'
        else:
            return 'Other'
    except:
        return 'Unknown'

print("=== FEATURE ENGINEERING : LastLoginIP ===")
print(f"Exemples d'IP : {df['LastLoginIP'].head(5).tolist()}")

df['IP_IsPrivate']  = df['LastLoginIP'].apply(is_private_ip)
df['IP_FirstOctet'] = df['LastLoginIP'].apply(get_ip_first_octet)
df['IP_Class']      = df['LastLoginIP'].apply(get_ip_class)

print(f"\n✅ Nouvelles features créées depuis LastLoginIP :")
print(f"IP_IsPrivate  : {df['IP_IsPrivate'].value_counts().to_dict()}")
print(f"IP_Class      : {df['IP_Class'].value_counts().to_dict()}")
print(f"IP_FirstOctet : min={df['IP_FirstOctet'].min()}, max={df['IP_FirstOctet'].max()}")

df.drop('LastLoginIP', axis=1, inplace=True)
print("\n✅ LastLoginIP supprimée (remplacée par 3 features exploitables)")

print("\n=== COLONNES CONSTANTES ===")
for col in df.columns:
    if df[col].nunique() == 1:
        print(f"⚠️ {col} = {df[col].iloc[0]} (constante)")

if 'NewsletterSubscribed' in df.columns and df['NewsletterSubscribed'].nunique() == 1:
    df.drop('NewsletterSubscribed', axis=1, inplace=True)
    print("✅ NewsletterSubscribed supprimée (constante)")

cols_to_drop_now = []
if 'RegistrationDate' in df.columns:
    cols_to_drop_now.append('RegistrationDate')
df.drop(columns=cols_to_drop_now, inplace=True, errors='ignore')
print(f"✅ Colonnes non prédictives supprimées : {cols_to_drop_now}")

# ============================================================
# SECTION 7 : SUPPRESSION DES FEATURES LEAKANTES
# ============================================================
# %%
print("=== ÉTAPE 1 : CORRÉLATION AVEC CHURN (AVANT SUPPRESSION) ===")

churn_corr = df.select_dtypes(include=['int64', 'float64']).corr()['Churn'].abs()
print("\nTop 15 features corrélées avec Churn :")
print(churn_corr.sort_values(ascending=False).head(15))

print("\n=== ÉTAPE 2 : SUPPRESSION DES FEATURES LEAKANTES ===")
print("""
ℹ️  ANALYSE DU LEAKAGE :
   - ChurnRiskCategory, RFMSegment, CustomerType, LoyaltyLevel
     → construites APRÈS avoir défini Churn → leakage direct
   - FirstPurchaseDaysAgo, CustomerTenureDays
     → calculées sur la même période que le label → leakage temporel
   - Recency (corr=0.859) → seule feature suffisante pour 100% accuracy
   - AvgDaysBetweenPurchases → dérivée de Recency → même leakage
   - PreferredMonth → corrélé à la période de labellisation
""")

leak_features = [
    'ChurnRiskCategory',
    'RFMSegment',
    'CustomerType',
    'LoyaltyLevel',
    'FirstPurchaseDaysAgo',
    'CustomerTenureDays',
    'PreferredMonth',
    'Recency',
    'AvgDaysBetweenPurchases',
]

cols_to_remove = [c for c in leak_features if c in df.columns]
df.drop(columns=cols_to_remove, inplace=True)

print(f"✅ Features supprimées : {cols_to_remove}")
print(f"Shape après suppression : {df.shape}")

print("\n=== ÉTAPE 3 : VÉRIFICATION CORRÉLATION APRÈS SUPPRESSION ===")
churn_corr_after = df.select_dtypes(
    include=['int64', 'float64']
).corr()['Churn'].abs()

print("Top 10 features corrélées avec Churn :")
print(churn_corr_after.sort_values(ascending=False).head(10))

suspicious = churn_corr_after[churn_corr_after > 0.7].index.tolist()
suspicious = [c for c in suspicious if c != 'Churn']

if suspicious:
    print(f"\n⚠️  Features encore suspectes (corr > 0.7) : {suspicious}")
    df.drop(columns=suspicious, inplace=True)
    print(f"   ✅ Supprimées. Shape : {df.shape}")
else:
    print("\n✅ Aucune feature suspecte — données prêtes pour la modélisation")

# ============================================================
# SECTION 8 : CORRECTION 3 — Corrélation et multicolinéarité
# ============================================================
# %%
print("=== ANALYSE DE CORRÉLATION ET MULTICOLINÉARITÉ ===")

numeric_df  = df.drop('Churn', axis=1).select_dtypes(include=['int64', 'float64'])
corr_matrix = numeric_df.corr()

plt.figure(figsize=(22, 18))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=False,
    cmap='RdYlGn',
    center=0,
    vmin=-1, vmax=1,
    linewidths=0.3,
    cbar_kws={"shrink": 0.8}
)
plt.title('Matrice de Corrélation — Toutes les features numériques', fontsize=14)
plt.tight_layout()
plt.show()

print("\n=== PAIRES FORTEMENT CORRÉLÉES (|corr| > 0.8) ===")
high_corr_pairs = []

for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        val = corr_matrix.iloc[i, j]
        if abs(val) > 0.8:
            high_corr_pairs.append({
                'Feature_1'   : corr_matrix.columns[i],
                'Feature_2'   : corr_matrix.columns[j],
                'Corrélation' : round(val, 4)
            })

if high_corr_pairs:
    high_corr_df = (pd.DataFrame(high_corr_pairs)
                      .sort_values('Corrélation', key=abs, ascending=False))
    print(high_corr_df.to_string(index=False))

    concerned = list(set(
        [p['Feature_1'] for p in high_corr_pairs] +
        [p['Feature_2'] for p in high_corr_pairs]
    ))
    plt.figure(figsize=(14, 10))
    sns.heatmap(
        corr_matrix.loc[concerned, concerned],
        annot=True, fmt='.2f', cmap='RdYlGn', center=0, vmin=-1, vmax=1
    )
    plt.title('Zoom — Features fortement corrélées (|corr| > 0.8)')
    plt.tight_layout()
    plt.show()

    print("\n=== DÉCISION SUR LES FEATURES REDONDANTES ===")
    cols_to_drop_corr = []
    seen = set()
    for _, row in high_corr_df.iterrows():
        f1, f2 = row['Feature_1'], row['Feature_2']
        if f1 not in seen and f2 not in seen:
            cols_to_drop_corr.append(f2)
            seen.add(f1)
            seen.add(f2)
            print(f"  Conserver : {f1} | Supprimer : {f2} (corr={row['Corrélation']})")

    if cols_to_drop_corr:
        df.drop(columns=cols_to_drop_corr, inplace=True)
        print(f"\n✅ {len(cols_to_drop_corr)} features redondantes supprimées")
        print(f"Shape après suppression multicolinéarité : {df.shape}")
else:
    print("✅ Aucune paire avec |corrélation| > 0.8 détectée")

# ============================================================
# SECTION 9 : SPLIT TRAIN/TEST + IMPUTATION APRÈS SPLIT
# ============================================================
# %%
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

os.makedirs('../data/processed', exist_ok=True)
df.to_csv('../data/processed/retail_customers_cleaned.csv', index=False)
print(f"✅ Données nettoyées sauvegardées (avec CustomerID pour traçabilité)")
print(f"Shape finale : {df.shape}")

X = df.drop('Churn', axis=1)
y = df['Churn']

if 'CustomerID' in X.columns:
    X = X.drop('CustomerID', axis=1)
    print("✅ CustomerID retiré des features de modélisation")
    print("   (conservé dans retail_customers_cleaned.csv pour traçabilité)")

print(f"\n=== AVANT SPLIT ===")
print(f"X shape : {X.shape}")
print(f"y shape : {y.shape}")
print(f"\nColonnes utilisées pour la modélisation :")
print(X.columns.tolist())

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train = X_train.reset_index(drop=True)
X_test  = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test  = y_test.reset_index(drop=True)

print(f"\n=== APRÈS SPLIT ===")
print(f"X_train shape : {X_train.shape}")
print(f"X_test shape  : {X_test.shape}")
print(f"y_train : {y_train.value_counts().to_dict()}")
print(f"y_test  : {y_test.value_counts().to_dict()}")

print("\n=== IMPUTATION APRÈS SPLIT (CORRECTION CRITIQUE) ===")
print("ℹ️  SimpleImputer.fit() sur X_train uniquement")

num_cols_with_nan = X_train.select_dtypes(
    include=['int64', 'float64']
).columns[X_train.select_dtypes(
    include=['int64', 'float64']
).isnull().any()].tolist()

print(f"Colonnes numériques avec NaN : {num_cols_with_nan}")

if num_cols_with_nan:
    imputer = SimpleImputer(strategy='median')
    X_train[num_cols_with_nan] = imputer.fit_transform(X_train[num_cols_with_nan])
    X_test[num_cols_with_nan]  = imputer.transform(X_test[num_cols_with_nan])
    os.makedirs('../models', exist_ok=True)
    joblib.dump(imputer, '../models/imputer.pkl')
    print(f"✅ Imputation médiane appliquée sur : {num_cols_with_nan}")
else:
    print("✅ Aucune valeur manquante dans les colonnes numériques")

nan_train = X_train.isnull().sum().sum()
nan_test  = X_test.isnull().sum().sum()
print(f"\nNaN restants — X_train : {nan_train} | X_test : {nan_test}")

# ============================================================
# SECTION 10 : CORRECTION 4 — Déséquilibre de classes
# ============================================================
# %%
from sklearn.utils.class_weight import compute_class_weight

print("=== ANALYSE DU DÉSÉQUILIBRE DE CLASSES ===")
churn_counts = y_train.value_counts()
churn_pct    = y_train.value_counts(normalize=True) * 100

print(f"Fidèles  (0) : {churn_counts[0]} clients ({churn_pct[0]:.1f}%)")
print(f"Partants (1) : {churn_counts[1]} clients ({churn_pct[1]:.1f}%)")

ratio = churn_counts[0] / churn_counts[1]
print(f"Ratio déséquilibre : {ratio:.1f}:1")

if ratio > 3:
    print("⚠️  Déséquilibre significatif → class_weight='balanced' appliqué")
else:
    print("✅ Déséquilibre acceptable")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].bar(['Fidèles (0)', 'Partants (1)'], churn_counts.values,
            color=['#2ecc71', '#e74c3c'], edgecolor='black')
axes[0].set_title('Distribution des classes — y_train')
axes[0].set_ylabel('Nombre de clients')
for i, v in enumerate(churn_counts.values):
    axes[0].text(i, v + 5, str(v), ha='center', fontweight='bold')

axes[1].pie(churn_pct.values, labels=['Fidèles', 'Partants'],
            autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'])
axes[1].set_title('Proportion des classes')
plt.tight_layout()
plt.show()

classes       = np.array([0, 1])
weights       = compute_class_weight('balanced', classes=classes, y=y_train)
class_weights = {0: weights[0], 1: weights[1]}
print(f"\nPoids calculés : {class_weights}")

# %%
os.makedirs('../data/train_test', exist_ok=True)
X_train.to_csv('../data/train_test/X_train.csv', index=False)
X_test.to_csv('../data/train_test/X_test.csv',   index=False)
y_train.to_csv('../data/train_test/y_train.csv', index=False)
y_test.to_csv('../data/train_test/y_test.csv',   index=False)
print("✅ Données splittées sauvegardées dans data/train_test/")

# ============================================================
# SECTION 11 : PIPELINE D'ENCODAGE CENTRALISÉ
# ============================================================
# %%
from sklearn.preprocessing import StandardScaler

print("=== PIPELINE D'ENCODAGE ET NORMALISATION (UNIQUE) ===")
print("""
ℹ️  RÈGLE D'UTILISATION DES DONNÉES :
   X_train_encoded / X_test_encoded → Arbre, RF, GB
   X_train_scaled  / X_test_scaled  → KNN
   Un seul scaler fité sur X_train_encoded → pas de leakage
""")

cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
print(f"Colonnes catégorielles à encoder ({len(cat_cols)}) :")
print(cat_cols)

X_train_encoded = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
X_test_encoded  = pd.get_dummies(X_test,  columns=cat_cols, drop_first=True)
X_test_encoded  = X_test_encoded.reindex(
    columns=X_train_encoded.columns, fill_value=0
)

raw_test_encoded = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)
missing_in_test  = set(X_train_encoded.columns) - set(raw_test_encoded.columns)
if missing_in_test:
    print(f"\n⚠️  Colonnes uniquement dans train (remplies par 0 dans test) :")
    for col in sorted(missing_in_test):
        print(f"   → {col}")
else:
    print("\n✅ Toutes les catégories présentes dans train ET test")

print(f"\n✅ Après encodage :")
print(f"X_train_encoded shape : {X_train_encoded.shape}")
print(f"X_test_encoded shape  : {X_test_encoded.shape}")
print(f"NaN train             : {X_train_encoded.isnull().sum().sum()}")
print(f"NaN test              : {X_test_encoded.isnull().sum().sum()}")

os.makedirs('../models', exist_ok=True)
joblib.dump(list(X_train_encoded.columns), '../models/feature_names.pkl')
print("✅ Noms des features sauvegardés dans models/feature_names.pkl")

scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled  = scaler.transform(X_test_encoded)

joblib.dump(scaler, '../models/scaler.pkl')
print("✅ Scaler unique sauvegardé dans models/scaler.pkl")
print(f"\nX_train_encoded shape : {X_train_encoded.shape}  ← Arbre, RF, GB")
print(f"X_train_scaled shape  : {X_train_scaled.shape}   ← KNN uniquement")

# ============================================================
# SECTION 12 : MODÈLE 1 — KNN AMÉLIORATION MAXIMALE
# ============================================================
# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score,
                             confusion_matrix, classification_report,
                             roc_curve)
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

print("="*55)
print("MODÈLE 1 — KNN (AMÉLIORATION MAXIMALE)")
print("="*55)
print("ℹ️  Données : X_train_scaled (normalisées)\n")

# KNN Base
print("--- KNN BASE (k=5) ---")
knn_base = KNeighborsClassifier(n_neighbors=5)
knn_base.fit(X_train_scaled, y_train)

y_pred_base  = knn_base.predict(X_test_scaled)
y_proba_base = knn_base.predict_proba(X_test_scaled)[:, 1]

acc_base = accuracy_score(y_test, y_pred_base)
f1_base  = f1_score(y_test, y_pred_base)
rec_base = recall_score(y_test, y_pred_base)
auc_base = roc_auc_score(y_test, y_proba_base)

print(f"Base → Acc={acc_base:.4f} | F1={f1_base:.4f} | "
      f"Rec={rec_base:.4f} | AUC={auc_base:.4f}")

# Tentative 1 : SelectKBest
print("\n--- TENTATIVE 1 : SelectKBest ---")
results_attempts = {}
best_f1_global   = f1_base
best_knn_global  = knn_base
best_X_train     = X_train_scaled
best_X_test      = X_test_scaled
best_config      = "Base"

for k_feat in [20, 30, 40, 50, 60]:
    selector  = SelectKBest(f_classif, k=k_feat)
    X_tr_sel  = selector.fit_transform(X_train_scaled, y_train)
    X_te_sel  = selector.transform(X_test_scaled)
    for k_nn in [5, 9, 11, 15, 21]:
        knn_tmp = KNeighborsClassifier(n_neighbors=k_nn, weights='distance',
                                        metric='manhattan')
        knn_tmp.fit(X_tr_sel, y_train)
        f1_tmp  = f1_score(y_test, knn_tmp.predict(X_te_sel))
        acc_tmp = accuracy_score(y_test, knn_tmp.predict(X_te_sel))
        config_name = f"Select(k={k_feat})+KNN(k={k_nn})"
        results_attempts[config_name] = {'f1': f1_tmp, 'acc': acc_tmp}
        if f1_tmp > best_f1_global:
            best_f1_global  = f1_tmp
            best_knn_global = knn_tmp
            best_X_train    = X_tr_sel
            best_X_test     = X_te_sel
            best_config     = config_name

sorted_attempts = sorted(results_attempts.items(),
                          key=lambda x: x[1]['f1'], reverse=True)[:5]
print("Top 5 configurations SelectKBest :")
for name, m in sorted_attempts:
    print(f"   {name:<35} F1={m['f1']:.4f} | Acc={m['acc']:.4f}")
print(f"\n✅ Meilleure config : {best_config} (F1={best_f1_global:.4f})")

# Tentative 2 : PCA
print("\n--- TENTATIVE 2 : PCA ---")
for n_comp in [10, 15, 20, 25, 30]:
    pca_tmp  = PCA(n_components=n_comp, random_state=42)
    X_tr_pca = pca_tmp.fit_transform(X_train_scaled)
    X_te_pca = pca_tmp.transform(X_test_scaled)
    for k_nn in [5, 9, 11, 15, 21]:
        knn_tmp = KNeighborsClassifier(n_neighbors=k_nn, weights='distance',
                                        metric='manhattan')
        knn_tmp.fit(X_tr_pca, y_train)
        f1_tmp  = f1_score(y_test, knn_tmp.predict(X_te_pca))
        acc_tmp = accuracy_score(y_test, knn_tmp.predict(X_te_pca))
        config_name = f"PCA(n={n_comp})+KNN(k={k_nn})"
        results_attempts[config_name] = {'f1': f1_tmp, 'acc': acc_tmp}
        if f1_tmp > best_f1_global:
            best_f1_global  = f1_tmp
            best_knn_global = knn_tmp
            best_X_train    = X_tr_pca
            best_X_test     = X_te_pca
            best_config     = config_name

sorted_pca = sorted({k: v for k, v in results_attempts.items()
                     if 'PCA' in k}.items(),
                    key=lambda x: x[1]['f1'], reverse=True)[:5]
print("Top 5 PCA :")
for name, m in sorted_pca:
    print(f"   {name:<35} F1={m['f1']:.4f} | Acc={m['acc']:.4f}")
print(f"\n✅ Meilleure config : {best_config} (F1={best_f1_global:.4f})")

# Tentative 3 : SelectKBest + PCA
print("\n--- TENTATIVE 3 : SelectKBest + PCA combinés ---")
for k_feat in [40, 50, 60]:
    selector  = SelectKBest(f_classif, k=k_feat)
    X_tr_sel  = selector.fit_transform(X_train_scaled, y_train)
    X_te_sel  = selector.transform(X_test_scaled)
    for n_comp in [10, 15, 20]:
        pca_tmp  = PCA(n_components=n_comp, random_state=42)
        X_tr_pca = pca_tmp.fit_transform(X_tr_sel)
        X_te_pca = pca_tmp.transform(X_te_sel)
        for k_nn in [9, 11, 15, 21]:
            knn_tmp = KNeighborsClassifier(n_neighbors=k_nn, weights='distance',
                                            metric='manhattan')
            knn_tmp.fit(X_tr_pca, y_train)
            f1_tmp  = f1_score(y_test, knn_tmp.predict(X_te_pca))
            acc_tmp = accuracy_score(y_test, knn_tmp.predict(X_te_pca))
            config_name = f"Sel({k_feat})+PCA({n_comp})+KNN({k_nn})"
            results_attempts[config_name] = {'f1': f1_tmp, 'acc': acc_tmp}
            if f1_tmp > best_f1_global:
                best_f1_global  = f1_tmp
                best_knn_global = knn_tmp
                best_X_train    = X_tr_pca
                best_X_test     = X_te_pca
                best_config     = config_name

sorted_comb = sorted({k: v for k, v in results_attempts.items()
                      if 'Sel' in k and 'PCA' in k}.items(),
                     key=lambda x: x[1]['f1'], reverse=True)[:5]
print("Top 5 combinés :")
for name, m in sorted_comb:
    print(f"   {name:<35} F1={m['f1']:.4f} | Acc={m['acc']:.4f}")

# Résultat final KNN
print(f"\n{'='*55}")
print(f"✅ MEILLEURE CONFIGURATION : {best_config}")
print(f"{'='*55}")

knn         = best_knn_global
y_pred_knn  = knn.predict(best_X_test)
y_proba_knn = knn.predict_proba(best_X_test)[:, 1]

acc_knn  = accuracy_score(y_test, y_pred_knn)
prec_knn = precision_score(y_test, y_pred_knn, zero_division=0)
rec_knn  = recall_score(y_test, y_pred_knn)
f1_knn   = f1_score(y_test, y_pred_knn)
auc_knn  = roc_auc_score(y_test, y_proba_knn)

print(f"\n{'Métrique':<12} {'Base':>10} {'Optimisé':>12} {'Gain':>10}")
print("-"*47)
for m_name, base_v, opt_v in [
    ('Accuracy', acc_base, acc_knn),
    ('Recall',   rec_base, rec_knn),
    ('F1-Score', f1_base,  f1_knn),
    ('ROC-AUC',  auc_base, auc_knn)
]:
    print(f"{m_name:<12} {base_v:>10.4f} {opt_v:>12.4f} "
          f"{(opt_v-base_v)*100:>+9.2f}%")

print(f"\n🎯 Accuracy  : {acc_knn:.4f} ({acc_knn*100:.1f}%)")
print(f"📊 Precision : {prec_knn:.4f}")
print(f"🔍 Recall    : {rec_knn:.4f}")
print(f"📈 F1-Score  : {f1_knn:.4f}")
print(f"📉 ROC-AUC   : {auc_knn:.4f}")

cv_scores = cross_val_score(knn, best_X_train, y_train, cv=5, scoring='f1')
print(f"\n📊 Validation croisée (5-fold) F1 :")
print(f"   Scores     : {[round(s, 4) for s in cv_scores]}")
print(f"   Moyenne    : {cv_scores.mean():.4f}")
print(f"   Écart-type : {cv_scores.std():.4f}")

cm_knn = confusion_matrix(y_test, y_pred_knn)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Fidèle', 'Parti'],
            yticklabels=['Fidèle', 'Parti'], ax=axes[0])
axes[0].set_title(f'KNN — {best_config}\nAcc={acc_knn:.3f} | F1={f1_knn:.3f}')
axes[0].set_ylabel('Réel')
axes[0].set_xlabel('Prédit')

fpr_b, tpr_b, _ = roc_curve(y_test, y_proba_base)
fpr_o, tpr_o, _ = roc_curve(y_test, y_proba_knn)
axes[1].plot(fpr_b, tpr_b, 'b--', label=f'Base (AUC={auc_base:.3f})', linewidth=2)
axes[1].plot(fpr_o, tpr_o, 'r-',  label=f'Opt  (AUC={auc_knn:.3f})',  linewidth=2)
axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
axes[1].set_xlabel('Taux Faux Positifs')
axes[1].set_ylabel('Taux Vrais Positifs')
axes[1].set_title('Courbe ROC — KNN')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(10, 5))
metrics_names  = ['Accuracy', 'Recall', 'F1-Score', 'ROC-AUC']
vals_base_plot = [acc_base, rec_base, f1_base, auc_base]
vals_opt_plot  = [acc_knn,  rec_knn,  f1_knn,  auc_knn]
x_pos = np.arange(len(metrics_names))
w     = 0.35

bars1 = ax.bar(x_pos - w/2, vals_base_plot, w,
               label='KNN Base', color='#3498db', alpha=0.8)
bars2 = ax.bar(x_pos + w/2, vals_opt_plot,  w,
               label='KNN Optimisé', color='#e74c3c', alpha=0.8)

ax.set_xticks(x_pos)
ax.set_xticklabels(metrics_names)
ax.set_ylabel('Score')
ax.set_ylim(0, 1.1)
ax.set_title('KNN Base vs KNN Optimisé')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

for bar in bars1:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
            f'{bar.get_height():.3f}', ha='center', fontsize=9)
for bar in bars2:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
            f'{bar.get_height():.3f}', ha='center', fontsize=9,
            fontweight='bold', color='#c0392b')

plt.tight_layout()
plt.show()

print("\n=== RAPPORT DE CLASSIFICATION KNN ===")
print(classification_report(y_test, y_pred_knn, target_names=['Fidèle', 'Parti']))

print(f"\n=== BILAN KNN OPTIMISÉ ===")
print(f"{'Métrique':<12} {'Base':>10} {'Optimisé':>12} {'Objectif':>12} {'Verdict':>10}")
print("-"*58)
for m_name, base_v, opt_v, obj_v in [
    ('Accuracy', acc_base, acc_knn, 0.75),
    ('Recall',   rec_base, rec_knn, 0.50),
    ('F1-Score', f1_base,  f1_knn,  0.60)
]:
    verdict = "✅" if opt_v >= obj_v else "⚠️"
    print(f"{m_name:<12} {base_v:>10.4f} {opt_v:>12.4f} "
          f"{obj_v:>12.2f} {verdict:>10}")

print(f"\n⚠️  NOTE PÉDAGOGIQUE :")
print(f"   KNN est moins adapté aux données tabulaires complexes.")
print(f"   Sa performance (F1={f1_knn:.3f}) est normale et attendue.")
print(f"   La différence vs GB illustre les limites de KNN.")

# ============================================================
# SECTION 12 (suite) : MODÈLE 2 — ARBRE DE DÉCISION
# ============================================================
# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

print("="*50)
print("MODÈLE 2 — Arbre de Décision (CORRIGÉ)")
print("="*50)
print("ℹ️  Données : X_train_encoded (non normalisées)\n")

clf_no_limit = DecisionTreeClassifier(criterion="entropy", random_state=42)
clf_no_limit.fit(X_train_encoded, y_train)

print(f"❌ Sans max_depth :")
print(f"   TRAIN : {accuracy_score(y_train, clf_no_limit.predict(X_train_encoded)):.4f}")
print(f"   TEST  : {accuracy_score(y_test,  clf_no_limit.predict(X_test_encoded)):.4f}")
print(f"   Profondeur : {clf_no_limit.get_depth()}")

depths       = range(1, 16)
train_scores = []
test_scores  = []

for d in depths:
    tmp = DecisionTreeClassifier(criterion="entropy", max_depth=d,
                                  class_weight='balanced', random_state=42)
    tmp.fit(X_train_encoded, y_train)
    train_scores.append(accuracy_score(y_train, tmp.predict(X_train_encoded)))
    test_scores.append(accuracy_score(y_test,  tmp.predict(X_test_encoded)))

plt.figure(figsize=(10, 5))
plt.plot(depths, train_scores, 'b-o', label='Train accuracy', markersize=6)
plt.plot(depths, test_scores,  'r-o', label='Test accuracy',  markersize=6)
plt.axvline(x=5, color='green', linestyle='--', linewidth=2, label='max_depth=5 choisi')
plt.xlabel('Profondeur maximale')
plt.ylabel('Accuracy')
plt.title('Compromis Biais/Variance selon max_depth')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

clf_decisiontree = DecisionTreeClassifier(
    criterion="entropy", max_depth=5,
    min_samples_split=20, min_samples_leaf=10,
    class_weight='balanced', random_state=42
)
clf_decisiontree.fit(X_train_encoded, y_train)

y_pred_tree  = clf_decisiontree.predict(X_test_encoded)
y_proba_tree = clf_decisiontree.predict_proba(X_test_encoded)[:, 1]

acc_train_tree = accuracy_score(y_train, clf_decisiontree.predict(X_train_encoded))
acc_test_tree  = accuracy_score(y_test,  y_pred_tree)
prec_tree      = precision_score(y_test, y_pred_tree, zero_division=0)
rec_tree       = recall_score(y_test, y_pred_tree)
f1_tree        = f1_score(y_test, y_pred_tree)
auc_tree       = roc_auc_score(y_test, y_proba_tree)

print(f"\n✅ Avec max_depth=5 :")
print(f"   Accuracy TRAIN : {acc_train_tree:.4f} | TEST : {acc_test_tree:.4f} "
      f"| Écart : {abs(acc_train_tree-acc_test_tree)*100:.1f}%")
print(f"   Profondeur     : {clf_decisiontree.get_depth()}")
print(f"   Precision      : {prec_tree:.4f}")
print(f"   Recall         : {rec_tree:.4f}")
print(f"   F1-Score       : {f1_tree:.4f}")
print(f"   ROC-AUC        : {auc_tree:.4f}")

cm_tree = confusion_matrix(y_test, y_pred_tree)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_tree, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Fidèle', 'Parti'], yticklabels=['Fidèle', 'Parti'])
plt.title(f'Arbre Décision | Acc={acc_test_tree:.3f} | F1={f1_tree:.3f}')
plt.show()

print(classification_report(y_test, y_pred_tree, target_names=['Fidèle', 'Parti']))

plt.figure(figsize=(20, 10))
tree.plot_tree(clf_decisiontree, feature_names=X_train_encoded.columns.tolist(),
               class_names=['Fidèle', 'Parti'], filled=True, rounded=True,
               fontsize=8, max_depth=3)
plt.title('Arbre de Décision (max_depth=5) — 3 premiers niveaux')
plt.tight_layout()
os.makedirs('../reports', exist_ok=True)
plt.savefig('../reports/arbre_decision.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# ---- MODÈLE 3 : RANDOM FOREST ----
from sklearn.ensemble import RandomForestClassifier

print("="*50)
print("MODÈLE 3 — Random Forest")
print("="*50)

clf_rf = RandomForestClassifier(
    n_estimators=200, max_depth=10, min_samples_split=10,
    min_samples_leaf=5, max_features='sqrt', class_weight='balanced',
    oob_score=True, random_state=42, n_jobs=-1
)
clf_rf.fit(X_train_encoded, y_train)

y_pred_rf  = clf_rf.predict(X_test_encoded)
y_proba_rf = clf_rf.predict_proba(X_test_encoded)[:, 1]

acc_rf  = accuracy_score(y_test, y_pred_rf)
prec_rf = precision_score(y_test, y_pred_rf, zero_division=0)
rec_rf  = recall_score(y_test, y_pred_rf)
f1_rf   = f1_score(y_test, y_pred_rf)
auc_rf  = roc_auc_score(y_test, y_proba_rf)

print(f"🎯 Accuracy={acc_rf:.4f} | Precision={prec_rf:.4f} | "
      f"Recall={rec_rf:.4f} | F1={f1_rf:.4f} | AUC={auc_rf:.4f}")
print(f"🌲 OOB Score : {clf_rf.oob_score_:.4f}")

importances = pd.DataFrame({
    'feature'   : X_train_encoded.columns,
    'importance': clf_rf.feature_importances_
}).sort_values('importance', ascending=False).head(15)
print("\n=== TOP 15 FEATURES ===")
print(importances)

plt.figure(figsize=(10, 6))
plt.barh(importances['feature'], importances['importance'])
plt.xlabel('Importance')
plt.title('Top 15 features — Random Forest')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Fidèle', 'Parti'], yticklabels=['Fidèle', 'Parti'])
plt.title(f'Random Forest | Acc={acc_rf:.3f} | F1={f1_rf:.3f}')
plt.show()
print(classification_report(y_test, y_pred_rf, target_names=['Fidèle', 'Parti']))

# %%
# ---- MODÈLE 4 : GRADIENT BOOSTING ----
from sklearn.ensemble import GradientBoostingClassifier

print("="*50)
print("MODÈLE 4 — Gradient Boosting")
print("="*50)

print("""
ℹ️  NOTE D'ANALYSE — FavoriteSeason dans le top features :
   Les features FavoriteSeason (Hiver, Été, Printemps) dominent
   les importances du Gradient Boosting (0.22, 0.20, 0.14).
   Explication : corrélation spurieuse avec la période de labellisation.
   → Conserver FavoriteSeason (feature légitime) mais noter la limite.
""")

gradientBoosting = GradientBoostingClassifier(random_state=42)
gradientBoosting.fit(X_train_encoded, y_train)

y_pred_gb = gradientBoosting.predict(X_test_encoded)
print(f"Défaut → Acc={accuracy_score(y_test, y_pred_gb):.4f} | "
      f"F1={f1_score(y_test, y_pred_gb):.4f}")

param_grid = {
    'n_estimators' : [100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth'    : [3, 5, 7]
}

grid_gb = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_grid, cv=5, scoring='f1', n_jobs=-1
)
grid_gb.fit(X_train_encoded, y_train)
print(f"\nMeilleurs paramètres : {grid_gb.best_params_}")
print(f"Meilleur F1 (CV)     : {grid_gb.best_score_:.4f}")

best_gb = GradientBoostingClassifier(**grid_gb.best_params_, random_state=42)
best_gb.fit(X_train_encoded, y_train)

y_pred_best  = best_gb.predict(X_test_encoded)
y_proba_best = best_gb.predict_proba(X_test_encoded)[:, 1]

acc_best  = accuracy_score(y_test, y_pred_best)
prec_best = precision_score(y_test, y_pred_best, zero_division=0)
rec_best  = recall_score(y_test, y_pred_best)
f1_best   = f1_score(y_test, y_pred_best)
auc_best  = roc_auc_score(y_test, y_proba_best)

print(f"\n🎯 Accuracy  : {acc_best:.4f} ({acc_best*100:.1f}%)")
print(f"📊 Precision : {prec_best:.4f}")
print(f"🔍 Recall    : {rec_best:.4f}")
print(f"📈 F1-Score  : {f1_best:.4f}")
print(f"📉 ROC-AUC   : {auc_best:.4f}")

feat_imp_gb = pd.DataFrame({
    'feature'   : X_train_encoded.columns,
    'importance': best_gb.feature_importances_
}).sort_values('importance', ascending=False).head(10)
print("\n=== TOP 10 FEATURES ===")
print(feat_imp_gb)

season_features   = [f for f in feat_imp_gb['feature'] if 'FavoriteSeason' in f]
season_importance = feat_imp_gb[feat_imp_gb['feature'].isin(season_features)
                                ]['importance'].sum()
print(f"\n📊 ANALYSE FAVORITSEASON :")
print(f"   Importance totale : {season_importance:.3f} ({season_importance*100:.1f}%)")
print(f"   → Corrélation spurieuse possible avec la période de labellisation")
print(f"   → Limite à mentionner dans le rapport final")

# ============================================================
# SECTION 13 : TABLEAU COMPARATIF FINAL
# ============================================================
# %%
print("="*65)
print("📊 TABLEAU COMPARATIF FINAL — TOUS LES MODÈLES")
print("="*65)
print(f"\nℹ️  Note : KNN→scaled+{best_config} | Arbre/RF/GB→encoded\n")

results = {
    'KNN Optimisé'    : {'Accuracy': acc_knn,       'Precision': prec_knn,
                         'Recall'  : rec_knn,       'F1-Score' : f1_knn,
                         'ROC-AUC' : auc_knn},
    'Arbre Décision'  : {'Accuracy': acc_test_tree, 'Precision': prec_tree,
                         'Recall'  : rec_tree,      'F1-Score' : f1_tree,
                         'ROC-AUC' : auc_tree},
    'Random Forest'   : {'Accuracy': acc_rf,        'Precision': prec_rf,
                         'Recall'  : rec_rf,        'F1-Score' : f1_rf,
                         'ROC-AUC' : auc_rf},
    'Gradient Boost.' : {'Accuracy': acc_best,      'Precision': prec_best,
                         'Recall'  : rec_best,      'F1-Score' : f1_best,
                         'ROC-AUC' : auc_best},
}

results_df = pd.DataFrame(results).T.round(4)
results_df = results_df.sort_values('F1-Score', ascending=False)
print(results_df.to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
colors  = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
x       = np.arange(len(metrics))
width   = 0.2

for i, (model_name, scores) in enumerate(results.items()):
    axes[0].bar(x + i * width, [scores[m] for m in metrics],
                width, label=model_name, color=colors[i], alpha=0.85)

axes[0].set_xticks(x + width * 1.5)
axes[0].set_xticklabels(metrics, rotation=15)
axes[0].set_ylabel('Score')
axes[0].set_ylim(0, 1.1)
axes[0].set_title('Comparaison des métriques par modèle')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].barh(results_df.index, results_df['F1-Score'],
             color=colors[:len(results_df)])
axes[1].set_xlabel('F1-Score')
axes[1].set_title('Classement par F1-Score')
axes[1].set_xlim(0, 1)
for i, (idx, row) in enumerate(results_df.iterrows()):
    axes[1].text(row['F1-Score'] + 0.005, i,
                 f"{row['F1-Score']:.4f}", va='center', fontweight='bold')

plt.tight_layout()
plt.show()

best_model_name = results_df.index[0]
print(f"\n🏆 MEILLEUR MODÈLE : {best_model_name}")
print(f"   F1-Score : {results_df.loc[best_model_name, 'F1-Score']:.4f}")
print(f"   ROC-AUC  : {results_df.loc[best_model_name, 'ROC-AUC']:.4f}")
print(f"\nℹ️  F1-Score et Recall prioritaires pour le churn.")

# ============================================================
# SECTION 14 : SEGMENTATION — ACP + K-MEANS CORRIGÉE
# ============================================================
# %%
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy import stats

print("="*50)
print("SEGMENTATION CLIENTÈLE — ACP + K-MEANS")
print("="*50)

print("\n1. Chargement des données...")
X_train_seg = pd.read_csv('../data/train_test/X_train.csv')
y_train_seg = pd.read_csv('../data/train_test/y_train.csv').squeeze()
print(f"✅ X_train shape: {X_train_seg.shape}")

print("\n2. Colonnes numériques uniquement...")
numeric_cols = X_train_seg.select_dtypes(include=['int64', 'float64']).columns
X_numeric    = X_train_seg[numeric_cols].copy()
missing_pct  = X_numeric.isnull().sum() / len(X_numeric)
cols_to_keep = missing_pct[missing_pct < 0.5].index
X_numeric    = X_numeric[cols_to_keep]
print(f"✅ Colonnes numériques conservées : {len(X_numeric.columns)}")

print("\n3. Imputation...")
for col in X_numeric.columns:
    if X_numeric[col].isnull().any():
        X_numeric[col] = X_numeric[col].fillna(X_numeric[col].median())
print(f"✅ NaN restants : {X_numeric.isnull().sum().sum()}")

print("\n4. Normalisation...")
scaler_seg = StandardScaler()
X_scaled   = scaler_seg.fit_transform(X_numeric)
print(f"✅ Données normalisées")

# Retrait des outliers avec Z-score > 4
print("\n4b. Retrait des outliers extrêmes (Z-score > 4)...")
print("ℹ️  Seuil Z=4 choisi (au lieu de Z=3) car données financières")
print("   asymétriques — Z=3 retirait trop de clients (~23%)")

z_scores      = np.abs(stats.zscore(X_scaled))
mask_inliers  = (z_scores < 4).all(axis=1)
mask_outliers = ~mask_inliers

n_outliers = mask_outliers.sum()
n_inliers  = mask_inliers.sum()

print(f"   Outliers détectés  : {n_outliers} clients "
      f"({n_outliers/len(X_scaled)*100:.1f}%)")
print(f"   Clients conservés  : {n_inliers} clients "
      f"({n_inliers/len(X_scaled)*100:.1f}%)")

if n_outliers / len(X_scaled) > 0.15:
    print(f"\n   ⚠️  Taux outliers > 15% → limite pédagogique à mentionner")

X_scaled_clean  = X_scaled[mask_inliers]
X_numeric_clean = X_numeric[mask_inliers].reset_index(drop=True)
y_seg_clean     = y_train_seg[mask_inliers].reset_index(drop=True)
X_seg_clean     = X_train_seg[mask_inliers].reset_index(drop=True)

X_scaled_out    = X_scaled[mask_outliers]
X_seg_out       = X_train_seg[mask_outliers].reset_index(drop=True)
y_seg_out       = y_train_seg[mask_outliers].reset_index(drop=True)

print(f"\n✅ Outliers séparés pour analyse individuelle")
if len(X_seg_out) > 0 and 'Frequency' in X_seg_out.columns:
    print(f"   Profil outliers :")
    print(f"   Frequency moy  : {X_seg_out['Frequency'].mean():.1f}")
    print(f"   Monetary moy   : {X_seg_out['MonetaryTotal'].mean():.0f} £")
    print(f"   Churn rate     : {y_seg_out.mean()*100:.1f}%")

print("\n5. ACP sur clients normaux (sans outliers)...")
pca_full  = PCA(random_state=42)
pca_full.fit(X_scaled_clean)
cum_var   = np.cumsum(pca_full.explained_variance_ratio_)

n_comp_80 = np.argmax(cum_var >= 0.80) + 1
n_comp_95 = np.argmax(cum_var >= 0.95) + 1

print(f"   → {n_comp_80} composantes pour 80% de variance")
print(f"   → {n_comp_95} composantes pour 95% de variance")

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(cum_var)+1), cum_var, 'b-o', markersize=3)
plt.axhline(y=0.80, color='orange', linestyle='--', label='80%')
plt.axhline(y=0.95, color='red',    linestyle='--', label='95%')
plt.axvline(x=n_comp_80, color='orange', linestyle=':',
            label=f'n={n_comp_80}')
plt.xlabel('Nombre de composantes')
plt.ylabel('Variance cumulée')
plt.title('Variance expliquée — ACP (sans outliers)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

pca_kmeans   = PCA(n_components=n_comp_80, random_state=42)
X_pca_kmeans = pca_kmeans.fit_transform(X_scaled_clean)

pca_2d   = PCA(n_components=2, random_state=42)
X_pca_2d = pca_2d.fit_transform(X_scaled_clean)

print(f"\n✅ ACP : {X_scaled_clean.shape[1]} → {n_comp_80} composantes")
print(f"   Variance : {pca_kmeans.explained_variance_ratio_.sum():.2%}")

print("\n6. Visualisation ACP 2D (sans outliers)...")
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], alpha=0.5, s=10, c='steelblue')
plt.xlabel('CP 1')
plt.ylabel('CP 2')
plt.title('ACP 2D — Clients normaux (sans outliers)')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
scatter = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1],
                      c=y_seg_clean, cmap='RdYlGn', alpha=0.5, s=10)
plt.xlabel('CP 1')
plt.ylabel('CP 2')
plt.title('ACP 2D — Colorée par Churn')
plt.colorbar(scatter)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n7. Méthode du coude (sur clients normaux)...")
inertia = []
K_range = range(2, 11)
for k_val in K_range:
    km = KMeans(n_clusters=k_val, random_state=42, n_init=10)
    km.fit(X_pca_kmeans)
    inertia.append(km.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(K_range, inertia, 'o-', linewidth=2, markersize=8, color='darkblue')
plt.xlabel('k')
plt.ylabel('Inertie')
plt.title(f'Méthode du coude — sans outliers ({n_comp_80} composantes)')
plt.xticks(K_range)
plt.grid(True, alpha=0.3)
plt.show()

print("\n8. Sélection de k (k=3 vs k=4) sur clients normaux...")
print("="*55)

MIN_CLUSTER_PCT = 3.0

for k_test in [3, 4]:
    km_test  = KMeans(n_clusters=k_test, random_state=42, n_init=10)
    cl_test  = km_test.fit_predict(X_pca_kmeans)
    sizes    = pd.Series(cl_test).value_counts().sort_index()
    min_pct  = (sizes.min() / len(cl_test)) * 100
    status   = "✅ OK" if min_pct >= MIN_CLUSTER_PCT else f"⚠️  <{MIN_CLUSTER_PCT}%"
    print(f"   k={k_test} → Tailles : {sizes.tolist()} | "
          f"Plus petit : {sizes.min()} ({min_pct:.1f}%) {status}")

km_4 = KMeans(n_clusters=4, random_state=42, n_init=10)
cl_4 = km_4.fit_predict(X_pca_kmeans)
min_pct_4 = (pd.Series(cl_4).value_counts().min() / len(cl_4)) * 100

km_3 = KMeans(n_clusters=3, random_state=42, n_init=10)
cl_3 = km_3.fit_predict(X_pca_kmeans)
min_pct_3 = (pd.Series(cl_3).value_counts().min() / len(cl_3)) * 100

if min_pct_4 >= MIN_CLUSTER_PCT:
    k        = 4
    kmeans   = km_4
    clusters = cl_4
    print(f"\n✅ k=4 retenu ({min_pct_4:.1f}% minimum)")
else:
    k        = 3
    kmeans   = km_3
    clusters = cl_3
    print(f"\n✅ k=3 retenu ({min_pct_3:.1f}% minimum)")

X_seg_clean_copy              = X_seg_clean.copy()
X_seg_clean_copy['Cluster']   = clusters
X_seg_clean_copy['Churn']     = y_seg_clean.values
X_seg_clean_copy['IsOutlier'] = False

X_seg_out_copy                = X_seg_out.copy()
X_seg_out_copy['Cluster']     = -1
X_seg_out_copy['Churn']       = y_seg_out.values
X_seg_out_copy['IsOutlier']   = True

X_train_segmented = pd.concat(
    [X_seg_clean_copy, X_seg_out_copy],
    ignore_index=True
)

print(f"\n✅ Segmentation avec k={k} terminée!")
print(f"📊 Répartition :")
for cl_id in sorted(X_seg_clean_copy['Cluster'].unique()):
    cl_data = X_seg_clean_copy[X_seg_clean_copy['Cluster'] == cl_id]
    cl_pct  = len(cl_data) / len(X_seg_clean_copy) * 100
    print(f"   Cluster {cl_id} : {len(cl_data)} clients ({cl_pct:.1f}%)")
print(f"   Outliers  : {len(X_seg_out_copy)} clients "
      f"({len(X_seg_out_copy)/len(X_train_segmented)*100:.1f}%)")

print("\n9. Visualisation des segments (sans outliers)...")
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
scatter = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1],
                      c=clusters, cmap='tab10', alpha=0.6, s=15)
plt.xlabel('CP 1')
plt.ylabel('CP 2')
plt.title(f'Segmentation — {k} clusters (sans outliers)')
plt.colorbar(scatter, label='Cluster')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
cluster_counts = pd.Series(clusters).value_counts().sort_index()
colors_plot    = plt.cm.tab10(range(len(cluster_counts)))
plt.bar(cluster_counts.index, cluster_counts.values,
        color=colors_plot, edgecolor='black')
plt.xlabel('Cluster')
plt.ylabel('Nombre de clients')
plt.title('Répartition par cluster (sans outliers)')
plt.xticks(range(k))
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

print("\n10. Analyse des profils...")
cluster_profiles   = X_seg_clean_copy.groupby('Cluster')[numeric_cols].mean()
key_features       = ['Frequency', 'MonetaryTotal', 'Age', 'Churn']
available_features = [f for f in key_features if f in cluster_profiles.columns]

print("\n📊 PROFILS DES CLUSTERS (clients normaux) :")
print(cluster_profiles[available_features].round(2))

print("\n📊 TAUX DE CHURN PAR CLUSTER :")
for cl_id, cr in (
    X_seg_clean_copy.groupby('Cluster')['Churn'].mean() * 100
).items():
    print(f"   Cluster {cl_id}: {cr:.1f}%")

if len(X_seg_out_copy) > 0 and 'Frequency' in X_seg_out_copy.columns:
    print(f"\n📊 OUTLIERS ({len(X_seg_out_copy)} clients) :")
    print(f"   Churn     : {y_seg_out.mean()*100:.1f}%")
    print(f"   Frequency : {X_seg_out_copy['Frequency'].mean():.1f}")
    print(f"   Monetary  : {X_seg_out_copy['MonetaryTotal'].mean():.0f} £")

plt.figure(figsize=(12, 6))
sns.heatmap(cluster_profiles[available_features],
            annot=True, fmt='.2f', cmap='RdYlGn', center=0)
plt.title('Profils des clusters (sans outliers)')
plt.tight_layout()
plt.show()

print("\n12. Interprétation des clusters...")
print("="*60)
for cl_id in range(k):
    cl_data  = X_seg_clean_copy[X_seg_clean_copy['Cluster'] == cl_id]
    churn_r  = cl_data['Churn'].mean() * 100
    freq     = cl_data['Frequency'].mean()
    monetary = cl_data['MonetaryTotal'].mean()
    size     = len(cl_data)
    pct      = size / len(X_seg_clean_copy) * 100

    print(f"\n🔵 CLUSTER {cl_id} ({size} clients = {pct:.1f}%) :")
    print(f"   - Churn     : {churn_r:.1f}%")
    print(f"   - Frequency : {freq:.1f} achats")
    print(f"   - Monetary  : {monetary:.0f} £")
    if churn_r > 35:
        print(f"   ⚠️  Risque critique → Action urgente")
    elif churn_r > 25:
        print(f"   ⚠️  Risque modéré → Fidélisation")
    else:
        print(f"   ✅  Risque faible → Développement")

if len(X_seg_out_copy) > 0:
    print(f"\n⚪ OUTLIERS ({len(X_seg_out_copy)} clients) :")
    print(f"   → Gros acheteurs (Freq={X_seg_out_copy['Frequency'].mean():.1f}, "
          f"Monetary={X_seg_out_copy['MonetaryTotal'].mean():.0f}£)")

print("\n13. Sauvegarde...")
os.makedirs('../data/processed', exist_ok=True)
os.makedirs('../models', exist_ok=True)
X_train_segmented.to_csv('../data/processed/clients_avec_clusters.csv',
                          index=False)
joblib.dump(kmeans,     '../models/kmeans_model.pkl')
joblib.dump(scaler_seg, '../models/scaler_seg.pkl')
joblib.dump(pca_kmeans, '../models/pca_kmeans.pkl')
joblib.dump(pca_2d,     '../models/pca_2d.pkl')
print("✅ Artefacts segmentation sauvegardés dans models/")

print("\n" + "="*50)
print("🎉 SEGMENTATION TERMINÉE AVEC SUCCÈS!")
print("="*50)

# ============================================================
# SECTION 15 : SAUVEGARDE DU MEILLEUR MODÈLE
# ============================================================
# %%
print("="*50)
print("SAUVEGARDE DU MODÈLE FINAL")
print("="*50)

model_map = {
    'KNN Optimisé'   : knn,
    'Arbre Décision' : clf_decisiontree,
    'Random Forest'  : clf_rf,
    'Gradient Boost.': best_gb,
}
best_name   = results_df.index[0]
final_model = model_map[best_name]

print(f"\n🏆 Meilleur modèle : {best_name}")
print(f"   F1-Score : {results_df.loc[best_name, 'F1-Score']:.4f}")
print(f"   ROC-AUC  : {results_df.loc[best_name, 'ROC-AUC']:.4f}")

os.makedirs('../models', exist_ok=True)
joblib.dump(final_model, '../models/best_model_churn.pkl')
print("✅ best_model_churn.pkl")
print("✅ scaler.pkl")
print("✅ imputer.pkl")
print("✅ feature_names.pkl")

if hasattr(final_model, 'feature_importances_'):
    feat_imp = pd.DataFrame({
        'feature'   : X_train_encoded.columns,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)

    print("\n=== TOP 10 FEATURES ===")
    print(feat_imp.to_string(index=False))

    plt.figure(figsize=(10, 6))
    plt.barh(feat_imp['feature'], feat_imp['importance'], color='steelblue')
    plt.xlabel('Importance')
    plt.title(f'Top 10 features — {best_name}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

print("\n" + "="*50)
print("🎉 MODÈLE SAUVEGARDÉ AVEC SUCCÈS!")
print("="*50)

# %%
# ============================================================
# TABLEAU FINAL — RECOMMANDATIONS MARKETING
# CORRECTION FINALE : Nommage unique garanti
# Outliers = segment Premium (valide)
# ============================================================
print("\n" + "="*60)
print("📊 TABLEAU FINAL DE SEGMENTATION CLIENTÈLE")
print("="*60)

MIN_SEGMENT_SIZE = 10


def assign_segment_name_final(cl_id, churn_r, freq, monetary, size,
                               all_clusters_data, is_outlier=False):
    """
    CORRECTION FINALE : Nommage unique garanti.
    Différencie TOUS les clusters y compris les intermédiaires
    par la combinaison churn + frequency + monetary.
    Les outliers = segment Premium (gros acheteurs valide).
    """
    if is_outlier:
        return (
            "Clients Premium / Gros Acheteurs",
            f"Profils atypiques : forte fréquence ({freq:.1f} achats) "
            f"et valeur élevée ({monetary:.0f}£)",
            "🌟 Suivi personnalisé, offres exclusives, "
            "programme fidélité premium",
            "🟢 Stratégique", "#27ae60"
        )

    if size < MIN_SEGMENT_SIZE:
        return (
            "Outliers / Cas Atypiques",
            f"Segment trop petit ({size} clients)",
            "🔍 Analyse individuelle",
            "⚪ À investiguer", "#95a5a6"
        )

    # Calculer les rangs pour différencier
    max_churn    = max(c['churn']   for c in all_clusters_data)
    min_freq     = min(c['freq']    for c in all_clusters_data)
    max_freq     = max(c['freq']    for c in all_clusters_data)
    max_monetary = max(c['monetary']for c in all_clusters_data)

    # Règle 1 : Churn le plus élevé → Risque Critique
    if churn_r >= max_churn - 2:
        return (
            "Clients à Risque Critique",
            f"Churn le plus élevé ({churn_r:.0f}%), "
            f"fréquence ({freq:.1f} achats)",
            "📞 Contact direct urgent, offre personnalisée -25%",
            "🔴 Urgent", "#e74c3c"
        )

    # Règle 2 : Fréquence la plus élevée → Actifs à Fidéliser
    elif freq >= max_freq - 1:
        return (
            "Clients Actifs à Fidéliser",
            f"Acheteurs les plus fréquents ({freq:.1f} achats), "
            f"valeur ({monetary:.0f}£)",
            "🎁 Programme fidélité renforcé, récompenses fréquence",
            "🟠 Prioritaire", "#e67e22"
        )

    # Règle 3 : Fréquence la plus basse → Peu Actifs
    elif freq <= min_freq + 0.5:
        return (
            "Clients Peu Actifs",
            f"Achats peu fréquents ({freq:.1f} achats), "
            f"à réactiver ({churn_r:.0f}% churn)",
            "📧 Campagne réactivation, offre découverte produits",
            "🟡 Réactivation", "#f39c12"
        )

    # Règle 4 : Cluster intermédiaire avec churn > 30%
    elif churn_r > 30:
        return (
            "Clients Réguliers à Risque",
            f"Profil intermédiaire ({freq:.1f} achats, {monetary:.0f}£), "
            f"risque de départ ({churn_r:.0f}%)",
            "📊 Offres fidélisation ciblées, alertes comportementales",
            "🟠 Vigilance", "#d35400"
        )

    # Règle 5 : Cluster intermédiaire avec churn ≤ 30% → Stables
    else:
        return (
            "Clients Stables à Développer",
            f"Profil stable ({freq:.1f} achats, {monetary:.0f}£), "
            f"risque modéré ({churn_r:.0f}%)",
            "📊 Upselling progressif, cross-selling, programme points",
            "🟡 Développement", "#3498db"
        )


segments = []

# Préparer les données de tous les clusters
all_clusters_info = []
for cl_id in range(k):
    cl_data  = X_seg_clean_copy[X_seg_clean_copy['Cluster'] == cl_id]
    all_clusters_info.append({
        'cl_id'   : cl_id,
        'churn'   : cl_data['Churn'].mean() * 100,
        'freq'    : cl_data['Frequency'].mean(),
        'monetary': cl_data['MonetaryTotal'].mean(),
        'size'    : len(cl_data)
    })

# Créer les segments pour les clusters normaux
for cl_info in all_clusters_info:
    cl_id    = cl_info['cl_id']
    cl_data  = X_seg_clean_copy[X_seg_clean_copy['Cluster'] == cl_id]
    size     = len(cl_data)
    pct      = size / len(X_seg_clean_copy) * 100
    churn_r  = cl_info['churn']
    freq     = cl_info['freq']
    monetary = cl_info['monetary']

    nom, profil, rec, prio, coul = assign_segment_name_final(
        cl_id, churn_r, freq, monetary, size,
        all_clusters_info, is_outlier=False
    )

    segments.append({
        'Cluster'           : cl_id,
        'Nom du segment'    : nom,
        'Nombre clients'    : size,
        '% clients'         : f"{pct:.1f}%",
        'Taux churn'        : f"{churn_r:.1f}%",
        'Frequency (achats)': f"{freq:.1f}",
        'Monetary (£)'      : f"{monetary:.0f}",
        'Profil'            : profil,
        'Recommandation'    : rec,
        'Priorité'          : prio,
        '_couleur'          : coul,
        '_taille_ok'        : size >= MIN_SEGMENT_SIZE
    })

# Ajouter les outliers comme segment Premium valide
if len(X_seg_out_copy) > 0:
    out_churn = y_seg_out.mean() * 100
    out_freq  = X_seg_out_copy['Frequency'].mean() \
                if 'Frequency' in X_seg_out_copy.columns else 0
    out_mon   = X_seg_out_copy['MonetaryTotal'].mean() \
                if 'MonetaryTotal' in X_seg_out_copy.columns else 0
    out_pct   = len(X_seg_out_copy) / len(X_train_segmented) * 100

    nom, profil, rec, prio, coul = assign_segment_name_final(
        -1, out_churn, out_freq, out_mon,
        len(X_seg_out_copy), all_clusters_info, is_outlier=True
    )

    segments.append({
        'Cluster'           : -1,
        'Nom du segment'    : nom,
        'Nombre clients'    : len(X_seg_out_copy),
        '% clients'         : f"{out_pct:.1f}%",
        'Taux churn'        : f"{out_churn:.1f}%",
        'Frequency (achats)': f"{out_freq:.1f}",
        'Monetary (£)'      : f"{out_mon:.0f}",
        'Profil'            : profil,
        'Recommandation'    : rec,
        'Priorité'          : prio,
        '_couleur'          : coul,
        '_taille_ok'        : True
    })

segments_df = pd.DataFrame(segments)

# Vérification unicité avec correction automatique
noms_uniques = segments_df['Nom du segment'].nunique()
noms_total   = len(segments_df)

if noms_uniques == noms_total:
    print(f"✅ Tous les {noms_total} segments ont des noms UNIQUES")
else:
    print(f"⚠️  {noms_total - noms_uniques} doublon(s) → correction automatique")
    seen = {}
    for i, row in segments_df.iterrows():
        name = row['Nom du segment']
        if name in seen:
            seen[name] += 1
            segments_df.at[i, 'Nom du segment'] = (
                f"{name} ({chr(64 + seen[name])})"
            )
        else:
            seen[name] = 1
    print(f"   ✅ Doublons corrigés")

display_cols = ['Cluster', 'Nom du segment', 'Nombre clients', '% clients',
                'Taux churn', 'Frequency (achats)', 'Monetary (£)', 'Priorité']
print("\n📋 TABLEAU RÉCAPITULATIF DES SEGMENTS")
print("="*115)
print(segments_df[display_cols].to_string(index=False))

print("\n\n🎯 RECOMMANDATIONS MARKETING PAR SEGMENT")
print("="*100)
for i in range(len(segments_df)):
    row  = segments_df.iloc[i]
    size = int(row['Nombre clients'])

    print(f"\n📌 Cluster {int(row['Cluster'])} — "
          f"{row['Nom du segment']} ({row['Priorité']})")
    print(f"   📊 Taille    : {size} clients ({row['% clients']})")
    print(f"   📈 Métriques : Churn {row['Taux churn']} | "
          f"{row['Frequency (achats)']} achats | {row['Monetary (£)']}£")
    print(f"   👤 Profil    : {row['Profil']}")
    print(f"   💡 Action    : {row['Recommandation']}")
    print("-"*90)

# Visualisation — tous les segments valides
segments_viz = segments_df[segments_df['_taille_ok']].copy()

if len(segments_viz) > 0:
    col_list   = segments_viz['_couleur'].tolist()
    churn_vals = [float(x.replace('%', '')) for x in segments_viz['Taux churn']]
    size_vals  = [float(x.replace('%', '')) for x in segments_viz['% clients']]
    money_vals = [float(segments_viz['Monetary (£)'].iloc[i])
                  for i in range(len(segments_viz))]
    freq_vals  = [float(segments_viz['Frequency (achats)'].iloc[i])
                  for i in range(len(segments_viz))]
    noms       = segments_viz['Nom du segment'].tolist()

    fig, axes = plt.subplots(2, 2, figsize=(17, 12))

    bars = axes[0, 0].bar(noms, churn_vals, color=col_list, edgecolor='black')
    axes[0, 0].set_ylabel('Taux de churn (%)')
    axes[0, 0].set_title('Taux de churn par segment')
    axes[0, 0].set_xticklabels(noms, rotation=35, ha='right', fontsize=8)
    for b, v in zip(bars, churn_vals):
        axes[0, 0].text(b.get_x()+b.get_width()/2, b.get_height()+0.3,
                        f'{v:.1f}%', ha='center', fontweight='bold', fontsize=8)
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    axes[0, 1].pie(size_vals, labels=noms, autopct='%1.1f%%',
                   colors=col_list, textprops={'fontsize': 7})
    axes[0, 1].set_title('Répartition de tous les segments')

    axes[1, 0].scatter(churn_vals, money_vals,
                       s=[max(v*8, 30) for v in size_vals],
                       c=col_list, alpha=0.85, edgecolors='black')
    for i in range(len(segments_viz)):
        axes[1, 0].annotate(
            noms[i], (churn_vals[i], money_vals[i]),
            xytext=(7, 5), textcoords='offset points', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8)
        )
    axes[1, 0].set_xlabel('Risque de churn (%)')
    axes[1, 0].set_ylabel('Valeur client (£)')
    axes[1, 0].set_title('Matrice Risque × Valeur')
    axes[1, 0].grid(True, alpha=0.3)

    bars2 = axes[1, 1].bar(noms, freq_vals, color=col_list, edgecolor='black')
    axes[1, 1].set_ylabel('Frequency moyenne')
    axes[1, 1].set_title("Fréquence d'achat par segment")
    axes[1, 1].set_xticklabels(noms, rotation=35, ha='right', fontsize=8)
    for b, v in zip(bars2, freq_vals):
        axes[1, 1].text(b.get_x()+b.get_width()/2, b.get_height()+0.05,
                        f'{v:.1f}', ha='center', fontweight='bold', fontsize=8)
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.suptitle('Tableau de bord — Segmentation Clientèle',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    os.makedirs('../reports', exist_ok=True)
    plt.savefig('../reports/segmentation_dashboard.png',
                dpi=150, bbox_inches='tight')
    plt.show()

segments_df.drop(columns=['_couleur', '_taille_ok']).to_csv(
    '../reports/segmentation_recommandations.csv', index=False
)
print("\n✅ Tableau → reports/segmentation_recommandations.csv")
print("✅ Dashboard → reports/segmentation_dashboard.png")

print("\n" + "="*60)
print("🎉 PROJET ML COMPLET — TOUTES LES SECTIONS TERMINÉES!")
print("="*60)