# 🎁 Machine Learning - Analyse Comportementale Clientèle Retail

## 📌 Description
Projet de prédiction du **churn client** pour une entreprise e-commerce de cadeaux.  
L'objectif est d'identifier les clients à risque de départ pour personnaliser les actions marketing.

## 🎯 Objectifs
- Prédire le départ des clients (churn)
- Segmenter la clientèle pour des actions ciblées
- Déployer une interface de prédiction en temps réel

## 📊 Données
- **4 372 clients**
- **52 features** (RFM, comportement, données clients)
- Données intentionnellement imparfaites (valeurs manquantes, outliers, leakage)

## 🛠️ Technologies utilisées
| Catégorie | Technologies |
|-----------|--------------|
| Langage | Python 3.10+ |
| Data | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| ML | Scikit-learn (GB, RF, Decision Tree, KNN) |
| Déploiement | Flask, HTML/CSS |
| Versioning | Git, GitHub |

## 📈 Modélisation

| Modèle | F1-Score | ROC-AUC | Recall |
|--------|----------|---------|--------|
| **Gradient Boosting** | **0.830** | **0.947** | 0.787 |
| Arbre Décision | 0.819 | 0.944 | **0.924** |
| KNN Optimisé | 0.818 | 0.925 | 0.825 |
| Random Forest | 0.783 | 0.907 | 0.787 |

🏆 **Modèle final retenu : Gradient Boosting** (meilleur compromis F1/AUC)

## 🔍 Segmentation clientèle (ACP + K-means)
- **4 clusters** identifiés après retrait des outliers (817 clients)
- Taux de churn par cluster : 19% à 38%
- Recommandations marketing personnalisées par segment

## 🌐 Déploiement Flask
- Interface web pour prédire le churn en temps réel
- API REST disponible
- Scores RFM et conseils personnalisés

# Projet Machine Learning - Prédiction du Churn

## 📦 Livrables

### Présentation et Rapport
[![PDF](https://img.shields.io/badge/Rapport-PDF-red)](https://github.com/myriamBenAbd0607/ProjetML/blob/main/presentations/rapport_projet_ml.pdf)
[![PPTX](https://img.shields.io/badge/Présentation-PowerPoint-orange)](https://github.com/myriamBenAbd0607/ProjetML/blob/main/presentations/presentation_projet_ml.pptx)

### Code source
- [🐍 Application Flask](https://github.com/myriamBenAbd0607/ProjetML/blob/main/app/app.py)
- [📓 Notebook principal](https://github.com/myriamBenAbd0607/ProjetML/blob/main/notebooks/test.py)

### Modèles et données
- [🤖 Modèle final](https://github.com/myriamBenAbd0607/ProjetML/tree/main/models)
- [📊 Rapports de segmentation](https://github.com/myriamBenAbd0607/ProjetML/tree/main/reports)

## 🚀 Exécution du projet

```bash
git clone https://github.com/myriamBenAbd0607/ProjetML.git
cd ProjetML
pip install -r requirements.txt
cd app
python app.py