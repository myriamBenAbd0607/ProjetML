# Projet ML Retail - Analyse Comportementale Clientèle

## Description
Application de prédiction du risque de départ client (churn) pour un e-commerce de cadeaux.  
Ce projet a été réalisé dans le cadre de l'atelier Machine Learning.

## Performances du modèle
- **Modèle** : Random Forest
- **Exactitude** : **91.9%**
- **Précision** : 92.3%
- **Rappel** : 91.1%
- **F1-Score** : 91.7%
- **Features** : 4 326 après encodage one-hot

## Segmentation client (K-Means)
| Segment | Clients | Taux Churn | Profil |
|---------|---------|------------|--------|
| Cluster 0 (VIP) | 547 (16%) | 3.3% | Achats fréquents, dépenses élevées |
| Cluster 1 (Risque) | 2 (0.1%) | 50% | Clients extrêmement inactifs |
| Cluster 2 (Occasionnels) | 2 936 (84%) | 39% | Faible fréquence d'achat |
| Cluster 3 (Super VIP) | 12 (0.3%) | 0% | Ultra fidèles, très haute valeur |

## Installation

### 1. Cloner le dépôt
```bash
git clone https://github.com/myriamBenAbd0607/ProjetML.git
cd projetML