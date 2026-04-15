# Notes d'exploration - Projet ML Retail

**Date:** 15 Avril 2026  
**Auteur:** Myriam Ben Abdallah  
**Projet:** Analyse Comportementale Clientèle - Prédiction du Churn

---

## 📊 1. STRUCTURE GÉNÉRALE DES DONNÉES

| Indicateur | Valeur |
|------------|--------|
| **Nombre de clients** | 4 372 |
| **Nombre de features** | 52 |
| **Période** | 2009-2011 |

### Types de données

| Type | Nb colonnes | Exemples |
|------|-------------|----------|
| `int64` | 20 | CustomerID, Recency, Frequency, Age |
| `float64` | 14 | MonetaryTotal, ReturnRatio, SatisfactionScore |
| `object` | 18 | RFMSegment, CustomerType, Region |

---

## 🎯 2. VARIABLE CIBLE (CHURN)

| Classe | Libellé | Nombre | Pourcentage |
|--------|---------|--------|-------------|
| **0** | Fidèle | 2 918 | 66.7% |
| **1** | Partant | 1 454 | 33.3% |

> ✅ Déséquilibre acceptable (ratio 2:1)

---

## ❓ 3. VALEURS MANQUANTES

| Colonne | Manquantes | % | Traitement |
|---------|------------|---|-------------|
| **Age** | 1 311 | 30.0% | Imputation par médiane (APRÈS split) |
| **AvgDaysBetweenPurchases** | 79 | 1.8% | Clients avec 1 seul achat → valeur cohérente |

---

## ⚠️ 4. VALEURS ABERRANTES / CODES SPÉCIAUX

### SupportTicketsCount

| Valeur | Nb clients | Signification | Traitement |
|--------|------------|---------------|-------------|
| -1 | 43 | Non applicable | → Remplacé par médiane (2) |
| 999 | 87 | Valeur extrême | → Remplacé par médiane (2) |
| 0-9 | 4 242 | Valeurs normales | Conservées |

### SatisfactionScore

| Valeur | Nb clients | Signification | Traitement |
|--------|------------|---------------|-------------|
| -1 | 115 | Non renseigné | → Remplacé par médiane (3) |
| 0 | 120 | Non renseigné | → Remplacé par médiane (3) |
| 99 | 114 | Valeur extrême | → Remplacé par médiane (3) |
| 1-5 | 4 023 | Valeurs normales | Conservées |

### MonetaryTotal (dépenses négatives)

| Indicateur | Valeur |
|------------|--------|
| **Clients avec dépense négative** | 44 (1%) |
| **ReturnRatio moyen (clients négatifs)** | 87% |
| **ReturnRatio moyen global** | 3% |

**Décision:** ✅ On GARDE ces clients (cas réels de retours/remboursements)

---

## 📅 5. ANALYSE DES DATES D'INSCRIPTION

### Répartition par année

| Année | Nb clients | % |
|-------|------------|---|
| 2009 | 36 | 0.8% |
| 2010 | 2 143 | 49.0% |
| 2011 | 2 193 | 50.2% |

### Taux de churn par mois d'inscription

| Mois | Taux churn | Risque |
|------|------------|--------|
| Mai | 41.1% | 🔴 Très élevé |
| Février | 40.6% | 🔴 Élevé |
| Janvier | 38.6% | 🟠 Modéré |
| Septembre | 23.9% | 🟢 Faible |

> 📌 **Insight métier:** Les clients inscrits au printemps partent plus que ceux inscrits en automne.

---

## 🔧 6. TRAITEMENTS EFFECTUÉS

### Feature Engineering

| Nouvelle feature | Source | Utilité |
|-----------------|--------|---------|
| `RegYear`, `RegMonth`, `RegDay` | RegistrationDate | Capturer saisonnalité |
| `IP_IsPrivate`, `IP_Class`, `IP_FirstOctet` | LastLoginIP | Détecter type client |
| `HasNegativeMonetary` | MonetaryTotal | Flag retours produits |

### Suppressions

| Colonne | Raison |
|---------|--------|
| `NewsletterSubscribed` | Constante (100% = "Yes") |
| `CustomerID` | Identifiant non prédictif |
| `RegistrationDate` | Remplacée par RegYear, RegMonth, etc. |
| `LastLoginIP` | Remplacée par 3 features |

---

## 🧹 7. RÉCAPITULATIF DU NETTOYAGE

| Étape | Résultat |
|-------|----------|
| **Shape initiale** | 4 372 × 52 |
| **Valeurs manquantes** | 0 (après imputation) |
| **Codes spéciaux** | 0 (remplacés) |
| **Features après nettoyage** | 41 |
| **Features après encodage** | 103 |

---

## 🤖 8. MODÈLES TESTÉS

| Modèle | F1-Score | ROC-AUC | Recall |
|--------|----------|---------|--------|
| **Gradient Boosting** | **0.830** | **0.947** | 0.787 |
| Arbre Décision | 0.819 | 0.944 | **0.924** |
| KNN Optimisé | 0.818 | 0.925 | 0.825 |
| Random Forest | 0.783 | 0.907 | 0.787 |

> 🏆 **Modèle final retenu:** Gradient Boosting

---

## 📊 9. SEGMENTATION CLIENT (K-Means)

| Segment | Taille | Churn | Profil | Action |
|---------|--------|-------|--------|--------|
| **Cluster 0** | 385 (12%) | 32.5% | 4.2 achats, 1167£ | Offres ciblées |
| **Cluster 1** | 1 143 (36%) | 37.5% | 6.2 achats, 1874£ | Contact urgent |
| **Cluster 2** | 217 (7%) | 29.5% | 5.0 achats, 1488£ | Upselling |
| **Cluster 3** | 1 413 (45%) | 30.8% | 3.0 achats, 949£ | Réactivation |
| **Outliers** | 339 (10%) | 32.4% | 11.9 achats, 7129£ | Suivi VIP |

---

## ✅ 10. PLAN D'ACTION RÉALISÉ

| Problème | Solution appliquée |
|----------|-------------------|
| Age (30% manquants) | Imputation par médiane APRÈS split |
| Codes spéciaux (-1, 999) | Remplacement par médiane |
| Dates inconsistantes | Parsing + extraction année/mois/jour |
| Colonnes constantes | Suppression |
| Features leakantes | Suppression (Recency, RFMSegment, etc.) |
| Multicolinéarité | Suppression 7 features redondantes |

---

## 📌 11. NOTES COMPLÉMENTAIRES

- ✅ Le fichier est prêt pour être versionné sur GitHub
- ✅ Les résultats finaux sont dans `reports/segmentation_recommandations.csv`
- ✅ Le modèle final est sauvegardé dans `models/best_model_churn.pkl`
