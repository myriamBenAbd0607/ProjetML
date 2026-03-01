# Notes d'exploration - Projet ML Retail

**Date:** 18 février 2026
**Auteur:** [Votre nom]

---

## 1. STRUCTURE GÉNÉRALE DES DONNÉES

- **Nombre de clients:** 4 372
- **Nombre de features:** 52
- **Types de données:**
  - int64 (20 colonnes) : variables numériques entières
  - float64 (14 colonnes) : variables numériques décimales
  - object (18 colonnes) : variables catégorielles

---

## 2. ANALYSE DE LA VARIABLE CIBLE (CHURN)

- **Churn = 0 (fidèle):** XX clients (XX%)
- **Churn = 1 (parti):** XX clients (XX%)

> À compléter après avoir calculé

---

## 3. VALEURS MANQUANTES

| Colonne | Manquantes | Pourcentage | Observation |
|---------|------------|-------------|-------------|
| Age | 1311 | 30.0% | ✅ Conforme au PDF |
| AvgDaysBetweenPurchases | 79 | 1.8% | Probablement clients avec 1 seul achat |

---

## 4. VALEURS ABERRANTES / CODES SPÉCIAUX

### SupportTicketsCount
- Valeurs normales: 0-8 (96.9% des clients)
- **Codes spéciaux:**
  - -1 : 43 clients (1.0%)
  - 999 : 87 clients (2.0%)
- **Total à traiter:** 130 clients (3%) 

### SatisfactionScore
- Valeurs normales: 1-5 (92.0% des clients)
- **Codes spéciaux:**
  - -1 : 115 clients (2.6%)
  - 0 : 120 clients (2.7%)
  - 99 : 114 clients (2.6%)
- **Total à traiter:** 349 clients (8%)

### MonetaryTotal
- Valeurs négatives: **À compter**
- Interprétation: retours produits ? remboursements ?

---

## 5. ANALYSE DES COLONNES CATÉGORIELLES

**18 colonnes catégorielles identifiées:**
- RFMSegment
- AgeCategory
- SpendingCategory
- CustomerType
- FavoriteSeason
- PreferredTimeOfDay
- Region
- LoyaltyLevel
- ChurnRiskCategory
- WeekendPreference
- BasketSizeCategory
- ProductDiversity
- Gender
- AccountStatus
- Country
- ... (liste complète à mettre)

---

## 6. PREMIÈRES OBSERVATIONS STATISTIQUES

| Feature | Min | Max | Moyenne | Médiane | Observation |
|---------|-----|-----|---------|---------|-------------|
| Recency | 1 | 374 | 92 | 50 | Présence de clients inactifs |
| Frequency | 1 | 248 | 5 | 3 | Distribution asymétrique |
| MonetaryTotal | -4287 | 279489 | 1898 | 648 | Très gros clients, valeurs négatives |

---

## 7. PROBLÈMES DE QUALITÉ IDENTIFIÉS (RÉSUMÉ)

1. **Valeurs manquantes** (Age, AvgDaysBetweenPurchases)
2. **Valeurs aberrantes** (SupportTickets, Satisfaction)
3. **Valeurs négatives** (MonetaryTotal, MonetaryMin)
4. **Formats de dates inconsistants** (RegistrationDate)
5. **Colonnes constantes à vérifier** (NewsletterSubscribed probablement)

---

## 8. PLAN D'ACTION POUR LE NETTOYAGE

| Problème | Solution proposée |
|----------|-------------------|
| Age (30% manquants) | Imputation (médiane ?) |
| AvgDaysBetweenPurchases (1.8% manquants) | Remplacer par 0 ou NaN |
| SupportTickets (-1, 999) | Remplacer par NaN ou valeur par défaut |
| Satisfaction (-1, 99) | Remplacer par NaN ou mode |
| MonetaryTotal négatif | À discuter (garder ? corriger ?) |
| RegistrationDate | Parsing et standardisation |
| Colonnes catégorielles | Encodage (one-hot, ordinal) |

---

## 9. QUESTIONS À RÉSOUDRE

- [ ] Combien de clients avec MonetaryTotal négatif ?
- [ ] Les valeurs -1 et 99 dans Satisfaction ont-elles une signification ?
- [ ] Que faire des valeurs extrêmes dans MonetaryTotal ?
- [ ] Y a-t-il des colonnes constantes à supprimer ?

---

## 10. NOTES COMPLÉMENTAIRES

- Le PDF mentionnait 30% de valeurs manquantes pour Age → confirmé
- NewsletterSubscribed semble constante ("Yes") → à supprimer probablement
- RegistrationDate a plusieurs formats → à parser

## Traitement des codes spéciaux

### SupportTicketsCount
- 43 valeurs -1 et 87 valeurs 999 → remplacées par la médiane (2.0)
- ✅ Plus aucune valeur spéciale

### SatisfactionScore
- 115 valeurs -1, 120 valeurs 0, 114 valeurs 99 → remplacées par la médiane (3.0)
- ✅ Plus aucune valeur spéciale

## Décision sur les valeurs négatives

- **44 clients** avec MonetaryTotal négatif (1%)
- Leur ReturnRatio moyen est de **87%** (vs 3% global)
- ✅ **Décision**: On les GARDE
- Justification: Ces cas existent dans la réalité, le modèle doit les apprendre

## Analyse des dates d'inscription
- **2009**: 36 clients (0.8%)
- **2010**: 2143 clients (49.0%)
- **2011**: 2193 clients (50.2%)

**Saisonnalité**:
- Pic d'inscriptions en avril-mai-juin (printemps)
- Moins d'inscriptions en janvier-février
## Colonnes supprimées
| Colonne | Raison |
|---------|--------|
| NewsletterSubscribed | Constante (100% = "Yes") |
✅ Valeurs manquantes traitées
Age → imputé par médiane

AvgDaysBetweenPurchases → remplacé par 0

✅ Codes spéciaux traités
SupportTickets (-1, 999) → remplacés par médiane

Satisfaction (-1, 0, 99) → remplacés par médiane

✅ Dates converties
RegistrationDate → format datetime

Nouvelles features: RegYear, RegMonth, RegDay, RegWeekday

✅ Colonnes inutiles supprimées
NewsletterSubscribed (constante)

✅ Valeurs négatives conservées (cas réels)
## Récapitulatif du nettoyage

- **Shape finale**: 4372 lignes, 55 colonnes
- **Valeurs manquantes**: 0
- **Codes spéciaux**: 0
- **Colonnes ajoutées**: RegYear, RegMonth, RegDay, RegWeekday
- **Colonnes supprimées**: NewsletterSubscribed

## Premier modèle - KNN (K Plus Proches Voisins)
### Résultats
- **Accuracy**: 90.9%
- **Précision**: 97.7% (excellente)
- **Rappel**: 74.2% (à améliorer)
- **F1-Score**: 0.84

### Interprétation métier
- ✅ Très fiable quand il prédit un départ
- ⚠️ Rate 1 client partant sur 4
- 📊 Performance globalement très bonne pour un premier modèle