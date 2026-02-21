# 🤖 Smart Resale AI : Documentation Technique & Architecture ML

**Système Expert d'Estimation de Valeur Résiduelle et d'Optimisation de Revente**

**Version :** 1.1

**Statut :** Production Ready

**Auteur :** Zied Snoussi

---

## 🏛️ 1. Vue d'Ensemble de l'Architecture

Le système repose sur une architecture modulaire dite **"End-to-End"**, transformant des données brutes hétérogènes (eBay/Amazon) en insights décisionnels actionnables. Le pipeline complet s'exécute en **70 secondes** sur une configuration standard.

### **Phase A : Ingénierie des Données & Alignement Sémantique**

* **Data Cleansing :** Application d'un "Hard Cap" à 5000€ et d'un filtrage statistique (méthode IQR) pour éliminer le bruit et les anomalies de prix.
* **Matching Multi-Sources :** Utilisation d'un moteur de recherche vectoriel (`TfidfVectorizer` + `NearestNeighbors`) pour réconcilier le catalogue Amazon (MSRP/Prix Neuf) avec les flux transactionnels eBay.
* **Validation des Données :** Filtre de cohérence logique supprimant les paires où le prix d'occasion excède de 150% le prix neuf identifié.

### **Phase B : Feature Engineering & NLP**

* **Vectorisation Sémantique :** Réduction de dimensionnalité via **LSA** (Latent Semantic Analysis) utilisant la Décomposition en Valeurs Singulières (**SVD**) pour capturer le contexte des titres sur 26 dimensions.
* **Extraction de Métadonnées :** Parsing par expressions régulières (Regex) pour identifier les variables critiques (Marque, Capacité, État).
* **Standardisation :** Normalisation via `StandardScaler` pour garantir la convergence et l'équité de poids entre les variables numériques et textuelles.

---

## 📊 2. Benchmarks de Performance & Métriques

Le modèle a été validé par un protocole de test rigoureux (Hold-out validation).

### **Performance de Régression (Valeur Précise)**

| Métrique | Score | Interprétation |
| --- | --- | --- |
| **Coefficient ** | **0.8589** | 86% de la variance du prix est capturée par le modèle. |
| **MAE (Erreur Moyenne)** | **32.80€** | Écart moyen extrêmement faible par rapport au prix réel. |
| **Biais Résiduel** | **Neutre** | Distribution d'erreur centrée sur zéro (pas de sur/sous-estimation systématique). |

### **Performance de Classification (Segmentation de Marché)**

Le modèle classifie les produits en trois tiers (Low, Mid, High) avec une **précision globale de 91.65%**.

* **Segment "Low" (Accessoires/Entrée de gamme) :** 93.4% de précision.
* **Segment "Mid" (Cœur de marché) :** 89.0% de rappel (minimise les faux négatifs).
* **Segment "High" (Produits Premium) :** 92.5% de précision (sécurise les estimations sur les objets à haute valeur).

---

## 🛠️ 3. Stack Technique & Structure

L'implémentation suit les standards de l'industrie avec une séparation stricte des préoccupations.

```text
/smart-resale-ml-models
├── data/ 
│   ├── raw/            # Datasets sources Amazon & eBay
│   └── processed/      # Données transformées et sets d'entraînement
├── models/             # Artefacts sérialisés (modèles .pkl, scalers)
├── src/
│   ├── pipeline/       # Scripts d'exécution (Step 1 à 4)
│   ├── utils/          # Moteurs NLP, Preprocessing et Visualisation
│   └── app.py          # Dashboard Streamlit de production
└── static/plots/       # Rapports d'analyse diagnostique (plots)

```

---

## 📈 4. Alignement avec les Standards CRISP-DM

Ce projet implémente les concepts fondamentaux du Machine Learning moderne :

1. **Traitement des Outliers :** Utilisation du seuil de 1.5x l'écart interquartile pour la robustesse statistique.
2. **Transformation des Variables :** Application de `np.log1p` sur les variables de prix pour normaliser les distributions asymétriques.
3. **Choix du Modèle :** Utilisation de **Random Forest** (Ensemble Learning), offrant une stabilité supérieure aux arbres de décision classiques et permettant l'analyse de l'importance des variables.
4. **Stratégie de Déploiement :** Persistance des modèles via `joblib` pour une inférence instantanée dans l'interface utilisateur.

---

## 💡 5. Business Logic : Aide à la Décision

Le système ne se contente pas de prédire ; il conseille. En comparant le `Prix Demandé` au `Prix Prédit`, l'algorithme génère des recommandations stratégiques :

* **"Undervalued" :** Opportunité d'achat immédiate (Arbitrage).
* **"Overpriced" :** Recommandation de baisse de prix pour accélérer la rotation de stock.
* **"Optimal" :** Alignement parfait avec les conditions du marché.