# 📘 Documentation Technique : Smart Resale ML

**Version :** 1.0

**Objectif :** Estimation de prix de revente et aide à la décision (Buy/Sell) basée sur le matching sémantique eBay/Amazon.

---

## 🏗️ 1. Architecture du Pipeline

Le projet est décomposé en 4 étapes modulaires exécutées séquentiellement en environ **70 secondes**.

### **Étape 1 : Préparation & Enrichissement Sémantique**

* **Nettoyage :** Suppression des outliers (prix > 5000€) et des valeurs aberrantes.
* **Matching TF-IDF :** Utilisation de `TfidfVectorizer` et `NearestNeighbors` pour mapper les produits eBay au catalogue Amazon (100 582 produits Tech).
* **Calcul du MSRP :** Récupération du prix "neuf" d'Amazon pour calculer la dépréciation.
* **Filtre d'Anomalies :** Suppression automatique des matchs où le prix d'occasion est > 1.5x le prix neuf.

### **Étape 2 : Feature Engineering**

* **Vecteurs SVD :** Transformation des titres textuels en 26 composantes numériques via la Décomposition en Valeurs Singulières (SVD).
* **Features Numériques :** Longueur du titre, marque, et score de confiance du matching.
* **Scaling :** Standardisation des données pour une performance optimale des modèles.

### **Étape 3 : Entraînement des Modèles**

* **Régression (Prix précis) :** Prédit la valeur exacte en Euros.
* **Classification (Tiers de prix) :** Catégorise le produit en "Low", "Mid", ou "High" via des quantiles.

---

## 📊 2. Analyse des Performances (Benchmarks)

D'après les derniers résultats obtenus sur votre ThinkPad :

### **Indicateurs de Régression**

* **R² Score : 0.8589** (Le modèle explique 86% de la variance des prix).
* **MAE (Mean Absolute Error) : 32.80€** (L'erreur moyenne est de seulement 32€ par objet).
* **Biais :** La distribution des résidus montre un modèle parfaitement centré sur 0.

### **Indicateurs de Classification**

* **Précision Globale : 91.65%**.
* **Rapport détaillé :**
* **Low :** 93.4% de précision (Excellent pour les accessoires).
* **Mid :** 89.0% de rappel (Idéal pour le cœur de marché).
* **High :** 92.5% de précision (Très fiable pour les produits de luxe/high-tech).



---

## 🖼️ 3. Interprétation des Graphiques

Votre pipeline génère automatiquement 4 rapports visuels dans `/static/plots/` :

1. **Réel vs Prédit :** Plus les points bleus collent à la ligne rouge, plus le modèle est performant.
2. **Matrice de Confusion :** Montre les cases où le modèle hésite (ex: confondre un prix "Mid" avec un "High").
3. **Distribution de l'Erreur :** Une cloche étroite signifie que les grosses erreurs sont rares.
4. **Importance des Variables :** Révèle que le **Texte (SVD)** et le **MSRP (Prix Amazon)** sont les moteurs principaux du prix.

---

## 📂 4. Structure des Fichiers

```text
/smart-resale-ml-models
├── data/
│   ├── raw/            # Datasets originaux (Amazon/eBay)
│   └── processed/      # Données prêtes pour le ML (.pkl, .joblib)
├── models/             # Modèles entraînés (.pkl)
├── src/
│   ├── pipeline/       # Étapes 1 à 4
│   ├── utils/          # Moteur de matching et visualisations
│   └── run_pipeline.py # Script de lancement unique
└── static/plots/       # Vos rapports visuels générés

```

---

## 💡 5. Recommandations de Business Logic

Le système génère des conseils automatiques basés sur la comparaison `Prix Actuel` vs `Prix Prédit` :

* **"Lower Price" :** Si le prix actuel est > 10% au-dessus de la prédiction.
* **"Increase Price" :** Si le prix actuel est < 10% en dessous de la prédiction.
* **"Optimal" :** Si l'écart est négligeable.

---

### 1. Compréhension métier et des données

* **Objectifs :** Dans `step3_training.py`, nous avons explicitement séparé la **Régression** (prix exact) et la **Classification** (tiers Low/Mid/High).
* **Identification :** Votre code identifie clairement les *Features* (MSRP via le matching Amazon, caractéristiques extraites par Regex dans `feature_utils.py`) et le *Label* (colonne `price_cleaned`).

### 2. Préparation des données (Data Preparation)

C'est la partie la plus dense de votre projet :

* **Nettoyage & Outliers :** La fonction `remove_outliers` (méthode IQR) et le "Hard Cap" à 5000€ dans `preprocessing.py` traitent directement ce point. L'utilisation de `np.log1p` (Log-transformation) pour le prix est également implémentée pour stabiliser la variance.
* **Mise à l'échelle :** Dans `step2_features.py`, la fonction `scale_features` applique un `StandardScaler` (équivalent robuste à Min-Max) pour que les prix et les vecteurs textuels soient comparables.
* **Feature Engineering :** Vos fonctions Regex extraient la marque, la mémoire et la taille d'écran, ce qui transforme du texte brut en variables hautement corrélées au prix.

### 3. Modélisation (Modeling)

Votre code utilise la **Forêt Aléatoire (Random Forest)** pour la régression et la classification.

* **Pourquoi ce choix ?** La Forêt Aléatoire est une extension des **Arbres de Décision** cités dans votre cours. Elle est plus performante et stable (évite l'overfitting). Elle répond parfaitement à la consigne de tester des modèles basés sur la logique décisionnelle.

### 4. Évaluation (Evaluation)

* **Découpage :** La fonction `prepare_train_test_split` dans `preprocessing.py` effectue la séparation Training/Test.
* **Métriques :** Dans `visual_utils.py` et `step4_evaluation.py`, nous générons la **Matrice de Confusion** et le rapport de classification (Précision, Recall, F1-Score), ainsi que le graphique de dispersion pour la régression.

### 5. Déploiement (Deployment)

* L'étape de **Persistance** (sauvegarde des fichiers `.pkl` et `.joblib` dans le dossier `models/`) est la porte d'entrée du déploiement.
* Ces fichiers permettent à une application externe (comme votre futur Dashboard) d'utiliser le modèle pour faire des prédictions en temps réel sans avoir à ré-entraîner l'IA.

---

### 📊 Tableau de correspondance Projet vs Cours

| Étape de votre cours | Implémentation dans votre code | Fichier source |
| --- | --- | --- |
| **Outliers** | Méthode IQR (Seuil 1.5) | `utils/preprocessing.py` |
| **Normalisation** | `StandardScaler` | `utils/feature_utils.py` |
| **Encodage** | NLP (TF-IDF + SVD) | `utils/feature_utils.py` |
| **Régression** | `RandomForestRegressor` | `pipeline/step3_training.py` |
| **Classification** | `RandomForestClassifier` | `pipeline/step3_training.py` |
| **Matrice de Confusion** | `plot_confusion_matrix` | `utils/visual_utils.py` |

**En résumé :** Votre projet ne se contente pas de suivre ces étapes, il les automatise de manière professionnelle. Vous avez transformé la théorie du cours en un **système de production réel**.