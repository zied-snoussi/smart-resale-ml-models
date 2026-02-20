import joblib
import pandas as pd
import numpy as np
import os
import sys

# Ajout du chemin src pour les imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def predict_price(title, msrp=None):
    """
    Réalise une prédiction en forçant l'alignement strict des colonnes 
    sur le modèle entraîné.
    """
    print(f"\n🔍 Analyse de l'objet : '{title}'")
    
    # 1. CHARGEMENT DES ARTEFACTS
    try:
        vectorizer_artifacts = joblib.load('models/tfidf_svd.pkl')
        scaler = joblib.load('models/scaler.pkl')
        reg_model = joblib.load('models/price_regressor.pkl')
        clf_model = joblib.load('models/price_classifier.pkl')
    except FileNotFoundError:
        print("❌ Erreur : Modèles introuvables. Lancez le pipeline complet.")
        return

    # 2. PRÉPARATION DES CARACTÉRISTIQUES NUMÉRIQUES
    num_features = {
        'msrp': msrp if msrp else 450.0,
        'has_brand': 1 if any(b in title.upper() for b in ['APPLE', 'SAMSUNG', 'SONY', 'NINTENDO']) else 0,
        'is_bundle': 1 if any(w in title.upper() for w in ['BUNDLE', 'LOT', 'PACK', 'WITH']) else 0,
        'title_len': len(title),
        'word_count': len(title.split()),
        'demand_score': 0.5 
    }

    # 3. VECTORISATION DU TEXTE (NLP)
    tfidf = vectorizer_artifacts['tfidf']
    svd = vectorizer_artifacts['svd']
    text_vector = tfidf.transform([title])
    text_svd = svd.transform(text_vector)
    
    # Création du dictionnaire pour les colonnes SVD
    text_features = {f'text_svd_{i}': val for i, val in enumerate(text_svd[0])}
    
    # 4. FUSION ET ALIGNEMENT STRICT (La solution au problème)
    # On combine les deux dictionnaires
    full_features = {**num_features, **text_features}
    
    # On crée le DataFrame
    X_input = pd.DataFrame([full_features])
    
    # RÉALIGNEMENT : On force l'ordre des colonnes tel qu'attendu par le scaler
    # scaler.feature_names_in_ contient l'ordre exact du 'fit'
    try:
        X_input = X_input[scaler.feature_names_in_]
        X_scaled = scaler.transform(X_input)
    except Exception as e:
        print(f"❌ Erreur lors de l'alignement des colonnes : {e}")
        return

    # 5. INFÉRENCE
    price = reg_model.predict(X_scaled)[0]
    category = clf_model.predict(X_scaled)[0]
    
    # 6. AFFICHAGE
    print("+" + "-"*48 + "+")
    print(f"| 💰 PRIX ESTIMÉ : {price:.2f} €")
    print(f"| 📊 SEGMENT     : {category}")
    print("+" + "-"*48 + "+")

if __name__ == "__main__":
    predict_price("Apple iPhone 13 Pro 128GB Sierra Blue")
    predict_price("Nintendo Switch BUNDLE with Mario Kart", msrp=320.0)