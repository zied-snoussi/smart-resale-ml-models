import pandas as pd
import joblib
import os
import logging
from utils.feature_utils import (
    extract_numerical_features, 
    process_text_vectors, 
    scale_features
)

# ==============================================================================
# CONFIGURATION DU LOGGING
# ==============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_step2_features():
    """
    ÉTAPE 2 : INGÉNIERIE DES CARACTÉRISTIQUES (FEATURE ENGINEERING)
    --------------------------------------------------------------
    Objectif : Transformer le texte en vecteurs sémantiques (SVD) et 
    standardiser les données numériques pour optimiser l'apprentissage.
    """
    logging.info("🚀 DÉMARRAGE DE L'ÉTAPE 2 : EXTRACTION DES CARACTÉRISTIQUES")

    # 1. CHARGEMENT DES DONNÉES PRÉPARÉES (ÉTAPE 1)
    input_path = 'data/processed/ebay_prep.pkl'
    if not os.path.exists(input_path):
        logging.error("❌ Fichier de préparation introuvable ! Veuillez exécuter l'Étape 1 d'abord.")
        return
    
    df = pd.read_pickle(input_path)

    # 2. EXTRACTION DES CARACTÉRISTIQUES NUMÉRIQUES
    # Génère des variables comme 'is_bundle', 'title_length', ou 'msrp'.
    logging.info("Extraction des variables numériques et catégorielles...")
    df_features = extract_numerical_features(df)

    # 3. VECTORISATION DU TEXTE (NLP PROFESSIONNEL)
    # Utilise TF-IDF + SVD pour transformer le 'Titre' en 20 colonnes numériques denses.
    # Cela permet de capturer la sémantique (ex: "neuf" vs "occasion") sans avoir des milliers de colonnes.
    logging.info("Génération des vecteurs sémantiques (SVD) à partir des titres...")
    X_text, vectorizer_artifacts = process_text_vectors(df['Title'])
    
    # 4. COMBINAISON ET NORMALISATION (SCALING)
    # Fusion des caractéristiques numériques et des vecteurs textuels.
    X_combined = pd.concat([df_features, X_text], axis=1)
    y = df['price_cleaned'] # Notre variable cible (Target)
    
    # Le StandardScaler est crucial pour que les variables à grande échelle 
    # n'écrasent pas les variables binaires lors de l'entraînement.
    logging.info(f"Normalisation des {X_combined.shape[1]} caractéristiques...")
    X_scaled, scaler = scale_features(X_combined)

    # 5. SAUVEGARDE DES ARTEFACTS ET TRANSFORMATEURS
    # On sépare les données de l'entraînement des modèles de transformation (scaler/tfidf).
    os.makedirs('models', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Données finales pour l'Étape 3 (Entraînement)
    joblib.dump(X_scaled, 'data/processed/X_final.joblib')
    joblib.dump(y, 'data/processed/y_final.joblib')
    
    # Sauvegarde des "Transformers" : indispensable pour traiter de nouvelles annonces 
    # de la même manière dans l'application finale (Inférence).
    joblib.dump(vectorizer_artifacts, 'models/tfidf_svd.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')

    logging.info(f"✅ Étape 2 terminée ! Nombre de features prêtes : {X_scaled.shape[1]}")

if __name__ == "__main__":
    run_step2_features()