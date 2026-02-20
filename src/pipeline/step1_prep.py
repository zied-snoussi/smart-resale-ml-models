import pandas as pd
import os
import logging
from utils.load_data import load_ebay_data
from utils.preprocessing import preprocess_ebay_data
from utils.enrichment import enrich_ebay_with_amazon

# ==============================================================================
# CONFIGURATION DU LOGGING (TRAÇABILITÉ PROFESSIONNELLE)
# ==============================================================================
# Le logging est essentiel en production pour surveiller l'état du pipeline
# sans polluer la sortie standard uniquement avec des 'print'.
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

def run_step1_preparation():
    """
    ÉTAPE 1 : PRÉPARATION ET ENRICHISSEMENT DES DONNÉES
    --------------------------------------------------
    Objectif : Nettoyer les données brutes eBay et les fusionner avec les 
    références de prix (MSRP) d'Amazon via un matching sémantique.
    """
    logging.info("🚀 DÉMARRAGE DE L'ÉTAPE 1 : PRÉPARATION DES DONNÉES")

    # 1. CHARGEMENT DES DONNÉES BRUTES
    # Utilisation des utilitaires modulaires pour maintenir un code propre et lisible.
    logging.info("Extraction des données eBay depuis les sources locales...")
    df_raw = load_ebay_data()
    
    # 2. NETTOYAGE ET GESTION DES VALEURS ABERRANTES (OUTLIERS)
    # Cette étape convertit les prix en nombres et élimine les annonces 
    # dont les prix sont irréalistes (ex: 0€ ou > 5000€).
    logging.info("Nettoyage des données et filtrage statistique des anomalies...")
    df_clean = preprocess_ebay_data(df_raw)

    # 3. ENRICHISSEMENT DES DONNÉES (MATCHING SÉMANTIQUE)
    # Le "coeur" du projet : faire correspondre les objets eBay au catalogue Amazon 
    # pour obtenir le prix neuf et la popularité du produit.
    logging.info("Enrichissement via le catalogue Amazon (Vecteurs TF-IDF)...")
    df_enriched = enrich_ebay_with_amazon(df_clean)

    # 4. CONTRÔLE DE QUALITÉ FINAL (SANITY FILTER)
    # Étape critique : On ne conserve que les lignes exploitables où le matching 
    # a réussi et où le prix nettoyé est présent.
    df_final = df_enriched[df_enriched['price_cleaned'] > 0].copy()
    
    # 5. SAUVEGARDE DES ARTEFACTS TRAITÉS
    # On crée le dossier de destination s'il n'existe pas.
    os.makedirs('data/processed', exist_ok=True)
    
    # Sauvegarde au format Pickle (plus rapide, conserve les types Python)
    # et CSV (format lisible par l'humain pour vérification manuelle).
    output_pickle = 'data/processed/ebay_prep.pkl'
    output_csv = 'data/processed/ebay_prep_debug.csv'
    
    df_final.to_pickle(output_pickle)
    df_final.to_csv(output_csv, index=False)
    
    logging.info(f"✅ Étape 1 terminée avec succès !")
    logging.info(f"📊 Volume final : {df_final.shape[0]:,} produits prêts pour l'entraînement.")
    logging.info(f"💾 Fichiers sauvegardés dans : data/processed/")

if __name__ == "__main__":
    try:
        run_step1_preparation()
    except Exception as e:
        logging.error(f"❌ Erreur critique lors du pipeline : {str(e)}")