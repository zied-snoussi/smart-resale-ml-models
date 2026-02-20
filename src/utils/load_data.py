import os
import pandas as pd

# ==============================================================================
# CONFIGURATION DES CHEMINS (PATH MANAGEMENT)
# ==============================================================================

# Définition du chemin racine du projet (Project Root)
# On remonte de 3 niveaux depuis le fichier actuel pour atteindre la racine.
# Cette approche permet au code de fonctionner sur n'importe quelle machine (Portable).
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ==============================================================================
# FONCTIONS DE CHARGEMENT (DATA INGESTION)
# ==============================================================================

def load_ebay_data():
    """
    Charge le jeu de données brutes provenant d'eBay.
    
    Returns:
        pd.DataFrame: Les annonces eBay non transformées.
    Raises:
        FileNotFoundError: Si le fichier CSV marketing est absent du dossier data/raw.
    """
    filepath = os.path.join(BASE_DIR, 'data', 'raw', 
                            'marketing_sample_for_ebay_com-ebay_com_product__20200601_20200831__30k_data.csv')
    
    # Vérification défensive : on s'assure que le fichier existe avant de tenter la lecture
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ Fichier eBay introuvable à l'emplacement : {filepath}")
    
    return pd.read_csv(filepath)

def load_amazon_products():
    """
    Charge le catalogue de produits Amazon pour servir de base de référence (MSRP).
    
    Returns:
        pd.DataFrame: Le catalogue Amazon complet.
    """
    filepath = os.path.join(BASE_DIR, 'data', 'raw', 'amazon_products.csv')
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ Catalogue Amazon introuvable : {filepath}")
        
    return pd.read_csv(filepath)

def load_amazon_categories():
    """
    Charge la table de correspondance des catégories Amazon.
    Indispensable pour filtrer uniquement les produits technologiques (Electronics).
    
    Returns:
        pd.DataFrame: Les définitions de catégories.
    """
    filepath = os.path.join(BASE_DIR, 'data', 'raw', 'amazon_categories.csv')
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ Fichier de catégories Amazon introuvable : {filepath}")
        
    return pd.read_csv(filepath)