import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import re

def clean_price(price_str):
    """
    Extrait la valeur numérique d'une chaîne de caractères représentant un prix.
    Gère les symboles monétaires et les séparateurs.
    """
    if pd.isna(price_str):
        return np.nan
    
    # Conversion en chaîne et suppression de tout ce qui n'est pas chiffre ou point
    price_str = str(price_str)
    price_str = re.sub(r'[^\d.]', '', price_str)
    
    try:
        return float(price_str)
    except (ValueError, TypeError):
        return np.nan

def remove_outliers(df, column, method='iqr', threshold=3):
    """
    Supprime les valeurs aberrantes (outliers) d'un DataFrame.
    
    Méthodes :
    - 'iqr' : Utilise l'Écart Interquartile (standard pour les distributions non-normales).
    - 'zscore' : Utilise l'écart-type (pour les distributions normales).
    """
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        before = len(df)
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        removed = before - len(df)
        
        print(f"   Nettoyage IQR : {removed:,} outliers supprimés (Plage : €{lower_bound:.2f} - €{upper_bound:.2f})")
    
    elif method == 'zscore':
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        before = len(df)
        df = df[z_scores < threshold]
        removed = before - len(df)
        print(f"   Nettoyage Z-Score : {removed:,} outliers supprimés")
    
    return df

def preprocess_ebay_data(df):
    """Pipeline de prétraitement spécifique aux données eBay."""
    print("🔧 Nettoyage des données eBay...")
    
    df = df.copy()
    
    # Suppression des lignes sans titre ou prix (données critiques manquantes)
    required_cols = ['Title', 'Price']
    df = df.dropna(subset=required_cols)
    
    # Nettoyage de la colonne prix et filtrage des valeurs nulles
    df['price_cleaned'] = df['Price'].apply(clean_price)
    df = df[df['price_cleaned'] > 0]
    
    # Gestion des anomalies de prix (Critique pour la performance du modèle)
    print("   Filtrage des anomalies de prix...")
    # 1. Plafond dur : on limite à 5000€ pour exclure les erreurs de saisie
    max_reasonable_price = 5000 
    before = len(df)
    df = df[df['price_cleaned'] <= max_reasonable_price]
    
    # 2. Suppression statistique des outliers (seuil agressif à 1.5 IQR)
    df = remove_outliers(df, 'price_cleaned', method='iqr', threshold=1.5)
    
    # Normalisation des notations et avis
    df['average_rating'] = pd.to_numeric(df['Average Rating'], errors='coerce').fillna(0)
    df['num_reviews'] = pd.to_numeric(df['Num Of Reviews'], errors='coerce').fillna(0)
    
    print(f"✓ Prétraitement eBay terminé : {len(df):,} lignes conservées")
    return df

def preprocess_amazon_data(df):
    """Pipeline de prétraitement spécifique aux données Amazon."""
    print("🔧 Nettoyage des données Amazon...")
    
    df = df.copy()
    required_cols = ['title', 'price']
    df = df.dropna(subset=required_cols)
    
    df['price_cleaned'] = df['price']
    df = df[df['price_cleaned'] > 0]
    
    # Nettoyage statistique des prix Amazon
    df = remove_outliers(df, 'price_cleaned', method='iqr', threshold=1.5)
    
    # Mapping des colonnes Amazon vers le format standard du projet
    df['average_rating'] = df['stars'].fillna(0)
    df['num_reviews'] = df['reviews'].fillna(0)
    
    print(f"✓ Prétraitement Amazon terminé : {len(df):,} lignes conservées")
    return df

def extract_features_ebay(df):
    """
    Ingénierie des caractéristiques (Feature Engineering) pour eBay.
    Objectif : Extraire des signaux prédictifs pour le prix de revente.
    """
    print("🔧 Extraction des features eBay...")
    
    features = pd.DataFrame()
    
    # 1. Caractéristiques de Prix (Cible et Log-Transformation pour normaliser)
    features['current_price'] = df['price_cleaned']
    features['log_price'] = np.log1p(df['price_cleaned'])
    
    # 2. Indicateurs de Qualité
    features['average_rating'] = df['average_rating']
    features['num_reviews'] = df['num_reviews']
    features['log_reviews'] = np.log1p(df['num_reviews'])
    features['has_reviews'] = (df['num_reviews'] > 0).astype(int)
    
    # 3. Analyse textuelle basique
    features['title_length'] = df['Title'].str.len()
    features['word_count'] = df['Title'].str.split().str.len()
    
    # 4. Extraction de l'état du produit (Mots-clés dans le titre)
    features['is_new'] = df['Title'].str.contains('new|New|NEW', na=False).astype(int)
    features['is_used'] = df['Title'].str.contains('used|Used|USED', na=False).astype(int)
    features['is_refurbished'] = df['Title'].str.contains('refurb|Refurb', na=False).astype(int)
    
    # 5. Présence de marques "Premium" (Fort impact sur la valeur résiduelle)
    premium_pattern = r'Apple|iPhone|iPad|MacBook|AirPods|Samsung|Galaxy|Sony'
    features['is_premium_brand'] = df['Title'].str.contains(premium_pattern, case=False, na=False).astype(int)
    
    # 6. Extraction de la taille d'écran (Regex)
    # Cherche des formats comme "12.9 inch", "15.6\"", "14.1''"
    screen_pattern = r'(\d+\.?\d*)\s*(?:inch|\"|\'\')'
    features['screen_size'] = pd.to_numeric(
        df['Title'].str.extract(screen_pattern, flags=re.IGNORECASE, expand=False), 
        errors='coerce'
    ).fillna(0)
    
    # 7. Extraction de la mémoire (Regex)
    # Cherche des formats comme "64GB", "256 GB", "1TB"
    memory_pattern = r'(\d+)\s*(?:GB|TB|Gigabyte)'
    features['memory_gb'] = pd.to_numeric(
        df['Title'].str.extract(memory_pattern, flags=re.IGNORECASE, expand=False), 
        errors='coerce'
    ).fillna(0)
    
    print(f"✓ Extraction terminée : {len(features.columns)} variables générées")
    return features

def extract_features_amazon(df):
    """Extraction des caractéristiques pour les produits Amazon."""
    print("🔧 Extraction des features Amazon...")
    
    features = pd.DataFrame()
    features['current_price'] = df['price_cleaned']
    features['log_price'] = np.log1p(df['price_cleaned'])
    
    # Indicateur de promotion (Prix catalogue vs Prix actuel)
    if 'listPrice' in df.columns:
        features['list_price'] = df['listPrice'].fillna(df['price_cleaned'])
        features['discount_pct'] = ((features['list_price'] - features['current_price']) / 
                                   features['list_price'] * 100).clip(0, 100)
    
    # Signaux de demande et popularité
    features['stars'] = df['average_rating']
    features['num_reviews'] = df['num_reviews']
    features['is_best_seller'] = df.get('isBestSeller', 0).astype(int)
    features['bought_last_month'] = df.get('boughtInLastMonth', 0).fillna(0)
    
    print(f"✓ Extraction terminée : {len(features.columns)} variables générées")
    return features

def prepare_train_test_split(X, y, test_size=0.2, random_state=42):
    """Découpage des données pour l'entraînement et l'évaluation."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )
    
    print(f"✓ Données prêtes | Entraînement : {len(X_train):,} | Test : {len(X_test):,}")
    return X_train, X_test, y_train, y_test