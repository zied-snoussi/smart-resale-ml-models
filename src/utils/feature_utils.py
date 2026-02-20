import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

def extract_numerical_features(df):
    """
    Extraction professionnelle de caractéristiques numériques à partir des données eBay.
    Cette étape transforme des métadonnées textuelles et catégorielles en signaux quantitatifs.
    """
    feats = pd.DataFrame(index=df.index)
    
    # 1. Caractéristiques basées sur le Titre
    # La longueur peut indiquer le niveau de détail fourni par le vendeur.
    feats['title_len'] = df['Title'].str.len().fillna(0)
    feats['word_count'] = df['Title'].str.split().str.len().fillna(0)
    
    # 2. Indicateurs binaires (Flags)
    # Détection de mots-clés suggérant des lots (souvent plus chers ou moins précis).
    feats['is_bundle'] = df['Title'].str.contains('bundle|lot|with', case=False, na=False).astype(int)
    # Vérifie si la marque est renseignée, un facteur souvent corrélé à la valeur.
    feats['has_brand'] = df['Manufacturer'].notna().astype(int)
    
    # 3. Signaux numériques issus de l'enrichissement (Étape 1)
    # Intégration du MSRP (prix neuf) et de la demande Amazon comme variables prédictives majeures.
    if 'original_price' in df.columns:
        feats['msrp'] = df['original_price'].fillna(0)
    if 'amazon_demand' in df.columns:
        feats['demand_score'] = df['amazon_demand'].fillna(0)
        
    return feats

def process_text_vectors(text_series, n_components=20):
    """
    Transforme le texte brut en vecteurs sémantiques denses via TF-IDF + SVD.
    Cette méthode est connue sous le nom de Latent Semantic Analysis (LSA).
    """
    # Étape A : TF-IDF pour transformer les mots en fréquences statistiques pondérées.
    # ngram_range=(1, 2) permet de capturer des expressions comme "iPhone 13" ou "Apple Watch".
    tfidf = TfidfVectorizer(
        stop_words='english', 
        max_features=1000, 
        ngram_range=(1, 2)
    )
    tfidf_matrix = tfidf.fit_transform(text_series.fillna(""))
    
    # Étape B : SVD pour la réduction de dimensionnalité.
    # On compresse 1000 colonnes de mots en 'n_components' axes sémantiques essentiels.
    # Cela évite le "fléau de la dimensionnalité" et réduit le bruit.
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd_matrix = svd.fit_transform(tfidf_matrix)
    
    # Conversion en DataFrame pour l'intégration au pipeline
    cols = [f'text_svd_{i}' for i in range(n_components)]
    df_svd = pd.DataFrame(svd_matrix, columns=cols, index=text_series.index)
    
    # Sauvegarde des artefacts (tfidf/svd) pour les réutiliser lors de l'inférence (API)
    artifacts = {'tfidf': tfidf, 'svd': svd}
    return df_svd, artifacts

def scale_features(df):
    """
    Standardise les caractéristiques (moyenne = 0, variance = 1).
    Essentiel pour les modèles de régression et de classification afin qu'une variable 
    à grande échelle (ex: MSRP) n'écrase pas les autres (ex: title_len).
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    
    df_scaled = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
    return df_scaled, scaler