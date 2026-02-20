import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

def extract_numerical_features(df):
    """Professional Feature Extraction from eBay data."""
    feats = pd.DataFrame(index=df.index)
    
    # 1. Title Based Features
    feats['title_len'] = df['Title'].str.len().fillna(0)
    feats['word_count'] = df['Title'].str.split().str.len().fillna(0)
    
    # 2. Flag Features (Boolean)
    feats['is_bundle'] = df['Title'].str.contains('bundle|lot|with', case=False, na=False).astype(int)
    feats['has_brand'] = df['Manufacturer'].notna().astype(int)
    
    # 3. Numeric Signals from Enrichment (Step 1)
    # Using MSRP (original_price) and Amazon Demand
    if 'original_price' in df.columns:
        feats['msrp'] = df['original_price'].fillna(0)
    if 'amazon_demand' in df.columns:
        feats['demand_score'] = df['amazon_demand'].fillna(0)
        
    return feats

def process_text_vectors(text_series, n_components=20):
    """Turns raw text into dense semantic vectors using TF-IDF + SVD."""
    # TF-IDF to turn words into numbers
    tfidf = TfidfVectorizer(
        stop_words='english', 
        max_features=1000, 
        ngram_range=(1, 2)
    )
    tfidf_matrix = tfidf.fit_transform(text_series.fillna(""))
    
    # SVD to compress 1000 columns into n_components (Dimensionality Reduction)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd_matrix = svd.fit_transform(tfidf_matrix)
    
    # Convert to DataFrame
    cols = [f'text_svd_{i}' for i in range(n_components)]
    df_svd = pd.DataFrame(svd_matrix, columns=cols, index=text_series.index)
    
    # We return the artifacts so we can save them for the API later
    artifacts = {'tfidf': tfidf, 'svd': svd}
    return df_svd, artifacts

def scale_features(df):
    """Standardizes features to have mean=0 and variance=1."""
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    
    df_scaled = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
    return df_scaled, scaler