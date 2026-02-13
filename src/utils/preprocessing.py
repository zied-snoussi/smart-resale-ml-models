import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import re

def clean_price(price_str):
    """Extract numeric price from string"""
    if pd.isna(price_str):
        return np.nan
    
    # Remove currency symbols and convert to float
    price_str = str(price_str)
    price_str = re.sub(r'[^\d.]', '', price_str)
    
    try:
        return float(price_str)
    except:
        return np.nan

def remove_outliers(df, column, method='iqr', threshold=3):
    """
    Remove outliers from dataset
    
    Args:
        df: DataFrame
        column: Column name to check for outliers
        method: 'iqr' or 'zscore'
        threshold: Number of IQRs or standard deviations
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
        
        print(f"   Removed {removed:,} outliers (range: €{lower_bound:.2f} - €{upper_bound:.2f})")
    
    elif method == 'zscore':
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        before = len(df)
        df = df[z_scores < threshold]
        removed = before - len(df)
        print(f"   Removed {removed:,} outliers using z-score")
    
    return df

def preprocess_ebay_data(df):
    """Preprocess eBay dataset"""
    print("🔧 Preprocessing eBay data...")
    
    df = df.copy()
    
    # Use correct column names: 'Title' and 'Price'
    required_cols = ['Title', 'Price']
    df = df.dropna(subset=required_cols)
    
    # Clean price column (it's a string in the data)
    df['price_cleaned'] = df['Price'].apply(clean_price)
    df = df[df['price_cleaned'] > 0]
    
    # Remove price outliers (CRITICAL FOR MODEL PERFORMANCE)
    print("   Removing price outliers...")
    # Aggressively remove outliers first (Hard Cap as per user recommendation)
    max_reasonable_price = 5000  # Cap at €5,000 for standard electronics
    before = len(df)
    df = df[df['price_cleaned'] <= max_reasonable_price]
    removed = before - len(df)
    if removed > 0:
        print(f"   Removed {removed:,} products with price > €{max_reasonable_price}")

    # Standard outlier removal after hard cap
    df = remove_outliers(df, 'price_cleaned', method='iqr', threshold=1.5)
    
    # Clean ratings
    df['average_rating'] = pd.to_numeric(df['Average Rating'], errors='coerce').fillna(0)
    df['num_reviews'] = pd.to_numeric(df['Num Of Reviews'], errors='coerce').fillna(0)
    
    print(f"✓ eBay data preprocessed: {len(df):,} rows remaining")
    return df

def preprocess_amazon_data(df):
    """Preprocess Amazon dataset"""
    print("🔧 Preprocessing Amazon data...")
    
    df = df.copy()
    
    # Amazon columns: 'title', 'price', 'stars', 'reviews'
    required_cols = ['title', 'price']
    df = df.dropna(subset=required_cols)
    
    # Price is already numeric in Amazon data
    df['price_cleaned'] = df['price']
    df = df[df['price_cleaned'] > 0]
    
    # Remove outliers
    print("   Removing price outliers...")
    df = remove_outliers(df, 'price_cleaned', method='iqr', threshold=1.5)
    
    # Clean ratings
    df['average_rating'] = df['stars'].fillna(0)
    df['num_reviews'] = df['reviews'].fillna(0)
    
    print(f"✓ Amazon data preprocessed: {len(df):,} rows remaining")
    return df

def extract_features_ebay(df):
    """Extract features for eBay products - for RESALE PRICE PREDICTION"""
    print("🔧 Extracting features for eBay...")
    
    features = pd.DataFrame()
    
    # 1. PRICE FEATURES (Current price as baseline)
    features['current_price'] = df['price_cleaned']
    features['log_price'] = np.log1p(df['price_cleaned'])
    
    # 2. PRODUCT QUALITY INDICATORS
    features['average_rating'] = df['average_rating']
    features['num_reviews'] = df['num_reviews']
    features['log_reviews'] = np.log1p(df['num_reviews'])
    features['has_reviews'] = (df['num_reviews'] > 0).astype(int)
    
    # 3. TEXT FEATURES (Title analysis)
    features['title_length'] = df['Title'].str.len()
    features['word_count'] = df['Title'].str.split().str.len()
    
    # 4. CONDITION INDICATORS (from title keywords)
    features['is_new'] = df['Title'].str.contains('new|New|NEW', na=False).astype(int)
    features['is_used'] = df['Title'].str.contains('used|Used|USED', na=False).astype(int)
    features['is_refurbished'] = df['Title'].str.contains('refurb|Refurb', na=False).astype(int)
    
    # 5. BRAND PRESENCE (Top Brands)
    # Checks specifically for "Premium" brands that hold value
    features['is_apple'] = df['Title'].str.contains(r'Apple|iPhone|iPad|MacBook|AirPods', case=False, na=False).astype(int)
    features['is_samsung'] = df['Title'].str.contains(r'Samsung|Galaxy', case=False, na=False).astype(int)
    features['is_sony'] = df['Title'].str.contains(r'Sony', case=False, na=False).astype(int)
    features['has_brand'] = df['Title'].str.contains('Apple|Samsung|Sony|LG|Dell|HP|Lenovo|Asus|Microsoft|Nintendo', case=False, na=False).astype(int)
    
    # 6. NEGATIVE SENTIMENT (High 1-star ratio = bad resale value)
    if 'One Star' in df.columns and 'Number Of Ratings' in df.columns:
        one_star = pd.to_numeric(df['One Star'], errors='coerce').fillna(0)
        total_ratings = pd.to_numeric(df['Number Of Ratings'], errors='coerce').fillna(1) # avoid div0
        features['negative_ratio'] = (one_star / total_ratings).clip(0, 1)
    else:
        features['negative_ratio'] = 0.0

    # 7. SCREEN SIZE (Regex Extraction from Title if column missing)
    # Extracts "12.9 inch", "15.6"", etc.
    if 'Screen Size' in df.columns:
        features['screen_size'] = pd.to_numeric(
            df['Screen Size'].str.extract(r'(\d+\.?\d*)', expand=False), 
            errors='coerce'
        ).fillna(0)
    
    # Fill missing screen size from title
    mask_no_screen = (features.get('screen_size', 0) == 0)
    scores_extracted = df.loc[mask_no_screen, 'Title'].str.extract(r'(\d+\.?\d*)\s*(?:inch|\"|\'\')', flags=re.IGNORECASE, expand=False)
    features.loc[mask_no_screen, 'screen_size'] = pd.to_numeric(scores_extracted, errors='coerce').fillna(0)
    
    # 8. MEMORY SIZE (Regex Extraction)
    # Extracts "64GB", "256 GB", "1TB"
    if 'Internal Memory' in df.columns:
        features['memory_gb'] = pd.to_numeric(
            df['Internal Memory'].str.extract(r'(\d+)', expand=False), 
            errors='coerce'
        ).fillna(0)
        
    # Fill missing memory from title
    mask_no_mem = (features.get('memory_gb', 0) == 0)
    mem_extracted = df.loc[mask_no_mem, 'Title'].str.extract(r'(\d+)\s*(?:GB|TB|Gigabyte)', flags=re.IGNORECASE, expand=False)
    features.loc[mask_no_mem, 'memory_gb'] = pd.to_numeric(mem_extracted, errors='coerce').fillna(0)
    
    print(f"✓ Extracted {len(features.columns)} features")
    return features

def extract_features_amazon(df):
    """Extract features for Amazon products"""
    print("🔧 Extracting features for Amazon...")
    
    features = pd.DataFrame()
    
    # 1. PRICE FEATURES
    features['current_price'] = df['price_cleaned']
    features['log_price'] = np.log1p(df['price_cleaned'])
    
    # 2. LIST PRICE vs ACTUAL PRICE (discount indicator)
    if 'listPrice' in df.columns:
        features['list_price'] = df['listPrice'].fillna(df['price_cleaned'])
        features['discount_pct'] = ((features['list_price'] - features['current_price']) / 
                                   features['list_price'] * 100).clip(0, 100)
    
    # 3. POPULARITY & DEMAND
    features['stars'] = df['average_rating']
    features['num_reviews'] = df['num_reviews']
    features['log_reviews'] = np.log1p(df['num_reviews'])
    features['is_best_seller'] = df['isBestSeller'].astype(int)
    features['bought_last_month'] = df['boughtInLastMonth'].fillna(0)
    features['log_bought'] = np.log1p(df['boughtInLastMonth'].fillna(0))
    
    # 4. TEXT FEATURES
    features['title_length'] = df['title'].str.len()
    features['word_count'] = df['title'].str.split().str.len()
    
    # 5. CONDITION/QUALITY INDICATORS
    features['has_reviews'] = (df['num_reviews'] > 0).astype(int)
    features['high_rating'] = (df['average_rating'] >= 4.0).astype(int)
    
    print(f"✓ Extracted {len(features.columns)} features")
    return features

def prepare_train_test_split(X, y, test_size=0.2, random_state=42):
    """Split data into train and test sets"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )
    
    print(f"✓ Train set: {len(X_train):,} samples")
    print(f"✓ Test set: {len(X_test):,} samples")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    from load_data import load_ebay_data, load_amazon_products
    
    print("\n" + "="*60)
    print("Testing eBay Preprocessing")
    print("="*60)
    df_ebay = load_ebay_data()
    df_ebay_clean = preprocess_ebay_data(df_ebay)
    features_ebay = extract_features_ebay(df_ebay_clean)
    
    print("\n📊 eBay Features Preview:")
    print(features_ebay.head(3))
    print(f"\nPrice statistics:")
    print(features_ebay['current_price'].describe())