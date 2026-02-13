import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from src.utils.load_data import load_amazon_products, load_amazon_categories

def clean_text_key(text):
    """Normalize text for matching (lowercase, remove special chars)"""
    if not isinstance(text, str):
        return ""
    # Remove contents in parenthesis like (Renewed) or (Refurbished)
    text = re.sub(r'\([^)]*\)', '', text)
    # Important: Keep numbers and letters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text).lower()
    return " ".join(text.split())

def enrich_ebay_with_amazon(df_ebay):
    """
    Match eBay products to Amazon dataset using TF-IDF + Nearest Neighbors
    to find Original Reference Price (MSRP).
    """
    print("\n" + "~"*40)
    print("✨ STARTING DATA ENRICHMENT (SEMANTIC MATCHING)")
    print("~"*40)

    # 1. Load Amazon Data (Reference Catalog)
    print("\n🔍 Loading Amazon Catalog...")
    cats = load_amazon_categories()
    tech_keywords = ['Electronics', 'Computer', 'Phone', 'Tablet', 'Laptop', 'Camera', 'Headphone']
    tech_cats = cats[cats['category_name'].str.contains('|'.join(tech_keywords), case=False, na=False)]
    valid_cat_ids = set(tech_cats['id'].unique())
    
    df_amazon = load_amazon_products()
    # Filter for tech categories to reduce noise
    df_amz = df_amazon[df_amazon['category_id'].isin(valid_cat_ids)].copy()
    
    # Preprocess Amazon Titles
    # We only care about products that have a price (otherwise they are useless for MSRP)
    df_amz = df_amz[df_amz['price'] > 0].reset_index(drop=True)
    df_amz['clean_title'] = df_amz['title'].apply(clean_text_key)
    
    print(f"   Reference Catalog: {len(df_amz):,} Tech Products")

    # 2. Build TF-IDF Search Index
    print("\n⚙️ Building Vector Search Index (TF-IDF)...")
    vectorizer = TfidfVectorizer(min_df=1, analyzer='word', stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df_amz['clean_title'])
    
    # Use NearestNeighbors for fast similarity search
    # Metric: cosine distance (1 - cosine similarity)
    nn_model = NearestNeighbors(n_neighbors=1, metric='cosine', n_jobs=-1)
    nn_model.fit(tfidf_matrix)
    print("   search index built successfully.")

    # 3. Operations on eBay Data
    print("\n🤝 Matching Datasets...")
    
    # Construct 'Query Strings' from eBay data
    # Combining Manufacturer + Model Name gives the cleanest signal
    # Fallback to Title if Model incomplete
    def make_query(row):
        brand = str(row.get('Manufacturer', '')).strip()
        model = str(row.get('Model Name', '')).strip()
        if len(model) > 3 and model.lower() != 'nan':
            return clean_text_key(f"{brand} {model}")
        return clean_text_key(row['Title'])
    
    print("   Generating queries from eBay data...")
    ebay_queries = df_ebay.apply(make_query, axis=1).tolist()
    
    # Transform queries to same vector space
    ebay_tfidf = vectorizer.transform(ebay_queries)
    
    # 4. Find Best Matches
    print("   Running similarity search...")
    distances, indices = nn_model.kneighbors(ebay_tfidf)
    
    # 5. Process Results
    matches_found = 0
    matched_features = {
        'original_price': [],
        'matched_asin': [],
        'match_score': [],
        'match_title': []
    }
    
    # Similarity Threshold (0.0 = identical, 1.0 = different)
    # 0.4 distance ~= 60% similarity. Adjust based on results.
    THRESHOLD_DISTANCE = 0.45 
    
    for i, distance in enumerate(distances):
        dist = distance[0]
        idx = indices[i][0]
        
        if dist < THRESHOLD_DISTANCE:
            # Match found!
            amz_row = df_amz.iloc[idx]
            
            # Determine original price (prefer List Price, fallback to Price)
            orig = amz_row.get('listPrice', 0)
            if orig == 0 or pd.isna(orig):
                orig = amz_row.get('price', 0)
                
            matched_features['original_price'].append(orig)
            matched_features['matched_asin'].append(amz_row['asin'])
            matched_features['match_score'].append(1 - dist) # Convert to similarity score
            matched_features['match_title'].append(amz_row['title'])
            matches_found += 1
        else:
            # No good match
            matched_features['original_price'].append(0)
            matched_features['matched_asin'].append(None)
            matched_features['match_score'].append(0)
            matched_features['match_title'].append(None)

    # 6. assign to dataframe
    df_ebay['original_price'] = matched_features['original_price']
    df_ebay['matched_asin'] = matched_features['matched_asin']
    df_ebay['match_confidence'] = matched_features['match_score']
    
    # Calculate Depreciation
    if 'price_cleaned' not in df_ebay.columns:
         df_ebay['price_cleaned'] = df_ebay['Price'].apply(lambda x: float(re.sub(r'[^\d.]', '', str(x))) if pd.notna(x) else 0)

    mask = (df_ebay['original_price'] > 0)
    df_ebay['depreciation_pct'] = 0.0
    df_ebay.loc[mask, 'depreciation_pct'] = (
        (df_ebay.loc[mask, 'original_price'] - df_ebay.loc[mask, 'price_cleaned']) 
        / df_ebay.loc[mask, 'original_price']
    ).clip(0, 1)

    print(f"\n✅ Enrichment Complete!")
    print(f"   Matches Found: {matches_found:,} ({matches_found/len(df_ebay)*100:.1f}%)")
    
    if matches_found > 0:
        print("\n👀 Sample Semantic Matches:")
        sample = df_ebay[df_ebay['matched_asin'].notna()].head(3)
        for _, row in sample.iterrows():
            print(f"   eBay:   {row['Manufacturer']} {row['Model Name']}")
            print(f"   Amazon: {row['matched_asin']} (Score: {row['match_confidence']:.2f})")
            print(f"   Prices: Used €{row['price_cleaned']} vs New €{row['original_price']}")
            print("-" * 50)
            
    return df_ebay
