import pandas as pd
import os
import joblib
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.preprocessing import extract_features_ebay, prepare_train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def run_feature_engineering():
    """
    Step 2: Feature Engineering
    - Load cleaned data
    - Extract features (text, numerical, log transforms)
    - Split into Train/Test sets
    - Save prepared datasets
    """
    print("\n" + "🚀"*30)
    print("STEP 2: FEATURE ENGINEERING")
    print("🚀"*30)

    # 1. Load Cleaned Data
    input_path = 'data/processed/ebay_prep.pkl'
        
    if not os.path.exists(input_path):
        print(f"❌ Error: processed data ({input_path}) not found. Run Step 1 first.")
        return

    print(f"\n📥 Loading data from {input_path}...")
    df_clean = pd.read_pickle(input_path)

    # 2. Extract Features
    print("\n⚙️ Extracting features...")
    features = extract_features_ebay(df_clean)
    
    # Add new enriched features if they exist and have enough coverage
    if 'depreciation_pct' in df_clean.columns:
        # Check coverage
        valid_count = df_clean['depreciation_pct'].gt(0).sum()
        if valid_count > 100: # Arbitrary threshold to decide if useful
            print(f"   ℹ️ Including depreciation feature ({valid_count} valid values)")
            features['depreciation_pct'] = df_clean['depreciation_pct']
            # Pass through the new demand feature too
            if 'amazon_demand' in df_clean.columns:
                 features['amazon_demand'] = df_clean['amazon_demand']
        else:
             print(f"   ℹ️ Skipping depreciation feature (only {valid_count} valid values)")

    # 3. ADVANCED TEXT FEATURES (TF-IDF + SVD)
    # We vectorized matching earlier, now let's use vectors for REGRESSION
    print("   🔤 Generating Text Vectors (TF-IDF)...")
    
    # Use 50 components (keep it light)
    tfidf = TfidfVectorizer(max_features=500, stop_words='english')
    svd = TruncatedSVD(n_components=20, random_state=42)
    
    # Fill NaN titles
    titles = df_clean['Title'].fillna("").astype(str)
    
    # Fit & Transform
    tfidf_matrix = tfidf.fit_transform(titles)
    svd_matrix = svd.fit_transform(tfidf_matrix)

    # Save Vectorizers for API inference
    print("   💾 Saving vectorizers...")
    joblib.dump(tfidf, 'models/tfidf.pkl')
    joblib.dump(svd, 'models/svd.pkl')
    
    # Add to features DataFrame
    svd_cols = [f'text_svd_{i}' for i in range(svd_matrix.shape[1])]
    df_svd = pd.DataFrame(svd_matrix, columns=svd_cols, index=features.index)
    features = pd.concat([features, df_svd], axis=1)
    print(f"   ✓ Added {len(svd_cols)} semantic text features")
    
    # Define Target and Features
    # Exclude target variables from X
    # Ensure current_price is in features
    if 'current_price' in features.columns:
        X = features.drop(['current_price', 'log_price'], axis=1, errors='ignore')
        y = features['current_price']
    else:
        # Fallback if cleaner didn't put it in features check original df
        X = features
        y = df_clean['price_cleaned']
    
    print(f"   Features shape: {X.shape}")
    print(f"   Target shape: {y.shape}")
    
    print(f"   Features shape: {X.shape}")
    print(f"   Target shape: {y.shape}")

    # 3. Train/Test Split
    print("\n✂️ Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = prepare_train_test_split(X, y, test_size=0.2)
    
    # 4. Save Split Data
    output_dir = 'data/processed'
    os.makedirs(output_dir, exist_ok=True)
    
    joblib.dump(X_train, f'{output_dir}/X_train.joblib')
    joblib.dump(X_test, f'{output_dir}/X_test.joblib')
    joblib.dump(y_train, f'{output_dir}/y_train.joblib')
    joblib.dump(y_test, f'{output_dir}/y_test.joblib')
    
    print(f"\n💾 Saved split datasets to {output_dir}/")
    print(f"   X_train: {X_train.shape}")
    print(f"   X_test: {X_test.shape}")

if __name__ == "__main__":
    run_feature_engineering()
