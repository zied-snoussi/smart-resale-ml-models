import pandas as pd
import os
import joblib
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.preprocessing import extract_features_ebay, prepare_train_test_split
from src.utils.monitoring import validate_dataframe, feature_schema, log_step_results
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
    # ALERT: we keep amazon_demand but EXCLUDE depreciation_pct as it leaks target price
    if 'amazon_demand' in df_clean.columns:
        # Check coverage
        valid_count = df_clean['amazon_demand'].gt(0).sum()
        if valid_count > 100: 
            print(f"   ℹ️ Including amazon_demand feature ({valid_count} valid values)")
            features['amazon_demand'] = df_clean['amazon_demand']
        else:
             print(f"   ℹ️ Skipping amazon_demand feature (only {valid_count} valid values)")

    # Define Target and Features
    # Ensure current_price is in features
    if 'current_price' in features.columns:
        X_raw = features.drop(['current_price', 'log_price'], axis=1, errors='ignore')
        y = features['current_price']
    else:
        # Fallback if cleaner didn't put it in features check original df
        X_raw = features
        y = df_clean['price_cleaned']

    # 3. Train/Test Split (EARLY SPLIT to avoid leakage in Vectorizers)
    print("\n✂️ Splitting data (80/20)...")
    X_train_raw, X_test_raw, y_train, y_test = prepare_train_test_split(X_raw, y, test_size=0.2)
    
    # 4. ADVANCED TEXT FEATURES (TF-IDF + SVD)
    # Fit ONLY on Training data, Transform both
    print("\n🔤 Generating Text Vectors (Fit on Train ONLY)...")
    
    tfidf = TfidfVectorizer(max_features=500, stop_words='english')
    svd = TruncatedSVD(n_components=20, random_state=42)
    
    # Titles corresponding to the split
    train_titles = df_clean.loc[X_train_raw.index, 'Title'].fillna("").astype(str)
    test_titles = df_clean.loc[X_test_raw.index, 'Title'].fillna("").astype(str)
    
    # Fit & Transform Train
    train_tfidf = tfidf.fit_transform(train_titles)
    train_svd = svd.fit_transform(train_tfidf)
    
    # Transform Test (NO FITTING)
    test_tfidf = tfidf.transform(test_titles)
    test_svd = svd.transform(test_tfidf)

    # Save Vectorizers for API inference
    print("   💾 Saving vectorizers...")
    joblib.dump(tfidf, 'models/tfidf.pkl')
    joblib.dump(svd, 'models/svd.pkl')
    
    # Combine with other features
    svd_cols = [f'text_svd_{i}' for i in range(train_svd.shape[1])]
    
    X_train_svd = pd.DataFrame(train_svd, columns=svd_cols, index=X_train_raw.index)
    X_test_svd = pd.DataFrame(test_svd, columns=svd_cols, index=X_test_raw.index)
    
    X_train = pd.concat([X_train_raw, X_train_svd], axis=1)
    X_test = pd.concat([X_test_raw, X_test_svd], axis=1)
    
    print(f"   ✓ Added {len(svd_cols)} semantic text features")
    print(f"   Final Features Shape: {X_train.shape}")

    # 5. Validate and Log
    validate_dataframe(X_train, feature_schema, "step2_features_train")
    log_step_results("step2_features", {
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
        "num_features": X_train.shape[1],
        "brand_coverage": X_train['has_brand'].mean()
    })

    # 6. Save Split Data
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
