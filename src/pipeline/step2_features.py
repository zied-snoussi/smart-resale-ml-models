import pandas as pd
import os
import joblib
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.preprocessing import extract_features_ebay, prepare_train_test_split

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
        else:
             print(f"   ℹ️ Skipping depreciation feature (only {valid_count} valid values)")
    
    # Define Target and Features
    # Exclude target variables from X
    X = features.drop(['current_price', 'log_price'], axis=1)
    y = features['current_price']
    
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
