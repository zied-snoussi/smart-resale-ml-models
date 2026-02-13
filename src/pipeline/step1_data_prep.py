import pandas as pd
import os
import joblib
import sys

# Add src to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.load_data import load_ebay_data
from src.utils.preprocessing import preprocess_ebay_data

def run_data_preparation():
    """
    Step 1: Data Preparation
    - Load raw data
    - clean data
    - Handle missing values
    - Remove outliers
    - Save cleaned data
    """
    print("\n" + "🚀"*30)
    print("STEP 1: DATA PREPARATION")
    print("🚀"*30)

    # 1. Load Data
    print("\n📥 Loading data...")
    df = load_ebay_data()
    print(f"   Raw data shape: {df.shape}")

    # 2. Preprocess
    print("\n🧹 Cleaning data...")
    df_clean = preprocess_ebay_data(df)
    
    # 3. Save Intermediate Data
    os.makedirs('data/processed', exist_ok=True)
    output_path = 'data/processed/ebay_clean.joblib'
    joblib.dump(df_clean, output_path)
    print(f"\n💾 Saved cleaned data to: {output_path}")
    print(f"   Cleaned data shape: {df_clean.shape}")

if __name__ == "__main__":
    run_data_preparation()
