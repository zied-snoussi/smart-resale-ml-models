import pandas as pd
import os
import joblib
import sys

# Add src to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.load_data import load_ebay_data
from src.utils.preprocessing import preprocess_ebay_data
from src.utils.enrichment import enrich_ebay_with_amazon

def run_data_preparation():
    """
    Step 1: Data Preparation
    - Load raw data
    - Match & Enrich with Amazon Data (New!)
    - Clean data & Handle outliers
    - Save cleaned data
    """
    print("\n" + "🚀"*30)
    print("STEP 1: DATA PREPARATION (CLEANING + ENRICHMENT)")
    print("🚀"*30)

    # 1. Load Data
    print("\n📥 Loading data...")
    df = load_ebay_data()
    print(f"   Raw data shape: {df.shape}")

    # 2. Preprocess (Cleaning)
    print("\n🧹 Cleaning data...")
    df_clean = preprocess_ebay_data(df)
    
    # 3. Enrich with Amazon Data (Matching)
    # This is now part of the preparation phase
    df_enriched = enrich_ebay_with_amazon(df_clean)
    
    # 4. Save Final Prepared Data
    os.makedirs('data/processed', exist_ok=True)
    
    # Use standard pickle for better compatibility
    output_path = 'data/processed/ebay_prep.pkl'
    df_enriched.to_pickle(output_path)
    
    print(f"\n💾 Saved PREPARED data to: {output_path}")
    print(f"   Final data shape: {df_enriched.shape}")

if __name__ == "__main__":
    run_data_preparation()
