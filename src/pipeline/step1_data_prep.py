import pandas as pd
import os
import joblib
import sys

# Add src to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.load_data import load_ebay_data
from src.utils.preprocessing import preprocess_ebay_data
from src.utils.enrichment import enrich_ebay_with_amazon
import pandera.pandas as pa
from src.utils.monitoring import validate_dataframe, ebay_clean_schema, log_step_results

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
    
    # NEW: Also save as CSV for easier inspection
    csv_output_path = 'data/processed/ebay_cleaned_matched.csv'
    df_enriched.to_csv(csv_output_path, index=False)
    
    # NEW: Validate and Log
    validate_dataframe(df_enriched, ebay_clean_schema, "step1_data_prep")
    log_step_results("step1_data_prep", {
        "raw_rows": len(df),
        "cleaned_rows": len(df_clean),
        "final_rows": len(df_enriched),
        "matches_found": df_enriched['matched_asin'].notna().sum()
    })

    print(f"\n💾 Saved PREPARED data to: {output_path}")
    print(f"💾 Saved PREPARED data (CSV) to: {csv_output_path}")
    print(f"   Final data shape: {df_enriched.shape}")

if __name__ == "__main__":
    run_data_preparation()
