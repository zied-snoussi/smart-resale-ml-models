import pandas as pd
import os
import logging
from utils.load_data import load_ebay_data
from utils.preprocessing import preprocess_ebay_data
from utils.enrichment import enrich_ebay_with_amazon

# Setup logging to track the professional process
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def run_step1_preparation():
    """
    STEP 1: DATA PREPARATION
    - Objective: Clean raw data and merge with Amazon MSRP reference.
    """
    logging.info("--- STARTING STEP 1: DATA PREPARATION ---")

    # 1. LOAD RAW DATA
    # Uses your shared utility to keep the pipeline script clean
    df_raw = load_ebay_data()
    
    # 2. CLEANING & OUTLIERS (The "Professional" Clean)
    # This handles price conversion and removes illogical prices (€0 or €1M)
    logging.info("Cleaning raw eBay data and handling outliers...")
    df_clean = preprocess_ebay_data(df_raw)

    # 3. DATA ENRICHMENT (Semantic Matching)
    # This is the "secret sauce": matching eBay items to Amazon MSRP/Demand
    logging.info("Enriching data with Amazon Catalog (TF-IDF Matching)...")
    df_enriched = enrich_ebay_with_amazon(df_clean)

    # 4. FINAL QUALITY CHECK (Sanity Filter)
    # Professional Step: Drop rows where matching failed or data is corrupted
    # We only keep rows where we have a valid price_cleaned
    df_final = df_enriched[df_enriched['price_cleaned'] > 0].copy()
    
    # 5. SAVE PROCESSED ARTIFACTS
    os.makedirs('data/processed', exist_ok=True)
    
    # Save as Pickle (preserves data types) and CSV (for your manual inspection)
    df_final.to_pickle('data/processed/ebay_prep.pkl')
    df_final.to_csv('data/processed/ebay_prep_debug.csv', index=False)
    
    logging.info(f"✅ Step 1 Complete! Final Shape: {df_final.shape}")
    logging.info(f"💾 Saved to data/processed/ebay_prep.pkl")

if __name__ == "__main__":
    run_step1_preparation()