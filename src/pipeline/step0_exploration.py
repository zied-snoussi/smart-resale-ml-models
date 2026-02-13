import pandas as pd
import sys
import os

# Add src to python path to allow importing from utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.load_data import load_ebay_data, load_amazon_products

def print_header(title):
    print("\n" + "🚀"*40)
    print(f"{title}")
    print("🚀"*40 + "\n")

def print_section(title):
    print(f"\n{'='*50}")
    print(f"📊 {title}")
    print(f"{'='*50}")

def analyze_dataset(df, name):
    print_section(f"ANALYZING: {name}")
    
    # 1. Volume
    rows, cols = df.shape
    print(f"\n📦 Data Volume:")
    print(f"   Total Rows:    {rows:,}")
    print(f"   Total Columns: {cols}")
    
    # 2. Features Overview
    print(f"\n📋 Features (Columns):")
    print(f"{'Name':<35} {'Type':<15} {'Missing %'}")
    print("-" * 65)
    
    missing = df.isnull().sum()
    for col in df.columns:
        dtype = str(df[col].dtype)
        missing_count = missing[col]
        missing_pct = (missing_count / rows) * 100
        print(f"{col:<35} {dtype:<15} {missing_pct:.1f}%")

    # 3. Numeric Stats
    print(f"\n📈 Quantitative Statistics:")
    print(df.describe().to_string())
    
    # 4. Sample
    print(f"\n👀 Sample Data (First 3 rows):")
    print(df.head(3).to_string())

def run_exploration():
    """
    Step 0: Data Exploration
    - Overview of raw datasets
    - Volume analysis
    - Feature inspection
    """
    print_header("STEP 0: DATA EXPLORATION & OVERVIEW")

    # 1. Inspect eBay Data
    try:
        df_ebay = load_ebay_data()
        analyze_dataset(df_ebay, "eBay Dataset")
    except Exception as e:
        print(f"❌ Error loading eBay data: {e}")

    # 2. Inspect Amazon Data
    try:
        df_amazon = load_amazon_products()
        analyze_dataset(df_amazon, "Amazon Dataset")
    except Exception as e:
        print(f"❌ Error loading Amazon data: {e}")

if __name__ == "__main__":
    run_exploration()
