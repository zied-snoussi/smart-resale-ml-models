import pandas as pd
import os

def load_ebay_data():
    """Load eBay dataset"""
    filepath = os.path.join('data', 'raw', 
                            'marketing_sample_for_ebay_com-ebay_com_product__20200601_20200831__30k_data.csv')
    print(f"Loading eBay data from {filepath}")
    df = pd.read_csv(filepath)
    print(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df

def load_amazon_products():
    """Load Amazon products dataset"""
    filepath = os.path.join('data', 'raw', 'amazon_products.csv')
    print(f"Loading Amazon products from {filepath}")
    df = pd.read_csv(filepath)
    print(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df

def load_amazon_categories():
    """Load Amazon categories dataset"""
    filepath = os.path.join('data', 'raw', 'amazon_categories.csv')
    print(f"Loading Amazon categories from {filepath}")
    df = pd.read_csv(filepath)
    print(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df

def get_combined_data():
    """Combine datasets for training"""
    ebay = load_ebay_data()
    amazon_prod = load_amazon_products()
    
    print("\n" + "="*50)
    print("Dataset Summary")
    print("="*50)
    print(f"eBay columns: {list(ebay.columns)[:5]}...")
    print(f"Amazon columns: {list(amazon_prod.columns)[:5]}...")
    
    return ebay, amazon_prod

if __name__ == "__main__":
    # Test loading
    print("Testing data loading...\n")
    ebay, amazon = get_combined_data()
    
    print("\n📊 eBay Dataset Preview:")
    print(ebay.head(3))
    
    print("\n📊 Amazon Dataset Preview:")
    print(amazon.head(3))