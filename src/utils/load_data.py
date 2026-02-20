import os
import pandas as pd

# This points to the project root (D:\Github\smart-resale-ml-models)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_ebay_data():
    filepath = os.path.join(BASE_DIR, 'data', 'raw', 
                            'marketing_sample_for_ebay_com-ebay_com_product__20200601_20200831__30k_data.csv')
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing: {filepath}")
    return pd.read_csv(filepath)

def load_amazon_products(): # <--- Check this name
    filepath = os.path.join(BASE_DIR, 'data', 'raw', 'amazon_products.csv')
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing: {filepath}")
    return pd.read_csv(filepath)

def load_amazon_categories(): # <--- Check this name
    filepath = os.path.join(BASE_DIR, 'data', 'raw', 'amazon_categories.csv')
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing: {filepath}")
    return pd.read_csv(filepath)